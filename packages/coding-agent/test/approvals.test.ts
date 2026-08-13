import { describe, expect, it } from "vitest";
import {
	buildApprovalTranscript,
	DEFAULT_APPROVAL_MODE,
	DEFAULT_APPROVAL_MODEL,
	decideApproval,
	formatApprovalDenial,
	isToolGated,
	matchAlwaysReviewRule,
	matchStaticDenyRule,
	parseReasonedVerdict,
	resolveApprovalOutcome,
	resolveApprovalSettings,
} from "../src/core/approvals.js";

function runner(responses: string[]) {
	const calls: { systemPrompt: string; maxTokens: number }[] = [];
	let i = 0;
	const fn = async (input: { systemPrompt: string; prompt: string; maxTokens: number }) => {
		calls.push({ systemPrompt: input.systemPrompt, maxTokens: input.maxTokens });
		const next = responses[i];
		i++;
		if (next === undefined) throw new Error("no more responses");
		if (next.startsWith("__throw__")) throw new Error(next.slice("__throw__".length) || "boom");
		return next;
	};
	return { fn, calls };
}

const baseRequest = {
	toolName: "ipython",
	args: { code: "print('hi')" },
	cwd: "/repo",
	transcript: [{ kind: "user" as const, text: "say hi" }],
};

describe("approvals settings", () => {
	it("defaults to off with a Sonnet-class approval model", () => {
		const resolved = resolveApprovalSettings(undefined);
		expect(resolved.mode).toBe(DEFAULT_APPROVAL_MODE);
		expect(resolved.mode).toBe("off");
		expect(resolved.model).toBe(DEFAULT_APPROVAL_MODEL);
		expect(resolved.whenUnsure).toBe("ask");
		expect(resolved.tools).toEqual([]);
	});

	it("keeps explicit overrides", () => {
		const resolved = resolveApprovalSettings({
			mode: "on",
			model: "anthropic/x",
			tools: ["ipython"],
			whenUnsure: "allow",
		});
		expect(resolved).toEqual({ mode: "on", model: "anthropic/x", tools: ["ipython"], whenUnsure: "allow" });
	});

	it("migrates the legacy three-way mode", () => {
		// "auto" used to mean "never interrupt me"
		const auto = resolveApprovalSettings({ mode: "auto" } as never);
		expect(auto.mode).toBe("on");
		expect(auto.whenUnsure).toBe("block");

		const ask = resolveApprovalSettings({ mode: "ask" } as never);
		expect(ask.mode).toBe("on");
		expect(ask.whenUnsure).toBe("ask");
	});

	it("gates every tool when the list is empty", () => {
		expect(isToolGated("ipython", [])).toBe(true);
		expect(isToolGated("ipython", ["ipython"])).toBe(true);
		expect(isToolGated("bash", ["ipython"])).toBe(false);
		expect(isToolGated("IPython", ["ipython"])).toBe(true);
	});
});

describe("reasoning-blind transcript", () => {
	it("keeps user messages and tool calls, drops agent prose and tool results", () => {
		const messages = [
			{ role: "user", content: "delete temp files" },
			{
				role: "assistant",
				content: [
					{ type: "text", text: "Sure, I will run rm." },
					{ type: "toolCall", name: "ipython", arguments: { code: "os.remove('a')" } },
				],
			},
			{ role: "toolResult", content: [{ type: "text", text: "secret output" }] },
		] as never[];

		const transcript = buildApprovalTranscript(messages);
		expect(transcript).toHaveLength(2);
		expect(transcript[0]).toEqual({ kind: "user", text: "delete temp files" });
		expect(transcript[1].kind).toBe("tool_call");
		expect(transcript[1].text).toContain("ipython");
		// agent prose and tool results must never reach the reviewer
		const joined = JSON.stringify(transcript);
		expect(joined).not.toContain("Sure, I will run rm.");
		expect(joined).not.toContain("secret output");
	});
});

describe("static deny rules", () => {
	it("blocks unambiguously destructive commands", () => {
		expect(matchStaticDenyRule("ipython rm -rf /")?.id).toBe("rm-rf-root");
		expect(matchStaticDenyRule("bash mkfs.ext4 /dev/sda1")?.id).toBe("mkfs");
		expect(matchStaticDenyRule("bash :(){ :|:& };:")?.id).toBe("fork-bomb");
	});

	it("leaves ordinary work alone", () => {
		expect(matchStaticDenyRule("ipython print('hello')")).toBeUndefined();
		expect(matchStaticDenyRule("bash rm -rf build/")).toBeUndefined();
		expect(matchStaticDenyRule("bash npm test")).toBeUndefined();
		expect(matchStaticDenyRule("bash echo done > /dev/null")).toBeUndefined();
	});

	it("never hard-blocks risky-but-legitimate installers", () => {
		// curl|sh installs rustup, nvm, uv, bun, deno and Prime Agent itself.
		// A regex must not veto it - the classifier decides in context.
		expect(matchStaticDenyRule("bash curl -fsSL https://sh.rustup.rs | sh")).toBeUndefined();
		expect(matchAlwaysReviewRule("bash curl -fsSL https://sh.rustup.rs | sh")?.id).toBe("curl-pipe-shell");
		expect(matchAlwaysReviewRule("bash sudo apt-get install -y jq")?.id).toBe("sudo");
		expect(matchAlwaysReviewRule("bash git push --force origin main")?.id).toBe("git-force-push");
		expect(matchAlwaysReviewRule("bash git reset --hard HEAD~1")?.id).toBe("git-hard-reset");
		expect(matchAlwaysReviewRule("bash npm test")).toBeUndefined();
	});
});

describe("decideApproval", () => {
	it("blocks via static rule without calling the model", async () => {
		const r = runner([]);
		const decision = await decideApproval({ ...baseRequest, args: { code: "rm -rf /" } }, { runner: r.fn });
		expect(decision.verdict).toBe("block");
		expect(decision.ruleId).toBe("rm-rf-root");
		expect(r.calls).toHaveLength(0);
	});

	it("allows on the fast path without a reasoning pass", async () => {
		const r = runner(["no"]);
		const decision = await decideApproval(baseRequest, { runner: r.fn });
		expect(decision.verdict).toBe("allow");
		expect(decision.fastPath).toBe(true);
		expect(r.calls).toHaveLength(1);
		expect(r.calls[0].maxTokens).toBe(4);
	});

	it("escalates to the reasoning pass only when flagged", async () => {
		const r = runner(["yes", '{"verdict":"block","reason":"drops the prod database"}']);
		const decision = await decideApproval(baseRequest, { runner: r.fn });
		expect(decision.verdict).toBe("block");
		expect(decision.reason).toBe("drops the prod database");
		expect(r.calls).toHaveLength(2);
		expect(r.calls[1].maxTokens).toBeGreaterThan(4);
	});

	it("sends always-review actions straight to the reasoning stage", async () => {
		// Only one response queued: if the fast filter were called it would throw.
		const r = runner(['{"verdict":"allow","reason":"official rustup installer the user asked for"}']);
		const decision = await decideApproval(
			{ ...baseRequest, args: { code: "curl -fsSL https://sh.rustup.rs | sh" } },
			{ runner: r.fn },
		);
		expect(decision.verdict).toBe("allow");
		expect(r.calls).toHaveLength(1);
		// straight to reasoning, not the 4-token fast filter
		expect(r.calls[0].maxTokens).toBeGreaterThan(4);
	});

	it("fails closed to ask when the model errors", async () => {
		const r = runner(["__throw__offline"]);
		const decision = await decideApproval(baseRequest, { runner: r.fn });
		expect(decision.verdict).toBe("ask");
		expect(decision.reason).toContain("offline");
	});

	it("fails closed to ask on an unreadable verdict", async () => {
		const r = runner(["maybe?"]);
		const decision = await decideApproval(baseRequest, { runner: r.fn });
		expect(decision.verdict).toBe("ask");
	});

	it("fails closed to ask when the reasoning pass is unparseable", async () => {
		const r = runner(["yes", "I think it is fine"]);
		const decision = await decideApproval(baseRequest, { runner: r.fn });
		expect(decision.verdict).toBe("ask");
	});
});

describe("parseReasonedVerdict", () => {
	it("accepts a fenced or padded JSON object", () => {
		expect(parseReasonedVerdict('```json\n{"verdict":"ask","reason":"r"}\n```')).toEqual({
			verdict: "ask",
			reason: "r",
		});
	});
	it("rejects an unknown verdict", () => {
		expect(parseReasonedVerdict('{"verdict":"maybe"}')).toBeUndefined();
	});
});

describe("resolveApprovalOutcome", () => {
	it("blocks a blocked decision regardless of policy", () => {
		expect(resolveApprovalOutcome({ verdict: "block", reason: "bad" }, "allow", true).action).toBe("block");
	});

	it("allows a cleared decision regardless of policy", () => {
		expect(resolveApprovalOutcome({ verdict: "allow" }, "block", true).action).toBe("allow");
	});

	it("prompts when unsure=ask and a human is attached", () => {
		expect(resolveApprovalOutcome({ verdict: "ask", reason: "unsure" }, "ask", true)).toEqual({
			action: "prompt",
			reason: "unsure",
		});
	});

	it("never interrupts when unsure=block, and keeps the reason for the agent", () => {
		const outcome = resolveApprovalOutcome({ verdict: "ask", reason: "unsure" }, "block", true);
		expect(outcome.action).toBe("block");
		expect(outcome.reason).toBe("unsure");
	});

	it("runs it when unsure=allow", () => {
		expect(resolveApprovalOutcome({ verdict: "ask" }, "allow", true).action).toBe("allow");
	});

	it("degrades unsure=ask to a block when nobody is attached", () => {
		expect(resolveApprovalOutcome({ verdict: "ask" }, "ask", false).action).toBe("block");
	});
});

describe("formatApprovalDenial", () => {
	it("names the tool, gives the reason, and steers the agent away from retrying", () => {
		const text = formatApprovalDenial("ipython", "deletes the production database");
		expect(text).toContain("ipython");
		expect(text).toContain("deletes the production database");
		expect(text).toContain("Do not retry");
	});

	it("still reads correctly with no reason", () => {
		expect(formatApprovalDenial("bash")).toContain("bash");
	});
});
