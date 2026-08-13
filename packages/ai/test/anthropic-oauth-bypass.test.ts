import type { MessageCreateParamsStreaming, MessageParam } from "@anthropic-ai/sdk/resources/messages.js";
import { describe, expect, it } from "vitest";
import {
	applyClaudeCodeOAuthTransform,
	buildBillingHeaderValue,
	buildClaudeCodeCompactSystemPrompt,
	CLAUDE_CODE_COMPACT_SYSTEM_PROMPT,
	CLAUDE_CODE_ENTRYPOINT,
	CLAUDE_CODE_SYSTEM_PROMPT_COMPACT_CHARS,
	computeCch,
	computeVersionSuffix,
	extractFirstUserMessageText,
	formatClaudeCodeHostContext,
	getClaudeCodeBetaOverrides,
	needsClaudeCodeSystemCompaction,
	prefixToolName,
	SYSTEM_IDENTITY,
	unprefixToolName,
} from "../src/providers/anthropic-oauth-bypass.js";

const VERSION = "2.1.75";

describe("anthropic-oauth-bypass signing", () => {
	it("computes the cch as the first 5 hex chars of SHA-256 of the text", () => {
		expect(computeCch("Hello there")).toBe("4e478");
	});

	it("computes the version suffix sampling chars at 4,7,20", () => {
		expect(computeVersionSuffix("Hello there", VERSION)).toBe("ad1");
	});

	it("pads short messages with zeros when sampling", () => {
		expect(computeVersionSuffix("hi", VERSION)).toBe("b84");
		expect(computeCch("hi")).toBe("8f434");
	});

	it("extracts text from the first user message", () => {
		const stringMsg: MessageParam[] = [{ role: "user", content: "Hello there" }];
		expect(extractFirstUserMessageText(stringMsg)).toBe("Hello there");

		// Non-user messages are skipped when locating the first user message.
		const blockMsg: MessageParam[] = [
			{ role: "assistant", content: "Let me think" },
			{ role: "user", content: [{ type: "text", text: "Hi" }] },
		];
		expect(extractFirstUserMessageText(blockMsg)).toBe("Hi");

		const imageOnlyMsg: MessageParam[] = [
			{
				role: "user",
				content: [{ type: "image", source: { type: "base64", media_type: "image/png", data: "x" } }],
			},
		];
		expect(extractFirstUserMessageText(imageOnlyMsg)).toBe("");
	});

	it("builds the full billing header value", () => {
		const messages: MessageParam[] = [{ role: "user", content: "Hello there" }];
		const value = buildBillingHeaderValue(messages, VERSION, CLAUDE_CODE_ENTRYPOINT);
		expect(value).toBe("x-anthropic-billing-header: cc_version=2.1.75.ad1; cc_entrypoint=sdk-cli; cch=4e478;");
	});
});

describe("applyClaudeCodeOAuthTransform", () => {
	it("injects the billing header as system[0] without cache_control and keeps identity", () => {
		const params: MessageCreateParamsStreaming = {
			model: "claude-sonnet-4-6",
			max_tokens: 1024,
			stream: true,
			messages: [{ role: "user", content: "Hello there" }],
			system: [
				{ type: "text", text: SYSTEM_IDENTITY, cache_control: { type: "ephemeral" } },
				{ type: "text", text: "You are Prime Agent.", cache_control: { type: "ephemeral" } },
			],
		};

		applyClaudeCodeOAuthTransform(params, VERSION);

		expect(params.system).toHaveLength(2);
		expect(params.system?.[0]).toMatchObject({
			type: "text",
			text: "x-anthropic-billing-header: cc_version=2.1.75.ad1; cc_entrypoint=sdk-cli; cch=4e478;",
		});
		expect(params.system?.[0]).not.toHaveProperty("cache_control");
		expect(params.system?.[1]).toMatchObject({ type: "text", text: SYSTEM_IDENTITY });

		// The app's own system prompt is prepended as plain text to the first
		// user message (matching upstream, no <system-reminder> wrapper).
		const firstUser = params.messages?.[0] as MessageParam;
		expect(typeof firstUser.content).toBe("string");
		const contentText = firstUser.content as string;
		expect(contentText.startsWith("You are Prime Agent.")).toBe(true);
		expect(contentText).toContain("Hello there");
	});

	it("prepends a synthetic identity when none is present", () => {
		const params: MessageCreateParamsStreaming = {
			model: "claude-sonnet-4-6",
			max_tokens: 1024,
			stream: true,
			messages: [{ role: "user", content: "Go" }],
			system: [{ type: "text", text: "Only app prompt" }],
		};

		applyClaudeCodeOAuthTransform(params, VERSION);

		expect(params.system).toHaveLength(2);
		expect(params.system?.[1]).toMatchObject({ type: "text", text: SYSTEM_IDENTITY });
	});

	it("drops stale billing headers and duplicate identity entries", () => {
		const params: MessageCreateParamsStreaming = {
			model: "claude-sonnet-4-6",
			max_tokens: 1024,
			stream: true,
			messages: [{ role: "user", content: "Go" }],
			system: [
				{ type: "text", text: "x-anthropic-billing-header: cc_version=1.0; cc_entrypoint=cli; cch=zzzzz;" },
				{ type: "text", text: SYSTEM_IDENTITY },
				{ type: "text", text: SYSTEM_IDENTITY },
				{ type: "text", text: "App prompt" },
			],
		};

		applyClaudeCodeOAuthTransform(params, VERSION);

		// billing header + exactly one identity
		expect(params.system).toHaveLength(2);
		expect(params.system?.[1]).toMatchObject({ type: "text", text: SYSTEM_IDENTITY });
		expect((params.system?.[0] as { text: string }).text).not.toContain("cc_version=1.0");
	});

	it("strips effort for haiku", () => {
		const params = {
			model: "claude-haiku-4-5",
			max_tokens: 1024,
			stream: true,
			messages: [{ role: "user", content: "Hi" }],
			output_config: { effort: "high" } as MessageCreateParamsStreaming["output_config"],
			thinking: {
				type: "adaptive",
				display: "summarized",
				effort: "high",
			} as unknown as MessageCreateParamsStreaming["thinking"],
		} as MessageCreateParamsStreaming;

		applyClaudeCodeOAuthTransform(params, VERSION);

		expect(params).not.toHaveProperty("output_config");
		// Effort is stripped but the thinking block shape is preserved.
		expect(params.thinking).toEqual({ type: "adaptive", display: "summarized" });
	});

	it("keeps adaptive thinking for haiku when there is no effort key", () => {
		const params = {
			model: "claude-haiku-4-5",
			max_tokens: 1024,
			stream: true,
			messages: [{ role: "user", content: "Hi" }],
			thinking: { type: "adaptive", display: "summarized" } as MessageCreateParamsStreaming["thinking"],
		} as MessageCreateParamsStreaming;

		applyClaudeCodeOAuthTransform(params, VERSION);

		expect(params.thinking).toEqual({ type: "adaptive", display: "summarized" });
	});

	it("keeps effort for non-haiku adaptive models", () => {
		const params = {
			model: "claude-opus-4-6",
			max_tokens: 1024,
			stream: true,
			messages: [{ role: "user", content: "Hi" }],
			output_config: { effort: "max" } as MessageCreateParamsStreaming["output_config"],
		} as MessageCreateParamsStreaming;

		applyClaudeCodeOAuthTransform(params, VERSION);

		expect(params.output_config).toEqual({ effort: "max" });
	});

	it("is a no-op when there are no messages", () => {
		const params: MessageCreateParamsStreaming = {
			model: "claude-sonnet-4-6",
			max_tokens: 1024,
			stream: true,
			messages: [],
			system: [{ type: "text", text: SYSTEM_IDENTITY }],
		};

		applyClaudeCodeOAuthTransform(params, VERSION);

		expect(params.system).toEqual([{ type: "text", text: SYSTEM_IDENTITY }]);
	});
});

describe("getClaudeCodeBetaOverrides", () => {
	it("excludes the effort beta for sonnet and haiku", () => {
		expect(getClaudeCodeBetaOverrides("claude-sonnet-4-6")).toEqual({ add: [], remove: ["effort-2025-11-24"] });
		expect(getClaudeCodeBetaOverrides("claude-haiku-4-5")).toEqual({ add: [], remove: ["effort-2025-11-24"] });
	});

	it("adds the effort beta for opus 4.6/4.7", () => {
		expect(getClaudeCodeBetaOverrides("claude-opus-4-6")).toEqual({ add: ["effort-2025-11-24"], remove: [] });
		expect(getClaudeCodeBetaOverrides("claude-opus-4-7")).toEqual({ add: ["effort-2025-11-24"], remove: [] });
	});

	it("returns no overrides for other models", () => {
		expect(getClaudeCodeBetaOverrides("claude-opus-4-5")).toEqual({ add: [], remove: [] });
	});
});

describe("applyClaudeCodeOAuthTransform tool pair repair", () => {
	it("synthesizes an adjacent placeholder tool_result for an orphaned tool_use", () => {
		const params = {
			model: "claude-sonnet-4-6",
			max_tokens: 1024,
			stream: true,
			messages: [
				{ role: "user", content: "Do a thing" },
				{
					role: "assistant",
					content: [
						{ type: "text", text: "Calling tool" },
						{ type: "tool_use", id: "toolu_01orphan", name: "Bash", input: { command: "ls" } },
					],
				},
			],
		} as MessageCreateParamsStreaming;

		applyClaudeCodeOAuthTransform(params, VERSION);

		const messages = params.messages as MessageParam[];
		const last = messages[messages.length - 1];
		expect(last.role).toBe("user");
		expect(last.content).toMatchObject([
			{
				type: "tool_result",
				tool_use_id: "toolu_01orphan",
				content: "Tool result unavailable (removed during context compaction).",
				is_error: true,
			},
		]);
	});

	it("leaves valid tool pairs untouched", () => {
		const params = {
			model: "claude-sonnet-4-6",
			max_tokens: 1024,
			stream: true,
			messages: [
				{ role: "user", content: "Use bash" },
				{
					role: "assistant",
					content: [{ type: "tool_use", id: "toolu_01ok", name: "Bash", input: { command: "ls" } }],
				},
				{ role: "user", content: [{ type: "tool_result", tool_use_id: "toolu_01ok", content: "file.txt" }] },
			],
		} as MessageCreateParamsStreaming;

		const before = JSON.stringify(params.messages);
		applyClaudeCodeOAuthTransform(params, VERSION);
		expect(JSON.stringify(params.messages)).toBe(before);
	});
});

describe("tool name prefixing", () => {
	it("prefixes and uppercases the first character", () => {
		expect(prefixToolName("ipython")).toBe("mcp_Ipython");
		expect(prefixToolName("Bash")).toBe("mcp_Bash");
		expect(prefixToolName("read")).toBe("mcp_Read");
	});

	it("is idempotent (skips already-prefixed names)", () => {
		expect(prefixToolName("mcp_Ipython")).toBe("mcp_Ipython");
	});

	it("unprefixes and restores the original leading case", () => {
		expect(unprefixToolName("mcp_Ipython")).toBe("ipython");
		expect(unprefixToolName("mcp_Bash")).toBe("bash");
		expect(unprefixToolName("mcp_Read")).toBe("read");
		expect(unprefixToolName("Bash")).toBe("Bash");
	});
});

describe("system prompt compaction", () => {
	it("applies to Opus 5 and Fable 5", () => {
		expect(needsClaudeCodeSystemCompaction("claude-opus-5")).toBe(true);
		expect(needsClaudeCodeSystemCompaction("claude-opus-5-preview")).toBe(true);
		expect(needsClaudeCodeSystemCompaction("claude-fable-5")).toBe(true);
		expect(needsClaudeCodeSystemCompaction("claude-fable-5-preview")).toBe(true);
	});

	it("excludes Sonnet 5", () => {
		expect(needsClaudeCodeSystemCompaction("claude-sonnet-5")).toBe(false);
		expect(needsClaudeCodeSystemCompaction("claude-sonnet-5-preview")).toBe(false);
	});

	it("excludes other models", () => {
		expect(needsClaudeCodeSystemCompaction("claude-opus-4-8")).toBe(false);
		expect(needsClaudeCodeSystemCompaction("claude-sonnet-4-6")).toBe(false);
		expect(needsClaudeCodeSystemCompaction("claude-haiku-4-5")).toBe(false);
		expect(needsClaudeCodeSystemCompaction(undefined)).toBe(false);
	});

	it("preserves a markerless custom prompt when it fits", () => {
		const customPrompt = "CUSTOM SYSTEM RULE\nUse the repository-native test command.";
		const compactSystemPrompt = buildClaudeCodeCompactSystemPrompt(customPrompt);

		expect(compactSystemPrompt).toContain(customPrompt);
		expect(compactSystemPrompt).toContain("# Claude OAuth runtime contract");
		expect(compactSystemPrompt.length).toBeLessThan(CLAUDE_CODE_SYSTEM_PROMPT_COMPACT_CHARS);
	});

	it("bounds an oversized markerless core and keeps both ends", () => {
		const oversized = `CORE START\n${"middle detail\n".repeat(2_000)}CORE END`;
		const compactSystemPrompt = buildClaudeCodeCompactSystemPrompt(oversized);

		expect(compactSystemPrompt.length).toBeLessThan(CLAUDE_CODE_SYSTEM_PROMPT_COMPACT_CHARS);
		expect(compactSystemPrompt).toContain("CORE START");
		expect(compactSystemPrompt).toContain("CORE END");
		expect(compactSystemPrompt).toContain("core excerpt shortened");
	});
});

describe("applyClaudeCodeOAuthTransform with keepSystemInPlace", () => {
	it("keeps the app system prompt in system[] instead of the user message", () => {
		const params: MessageCreateParamsStreaming = {
			model: "claude-opus-5",
			max_tokens: 1024,
			stream: true,
			messages: [{ role: "user", content: "Hi" }],
			system: [
				{ type: "text", text: SYSTEM_IDENTITY },
				{ type: "text", text: "Compact app prompt." },
			],
		};
		applyClaudeCodeOAuthTransform(params, VERSION, true);
		expect(params.system).toHaveLength(3);
		expect((params.system?.[0] as { text: string }).text.startsWith("x-anthropic-billing-header: cc_version=")).toBe(
			true,
		);
		expect(params.system?.[1]).toMatchObject({ type: "text", text: SYSTEM_IDENTITY });
		expect(params.system?.[2]).toMatchObject({ type: "text", text: "Compact app prompt." });
		expect(params.messages?.[0]).toEqual({ role: "user", content: "Hi" });
	});

	it("keeps a compact stable system prompt and relocates the complete host context", () => {
		const primeCore = [
			"You are a general purpose agent that uses code to solve tasks.",
			"IPython is the agent's long-lived notebook.",
			"Use %%bash as the first line of a shell cell.",
			"await rlm('sub-task') returns immediately and never returns the child's answer.",
			"# Delegating to sub-agents",
			"Delegate independent context-heavy work.",
			"RLM CORE DETAIL\n".repeat(500),
		].join("\n");
		const hostContext = [
			primeCore,
			"# Continual Harness State",
			"recent refinements: 4",
			"# Project Context",
			"## /repo/AGENTS.md",
			"NEVER run: npm run dev",
			"The following skills provide specialized instructions for specific tasks.",
			"<name>deepwiki</name>",
			"APPENDED SYSTEM INSTRUCTION",
			"x".repeat(50_000),
		].join("\n");
		const compactSystemPrompt = buildClaudeCodeCompactSystemPrompt(hostContext);
		const params: MessageCreateParamsStreaming = {
			model: "claude-opus-5",
			max_tokens: 1024,
			stream: true,
			messages: [{ role: "user", content: "Hi" }],
			system: [
				{ type: "text", text: SYSTEM_IDENTITY },
				{ type: "text", text: compactSystemPrompt },
			],
		};

		applyClaudeCodeOAuthTransform(params, VERSION, true, hostContext);

		expect(CLAUDE_CODE_COMPACT_SYSTEM_PROMPT.length).toBeGreaterThan(2_000);
		expect(compactSystemPrompt.length).toBeGreaterThan(10_000);
		expect(compactSystemPrompt.length).toBeLessThan(CLAUDE_CODE_SYSTEM_PROMPT_COMPACT_CHARS);
		expect(compactSystemPrompt).toContain(primeCore);
		expect(compactSystemPrompt).toContain("IPython is the agent's long-lived notebook");
		expect(compactSystemPrompt).toContain("Use %%bash as the first line");
		expect(compactSystemPrompt).toContain("never returns the child's answer");
		expect(params.system?.[0]).toMatchObject({
			text: buildBillingHeaderValue([{ role: "user", content: "Hi" }], VERSION, CLAUDE_CODE_ENTRYPOINT),
		});
		expect(params.system?.[2]).toMatchObject({ type: "text", text: compactSystemPrompt });
		const content = params.messages?.[0]?.content;
		expect(typeof content).toBe("string");
		expect(content).toBe(`${formatClaudeCodeHostContext(hostContext)}\n\nHi`);
		expect(content).toContain("recent refinements: 4");
		expect(content).toContain("NEVER run: npm run dev");
		expect(content).toContain("<name>deepwiki</name>");
		expect(content).toContain("APPENDED SYSTEM INSTRUCTION");
		expect(content).toContain("x".repeat(50_000));
	});
});
