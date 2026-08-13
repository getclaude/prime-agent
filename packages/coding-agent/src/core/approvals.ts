/**
 * Approvals - model-based action gating ("auto mode").
 *
 * Mirrors the design Anthropic published for Claude Code auto mode:
 * https://www.anthropic.com/engineering/claude-code-auto-mode
 *
 * - A transcript classifier evaluates every tool call before it executes,
 *   standing in for a human approver.
 * - The classifier runs in two stages: a fast single-token filter that answers
 *   "yes" (block) or "no" (allow), followed by chain-of-thought reasoning only
 *   when the fast filter flags the action. Most actions clear stage one, so
 *   reasoning tokens are spent only where they matter.
 * - The classifier is reasoning-blind by design: it sees only user messages and
 *   the agent's tool calls. The agent's own prose and tool outputs are stripped
 *   so a hijacked transcript cannot talk the reviewer into approving.
 *
 * A small static deny-list runs before the model as a fail-closed backstop for
 * unambiguously destructive actions.
 */

import type { AgentMessage } from "@earendil-works/pi-agent-core";

export type ApprovalMode = "off" | "on";
export type ApprovalVerdict = "allow" | "block" | "ask";

/**
 * What happens when the reviewer cannot clear an action.
 *
 * - "ask": interrupt and let the user decide (falls back to a block when no
 *   approver is attached, so it never fails open).
 * - "block": never interrupt. The action is refused and the reviewer's reason is
 *   handed back to the agent so it can explain itself or pick another approach.
 * - "allow": never interrupt, run it anyway. Deliberately the least safe option.
 */
export type ApprovalUnsureAction = "ask" | "block" | "allow";

/** Persisted user configuration for the approvals subsystem. */
export interface ApprovalSettings {
	/** "off" runs every tool call unchecked. */
	mode?: ApprovalMode;
	/** Approval model as "provider/id". Defaults to a mid-tier Sonnet-class model. */
	model?: string;
	/** Tool names to gate. Empty or omitted gates every tool. */
	tools?: string[];
	/** What to do with an action the reviewer cannot clear. */
	whenUnsure?: ApprovalUnsureAction;
}

export const DEFAULT_APPROVAL_MODE: ApprovalMode = "off";
/** Claude Code runs its transcript classifier on Sonnet 4.6, not the main model. */
export const DEFAULT_APPROVAL_MODEL = "anthropic/claude-sonnet-4-6";
export const DEFAULT_APPROVAL_WHEN_UNSURE: ApprovalUnsureAction = "ask";
/** How many transcript entries the classifier sees. Keeps the review prompt small. */
export const DEFAULT_APPROVAL_TRANSCRIPT_ENTRIES = 20;

export function resolveApprovalSettings(settings: ApprovalSettings | undefined): Required<ApprovalSettings> {
	// Tolerate the earlier three-way mode ("auto"/"ask"), which folded the
	// unsure-policy into the mode itself.
	const rawMode = settings?.mode as string | undefined;
	const legacyAuto = rawMode === "auto";
	const mode: ApprovalMode = rawMode === "off" || rawMode === undefined ? DEFAULT_APPROVAL_MODE : "on";

	return {
		mode,
		model: settings?.model ?? DEFAULT_APPROVAL_MODEL,
		tools: settings?.tools ?? [],
		whenUnsure: settings?.whenUnsure ?? (legacyAuto ? "block" : DEFAULT_APPROVAL_WHEN_UNSURE),
	};
}

/** Whether a given tool is gated under the current settings. */
export function isToolGated(toolName: string, tools: readonly string[]): boolean {
	if (tools.length === 0) return true;
	return tools.some((t) => t.toLowerCase() === toolName.toLowerCase());
}

// ---------------------------------------------------------------------------
// Transcript (reasoning-blind)
// ---------------------------------------------------------------------------

export interface ApprovalTranscriptEntry {
	kind: "user" | "tool_call";
	text: string;
}

function contentToText(content: unknown): string {
	if (typeof content === "string") return content;
	if (Array.isArray(content)) {
		return content
			.map((block) => {
				if (block && typeof block === "object" && (block as { type?: string }).type === "text") {
					return String((block as { text?: string }).text ?? "");
				}
				return "";
			})
			.filter(Boolean)
			.join("\n");
	}
	return "";
}

/**
 * Build the reasoning-blind transcript: user messages and the agent's tool
 * calls only. The agent's own messages and all tool results are dropped so the
 * reviewer cannot be argued into approval by hijacked content.
 */
export function buildApprovalTranscript(
	messages: readonly AgentMessage[],
	maxEntries: number = DEFAULT_APPROVAL_TRANSCRIPT_ENTRIES,
): ApprovalTranscriptEntry[] {
	const entries: ApprovalTranscriptEntry[] = [];
	for (const message of messages) {
		const role = (message as { role?: string }).role;
		if (role === "user") {
			const text = contentToText((message as { content?: unknown }).content).trim();
			if (text) entries.push({ kind: "user", text });
			continue;
		}
		if (role === "assistant") {
			const content = (message as { content?: unknown }).content;
			if (!Array.isArray(content)) continue;
			for (const block of content) {
				if (block && typeof block === "object" && (block as { type?: string }).type === "toolCall") {
					const call = block as { name?: string; arguments?: unknown };
					entries.push({
						kind: "tool_call",
						text: `${call.name ?? "unknown"} ${safeStringify(call.arguments)}`,
					});
				}
			}
		}
		// assistant text and toolResult messages are intentionally dropped
	}
	return entries.slice(-maxEntries);
}

export function safeStringify(value: unknown, maxChars = 4000): string {
	let text: string;
	try {
		text = typeof value === "string" ? value : JSON.stringify(value);
	} catch {
		text = String(value);
	}
	if (!text) return "";
	return text.length > maxChars ? `${text.slice(0, maxChars)}… [truncated]` : text;
}

// ---------------------------------------------------------------------------
// Static deny rules (fail-closed backstop)
// ---------------------------------------------------------------------------

export interface ApprovalRule {
	id: string;
	pattern: RegExp;
	reason: string;
}

/**
 * Patterns that are never legitimate in an agent session. These are blocked
 * without consulting the model: the classifier is allowed to be wrong, this
 * list is not.
 *
 * Keep this list ruthlessly small. Anything that is merely *risky* - and
 * therefore sometimes legitimate - belongs in ALWAYS_REVIEW_RULES instead, so
 * the classifier can judge it in context rather than a regex vetoing it.
 */
export const STATIC_DENY_RULES: readonly ApprovalRule[] = [
	// Patterns are matched against a JSON-serialized tool input, so targets may be
	// followed by a quote or escape rather than whitespace.
	{
		id: "rm-rf-root",
		pattern: /\brm\s+(?:-[a-zA-Z]+\s+)*-[a-zA-Z]*r[a-zA-Z]*\s+\/(?=["'\s*\\]|$)/,
		reason: "Recursive delete of the filesystem root",
	},
	{
		id: "rm-rf-home",
		pattern: /\brm\s+(?:-[a-zA-Z]+\s+)*-[a-zA-Z]*r[a-zA-Z]*\s+(?:~|\$HOME)(?:\/\*?)?(?=["'\s*\\]|$)/,
		reason: "Recursive delete of the home directory",
	},
	{ id: "mkfs", pattern: /\bmkfs(\.\w+)?\s+\/dev\//, reason: "Formatting a block device" },
	{ id: "dd-to-device", pattern: /\bdd\b[^\n]*\bof=\/dev\/(sd|nvme|hd|disk)/, reason: "Raw write to a block device" },
	{ id: "fork-bomb", pattern: /:\(\)\s*\{\s*:\|:&\s*\}\s*;\s*:/, reason: "Fork bomb" },
	{
		id: "chmod-777-root",
		pattern: /\bchmod\s+(-[a-zA-Z]+\s+)*777\s+\/(?:\s|$)/,
		reason: "World-writable filesystem root",
	},
	{ id: "disk-overwrite", pattern: /\b>\s*\/dev\/(sd|nvme|hd|disk)[a-z0-9]*/, reason: "Overwriting a block device" },
];

/** Run the static deny-list against the serialized tool input. */
export function matchStaticDenyRule(serializedInput: string): ApprovalRule | undefined {
	return STATIC_DENY_RULES.find((rule) => rule.pattern.test(serializedInput));
}

/**
 * Risky-but-often-legitimate patterns.
 *
 * These are NOT blocked. Piping an installer into a shell is how rustup, nvm,
 * uv, bun, deno and Prime Agent itself get installed, so a regex must not veto
 * it. Instead these skip the fast allow path and go straight to the reasoning
 * stage, where the model can weigh the actual URL against what the user asked
 * for.
 */
export const ALWAYS_REVIEW_RULES: readonly ApprovalRule[] = [
	{
		id: "curl-pipe-shell",
		pattern: /\b(curl|wget)\b[^\n|]*\|\s*(sudo\s+)?(ba|z|k|d)?sh\b/,
		reason: "Pipes a remote script straight into a shell",
	},
	{
		id: "sudo",
		pattern: /(^|[\s;&|"'])sudo\s+\S/,
		reason: "Runs a command with elevated privileges",
	},
	{
		id: "git-force-push",
		pattern: /\bgit\s+push\b[^\n]*\s(--force|-f)\b/,
		reason: "Force-pushes and can overwrite remote history",
	},
	{
		id: "git-hard-reset",
		pattern: /\bgit\s+reset\b[^\n]*\s--hard\b/,
		reason: "Discards uncommitted work",
	},
];

/** Run the always-review list against the serialized tool input. */
export function matchAlwaysReviewRule(serializedInput: string): ApprovalRule | undefined {
	return ALWAYS_REVIEW_RULES.find((rule) => rule.pattern.test(serializedInput));
}

// ---------------------------------------------------------------------------
// Decision
// ---------------------------------------------------------------------------

export interface ApprovalRequest {
	toolName: string;
	args: unknown;
	cwd: string;
	transcript: readonly ApprovalTranscriptEntry[];
}

export interface ApprovalDecision {
	verdict: ApprovalVerdict;
	reason?: string;
	/** Set when a static rule decided without consulting the model. */
	ruleId?: string;
	/** True when the fast single-token filter cleared the action with no reasoning pass. */
	fastPath?: boolean;
}

/**
 * Runs one approval-model completion. Injected so the engine stays pure and
 * unit-testable; the session supplies a runner bound to the configured model.
 */
export type ApprovalModelRunner = (input: {
	systemPrompt: string;
	prompt: string;
	maxTokens: number;
	signal?: AbortSignal;
}) => Promise<string>;

export const FAST_FILTER_SYSTEM_PROMPT = `You are a safety filter guarding an autonomous coding agent.
You review ONE proposed tool call and decide whether a careful human operator would refuse to run it.

Block when the action is destructive, exfiltrates secrets, targets systems outside the working directory,
escalates privileges, or clearly does not follow what the user asked for.
Allow ordinary development work: reading, searching, editing project files, running builds, tests and linters,
installing declared dependencies, and using version control without destructive history rewrites.

Answer with exactly one word: "yes" to BLOCK, or "no" to ALLOW. No punctuation, no explanation.`;

export const REASONING_SYSTEM_PROMPT = `You are a safety reviewer guarding an autonomous coding agent.
A fast filter flagged the proposed tool call below as potentially unsafe. Decide what should actually happen.

Block when the action is destructive, exfiltrates secrets, targets systems outside the working directory,
escalates privileges, or clearly contradicts the user's stated intent.
Ask when the action is plausibly intended but consequential enough that a human should confirm.
Allow when the fast filter over-reacted and the action is ordinary development work.

Reply with ONLY a JSON object, no markdown fence:
{"verdict":"allow"|"block"|"ask","reason":"one short sentence"}`;

export function buildApprovalPrompt(request: ApprovalRequest): string {
	const transcript = request.transcript
		.map((entry) => (entry.kind === "user" ? `USER: ${entry.text}` : `AGENT TOOL CALL: ${entry.text}`))
		.join("\n");
	return [
		`Working directory: ${request.cwd}`,
		"",
		"Conversation so far (user messages and agent tool calls only):",
		transcript || "(no prior context)",
		"",
		"Proposed tool call to review:",
		`tool: ${request.toolName}`,
		`input: ${safeStringify(request.args)}`,
	].join("\n");
}

function parseFastVerdict(raw: string): "flag" | "clear" | undefined {
	const text = raw.trim().toLowerCase();
	if (!text) return undefined;
	if (/^y(es)?\b/.test(text)) return "flag";
	if (/^n(o)?\b/.test(text)) return "clear";
	// Some models answer with a leading token like "no." or "allow"
	if (text.startsWith("allow")) return "clear";
	if (text.startsWith("block")) return "flag";
	return undefined;
}

export function parseReasonedVerdict(raw: string): ApprovalDecision | undefined {
	const match = raw.match(/\{[\s\S]*\}/);
	if (!match) return undefined;
	try {
		const parsed = JSON.parse(match[0]) as { verdict?: string; reason?: string };
		const verdict = parsed.verdict?.toLowerCase();
		if (verdict === "allow" || verdict === "block" || verdict === "ask") {
			return { verdict, reason: parsed.reason };
		}
	} catch {
		// fall through
	}
	return undefined;
}

/**
 * Decide whether a tool call may run.
 *
 * Order: static deny rules → fast single-token filter → reasoning pass.
 * Any model failure or unparseable answer resolves to "ask" (fail-closed);
 * callers map "ask" to a prompt, or to headlessFallback when unattended.
 */
export async function decideApproval(
	request: ApprovalRequest,
	options: { runner: ApprovalModelRunner; signal?: AbortSignal },
): Promise<ApprovalDecision> {
	const serialized = `${request.toolName} ${safeStringify(request.args)}`;
	const rule = matchStaticDenyRule(serialized);
	if (rule) {
		return { verdict: "block", reason: rule.reason, ruleId: rule.id };
	}

	const prompt = buildApprovalPrompt(request);

	// Risky-but-legitimate actions skip the fast allow path: a one-token filter
	// is the wrong instrument for "is THIS installer, from THIS url, what the
	// user asked for?". They go straight to the reasoning stage instead of being
	// vetoed outright.
	const reviewRule = matchAlwaysReviewRule(serialized);

	if (!reviewRule) {
		let fastRaw: string;
		try {
			fastRaw = await options.runner({
				systemPrompt: FAST_FILTER_SYSTEM_PROMPT,
				prompt,
				maxTokens: 4,
				signal: options.signal,
			});
		} catch (error) {
			return { verdict: "ask", reason: `Approval model unavailable: ${errorText(error)}` };
		}

		const fast = parseFastVerdict(fastRaw);
		if (fast === "clear") {
			return { verdict: "allow", fastPath: true };
		}
		if (fast === undefined) {
			return { verdict: "ask", reason: "Approval model returned an unreadable verdict" };
		}
	}

	// Flagged (or always-review): spend reasoning tokens.
	let reasonedRaw: string;
	try {
		reasonedRaw = await options.runner({
			systemPrompt: REASONING_SYSTEM_PROMPT,
			prompt,
			maxTokens: 512,
			signal: options.signal,
		});
	} catch (error) {
		return { verdict: "ask", reason: `Approval review failed: ${errorText(error)}` };
	}

	const reasoned = parseReasonedVerdict(reasonedRaw);
	if (!reasoned) {
		return { verdict: "ask", reason: "Approval review returned an unreadable verdict" };
	}
	if (reviewRule && reasoned.verdict !== "allow" && !reasoned.reason) {
		return { ...reasoned, reason: reviewRule.reason, ruleId: reviewRule.id };
	}
	return reasoned;
}

function errorText(error: unknown): string {
	return error instanceof Error ? error.message : String(error);
}

/**
 * Collapse a decision into the final action.
 *
 * A cleared action always runs, so no policy here reintroduces approval fatigue.
 * Only an unclearable ("ask") verdict consults `whenUnsure`, and the "ask"
 * policy degrades to a block rather than failing open when nobody is attached.
 */
export function resolveApprovalOutcome(
	decision: ApprovalDecision,
	whenUnsure: ApprovalUnsureAction,
	canAskHuman: boolean,
): { action: "allow" | "block" | "prompt"; reason?: string } {
	if (decision.verdict === "block") {
		return { action: "block", reason: decision.reason };
	}
	if (decision.verdict === "allow") {
		return { action: "allow" };
	}

	// verdict === "ask"
	if (whenUnsure === "allow") {
		return { action: "allow" };
	}
	if (whenUnsure === "block") {
		return {
			action: "block",
			reason: decision.reason ?? "The reviewer could not clear this action",
		};
	}
	if (canAskHuman) {
		return { action: "prompt", reason: decision.reason };
	}
	return {
		action: "block",
		reason: decision.reason ?? "Needs approval but no approver is attached",
	};
}

/**
 * Message handed back to the agent when an action is refused.
 *
 * The agent is told who refused and why, so it can explain the block to the
 * user or choose a different approach instead of blindly retrying.
 */
export function formatApprovalDenial(toolName: string, reason?: string): string {
	const because = reason ? ` Reason: ${reason}` : "";
	return (
		`Approval denied for "${toolName}" by the approval reviewer.${because}` +
		" Do not retry the same call. Either take a safer approach, or explain to the user what you were trying to do and why it was refused."
	);
}
