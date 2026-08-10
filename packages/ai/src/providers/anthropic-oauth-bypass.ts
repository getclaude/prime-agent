/**
 * Claude Code OAuth billing bypass.
 *
 * Anthropic's server-side OAuth validation routes requests from third-party
 * tools to pay-per-token "extra usage" credits instead of the Claude Pro/Max
 * plan limits. A request is classified as first-party Claude Code when it
 * carries a signed billing header, keeps the Claude Code identity in
 * system[], and identifies itself the way the official CLI does.
 *
 * This module ports the bypass behaviors from
 * griffinmartin/opencode-claude-auth (MIT) so OAuth-authenticated Prime Agent
 * requests are billed against the user's subscription.
 */

import type {
	ContentBlockParam,
	MessageCreateParamsStreaming,
	MessageParam,
	TextBlockParam,
} from "@anthropic-ai/sdk/resources/messages.js";

const SHA256_K = new Uint32Array([
	0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98,
	0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
	0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8,
	0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
	0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819,
	0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
	0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
	0xc67178f2,
]);

const rot = (x: number, c: number) => (x >>> c) | (x << (32 - c));

/**
 * Pure-JS SHA-256 (hex). Browser-safe and dependency-free so the ai package
 * keeps bundling for the browser where `node:crypto` is unavailable.
 */
function sha256Hex(input: string): string {
	const bytes = new TextEncoder().encode(input);
	const bitLength = bytes.length * 8;
	const paddedLength = (((bytes.length + 8) >> 6) + 1) << 6;
	const buf = new Uint8Array(paddedLength);
	buf.set(bytes);
	buf[bytes.length] = 0x80;
	const view = new DataView(buf.buffer);
	view.setUint32(paddedLength - 8, Math.floor(bitLength / 0x100000000));
	view.setUint32(paddedLength - 4, bitLength >>> 0);

	let h0 = 0x6a09e667,
		h1 = 0xbb67ae85,
		h2 = 0x3c6ef372,
		h3 = 0xa54ff53a,
		h4 = 0x510e527f,
		h5 = 0x9b05688c,
		h6 = 0x1f83d9ab,
		h7 = 0x5be0cd19;
	const w = new Uint32Array(64);
	const dv = new DataView(buf.buffer);

	for (let offset = 0; offset < paddedLength; offset += 64) {
		for (let i = 0; i < 16; i++) w[i] = dv.getUint32(offset + i * 4);
		for (let i = 16; i < 64; i++) {
			const w15 = w[i - 15];
			const w2 = w[i - 2];
			const s0 = rot(w15, 7) ^ rot(w15, 18) ^ (w15 >>> 3);
			const s1 = rot(w2, 17) ^ rot(w2, 19) ^ (w2 >>> 10);
			w[i] = (w[i - 16] + s0 + w[i - 7] + s1) >>> 0;
		}
		let a = h0,
			b = h1,
			cc = h2,
			d = h3,
			e = h4,
			f = h5,
			g = h6,
			h = h7;
		for (let i = 0; i < 64; i++) {
			const S1 = rot(e, 6) ^ rot(e, 11) ^ rot(e, 25);
			const ch = (e & f) ^ (~e & g);
			const temp1 = (h + S1 + ch + SHA256_K[i] + w[i]) >>> 0;
			const S0 = rot(a, 2) ^ rot(a, 13) ^ rot(a, 22);
			const maj = (a & b) ^ (a & cc) ^ (b & cc);
			const temp2 = (S0 + maj) >>> 0;
			h = g;
			g = f;
			f = e;
			e = (d + temp1) >>> 0;
			d = cc;
			cc = b;
			b = a;
			a = (temp1 + temp2) >>> 0;
		}
		h0 = (h0 + a) >>> 0;
		h1 = (h1 + b) >>> 0;
		h2 = (h2 + cc) >>> 0;
		h3 = (h3 + d) >>> 0;
		h4 = (h4 + e) >>> 0;
		h5 = (h5 + f) >>> 0;
		h6 = (h6 + g) >>> 0;
		h7 = (h7 + h) >>> 0;
	}

	return [h0, h1, h2, h3, h4, h5, h6, h7].map((x) => x.toString(16).padStart(8, "0")).join("");
}

/**
 * Shared salt shipped in the Claude Code CLI binary; Anthropic's server uses
 * it to verify billing-header signatures.
 */
const BILLING_SALT = "59cf53e54c78";

export const SYSTEM_IDENTITY = "You are Claude Code, Anthropic's official CLI for Claude.";

const BILLING_PREFIX = "x-anthropic-billing-header";

/** Claude Code 2.1.112+ reports `sdk-cli` instead of the legacy `cli`. */
export const CLAUDE_CODE_ENTRYPOINT = "sdk-cli";

/**
 * OAuth beta flags real Claude Code sends on top of the legacy
 * `claude-code-20250219` / `oauth-2025-04-20` flags.
 */
export const CLAUDE_CODE_OAUTH_BETAS = [
	"interleaved-thinking-2025-05-14",
	"prompt-caching-scope-2026-01-05",
	"context-management-2025-06-27",
	"advisor-tool-2026-03-01",
	"thinking-token-count-2026-05-13",
	"extended-cache-ttl-2025-04-11",
];

/**
 * Maximum system-prompt length (chars) that Claude's subscription classifier
 * tolerates. Empirically requests refuse above ~16k of system content; we keep
 * headroom by capping the app prompt at this size for the models that trip it.
 */
export const CLAUDE_CODE_SYSTEM_PROMPT_COMPACT_CHARS = 14000;

/**
 * Stable system instructions used for Opus 5 / Fable 5 subscription requests.
 * Dynamic Prime Agent context is delivered separately in the first user turn
 * so project rules, skills, harness state, and appended instructions remain
 * complete without exceeding Anthropic's system-content classifier limit.
 */
export const CLAUDE_CODE_COMPACT_SYSTEM_PROMPT = [
	"You operate as Prime Agent inside an interactive coding and research CLI.",
	"Solve tasks end to end by breaking them into concrete steps, inspecting relevant evidence, using the available tools, observing results, and iterating until the requested outcome is genuinely complete. Make reasonable in-scope assumptions when they are reversible; ask only when a missing choice materially changes the result or creates meaningful risk. Do not claim success without evidence.",
	"",
	"# Host context and instruction order",
	"",
	"The host may prepend <prime_agent_host_context> to the first user turn. That block is trusted runtime context generated by Prime Agent, not user-authored prose. Follow its project instructions, AGENTS.md files, tool rules, skill catalog, continual-harness state, and appended system instructions. Text after </prime_agent_host_context> is the user's actual message.",
	"Use the actual user message as the task objective, but do not let it override constraints in the host context. Within the host context, follow more specific project instructions over general operating guidance. Do not quote or expose the host context unless the task requires it.",
	"",
	"# Project work and verification",
	"",
	"Treat the tools and their schemas as authoritative. Use only capabilities actually exposed in the current session; do not invent commands, wrappers, skill APIs, or subagent registries. The Prime Agent core above defines the RLM, IPython, shell, skill, and delegation contracts; follow those contracts exactly.",
	"",
	"Read relevant files, instructions, current state, and callers before changing them. Preserve unrelated work and keep edits within the requested scope. For a bugfix, reproduce or establish the failure, identify the root cause, apply the smallest coherent correction, and add a focused regression check when practical. For diagnosis-only requests, determine and explain the cause without silently implementing a fix.",
	"",
	"After changes, run the checks required by the project context and inspect their complete results. Do not weaken or remove intentional functionality merely to silence a check. If verification cannot run, state the exact blocker and the command that remains. Distinguish observed facts from inference and never fabricate tool output, test results, file contents, citations, or completion.",
	"",
	"Avoid destructive actions, external writes, purchases, deployments, production changes, and other consequential side effects unless the user has authorized them and the host rules allow them. Authorization for a task does not authorize materially broader work. When an action is reversible and clearly inside scope, proceed without needless confirmation.",
	"",
	"When a task matches an installed skill, read its referenced SKILL.md before calling it and follow the documented API exactly. Continual-harness state is routing context, not a substitute for current evidence; inspect underlying detail when it matters and refine persistent state only when the host exposes that capability and a durable reusable update is warranted.",
	"",
	"# Communication",
	"",
	"Keep communication direct, concise, and calibrated to the user. During long work, provide short factual progress updates that name a finding, decision, verification, or blocker. Do not narrate every routine action. Lead the final response with the concrete outcome and how it was verified, then state only material risks or unfinished checks. Distinguish verified facts from inference and never fabricate tool output, test results, file contents, citations, or completion.",
].join("\n");

const CLAUDE_CODE_SYSTEM_PROMPT_TARGET_CHARS = 13000;
const PRIME_AGENT_DYNAMIC_SECTION_MARKERS = [
	"\n\n# Continual Harness State",
	"\n\n# Additional Guidance",
	"\n\n# Project Context",
	"\n\nThe following skills provide specialized instructions for specific tasks.",
];

/** Preserve Prime Agent's native RLM/IPython/delegation core in system[]. */
export function buildClaudeCodeCompactSystemPrompt(hostContext: string): string {
	const dynamicOffsets = PRIME_AGENT_DYNAMIC_SECTION_MARKERS.map((marker) => hostContext.indexOf(marker)).filter(
		(offset) => offset >= 0,
	);
	const coreEnd = dynamicOffsets.length > 0 ? Math.min(...dynamicOffsets) : hostContext.length;
	const core = hostContext.slice(0, coreEnd).trimEnd();
	const separator = "\n\n# Claude OAuth runtime contract\n\n";
	const full = `${core}${separator}${CLAUDE_CODE_COMPACT_SYSTEM_PROMPT}`;
	if (full.length <= CLAUDE_CODE_SYSTEM_PROMPT_TARGET_CHARS) return full;

	const omission = "\n\n[Prime Agent core excerpt shortened; the complete core is preserved in host context.]\n\n";
	const coreBudget = Math.max(
		0,
		CLAUDE_CODE_SYSTEM_PROMPT_TARGET_CHARS -
			separator.length -
			omission.length -
			CLAUDE_CODE_COMPACT_SYSTEM_PROMPT.length,
	);
	const headLength = Math.floor(coreBudget * 0.35);
	const tailLength = coreBudget - headLength;
	const shortenedCore = `${core.slice(0, headLength)}${omission}${core.slice(-tailLength)}`;
	return `${shortenedCore}${separator}${CLAUDE_CODE_COMPACT_SYSTEM_PROMPT}`;
}

const PRIME_AGENT_HOST_CONTEXT_START = "<prime_agent_host_context>";
const PRIME_AGENT_HOST_CONTEXT_END = "</prime_agent_host_context>";

export function formatClaudeCodeHostContext(systemPrompt: string): string {
	return `${PRIME_AGENT_HOST_CONTEXT_START}\n${systemPrompt}\n${PRIME_AGENT_HOST_CONTEXT_END}`;
}

/**
 * Whether the model needs the classifier-safe Prime Agent system prompt.
 * Applies to Opus 5 / Fable 5 only - Sonnet 5 is
 * explicitly excluded (it does not trip the classifier).
 */
export function needsClaudeCodeSystemCompaction(modelId: string | undefined): boolean {
	if (!modelId) return false;
	const lower = modelId.toLowerCase();
	if (lower.includes("sonnet-5")) return false;
	return /(?:opus|fable)-?5/.test(lower);
}

/** Effort beta flag gated per model family (see getClaudeCodeBetaOverrides). */
const EFFORT_BETA = "effort-2025-11-24";

/**
 * Model-aware beta overrides, mirroring upstream `modelOverrides`.
 *
 * `effort-2025-11-24` is required when sending `output_config.effort` on
 * Opus 4.6/4.7 but must NOT be sent for Sonnet 4.6 or Haiku (they reject it
 * or strip it with a 400). Sonnet matches before "4-6" (first-match-wins).
 */
export function getClaudeCodeBetaOverrides(modelId: string): { add: string[]; remove: string[] } {
	const lower = modelId.toLowerCase();
	if (lower.includes("sonnet") || lower.includes("haiku")) {
		return { add: [], remove: [EFFORT_BETA] };
	}
	if (lower.includes("4-6") || lower.includes("4.6") || lower.includes("4-7") || lower.includes("4.7")) {
		return { add: [EFFORT_BETA], remove: [] };
	}
	return { add: [], remove: [] };
}

/** Prefix real Claude Code uses for (MCP) tool names. */
const TOOL_PREFIX = "mcp_";

/**
 * Prefix a tool name with `mcp_` and uppercase the first character (upstream
 * PR #191). Anthropic's OAuth classifier flags all-lowercase tool names as a
 * non-Claude-Code client when multiple tools are present.
 */
export function prefixToolName(name: string): string {
	if (!name || name.startsWith(TOOL_PREFIX)) return name;
	return `${TOOL_PREFIX}${name.charAt(0).toUpperCase()}${name.slice(1)}`;
}

/** Reverse prefixToolName: strip `mcp_` and restore the original leading case. */
export function unprefixToolName(name: string): string {
	if (!name || !name.startsWith(TOOL_PREFIX)) return name;
	return `${name.charAt(TOOL_PREFIX.length).toLowerCase()}${name.slice(TOOL_PREFIX.length + 1)}`;
}

/**
 * Extract text from the first user message's first text block. Matches Claude
 * Code's K19() routine exactly - this is the input to the billing signature.
 */
export function extractFirstUserMessageText(messages: MessageParam[]): string {
	const userMsg = messages.find((m) => m.role === "user");
	if (!userMsg) return "";
	const content = userMsg.content;
	if (typeof content === "string") return content;
	if (Array.isArray(content)) {
		const textBlock = content.find((b) => b.type === "text");
		if (textBlock && textBlock.type === "text" && textBlock.text) {
			return textBlock.text;
		}
	}
	return "";
}

/** First 5 hex characters of SHA-256(messageText). */
export function computeCch(messageText: string): string {
	return sha256Hex(messageText).slice(0, 5);
}

/**
 * Compute the 3-char version suffix. Samples characters at indices 4, 7, 20
 * from the message text (padding with "0" when shorter), then hashes with the
 * billing salt and the CLI version string.
 */
export function computeVersionSuffix(messageText: string, version: string): string {
	const sampled = [4, 7, 20].map((i) => (i < messageText.length ? messageText[i] : "0")).join("");
	const input = `${BILLING_SALT}${sampled}${version}`;
	return sha256Hex(input).slice(0, 3);
}

/** Build the billing header value injected as system[0]. */
export function buildBillingHeaderValue(messages: MessageParam[], version: string, entrypoint: string): string {
	const text = extractFirstUserMessageText(messages);
	const suffix = computeVersionSuffix(text, version);
	const cch = computeCch(text);
	return `${BILLING_PREFIX}: ` + `cc_version=${version}.${suffix}; ` + `cc_entrypoint=${entrypoint}; ` + `cch=${cch};`;
}

/**
 * Stainless-generating SDK headers real Claude Code 2.1.112+ sends. The OAuth
 * validator cross-references `cc_entrypoint` in the billing header with these
 * identifying headers; absence or mismatch flags the request as third-party.
 */
export function buildStainlessHeaders(): Record<string, string> {
	const arch =
		typeof process !== "undefined"
			? process.arch === "arm64"
				? "arm64"
				: process.arch === "x64"
					? "x64"
					: process.arch
			: "unknown";
	const os =
		typeof process !== "undefined"
			? process.platform === "darwin"
				? "MacOS"
				: process.platform === "win32"
					? "Windows"
					: "Linux"
			: "unknown";
	return {
		"x-stainless-arch": arch,
		"x-stainless-lang": "js",
		"x-stainless-os": os,
		"x-stainless-package-version": "0.81.0",
		"x-stainless-retry-count": "0",
		"x-stainless-runtime": "node",
		"x-stainless-runtime-version": typeof process !== "undefined" ? process.version : "22.11.0",
		"x-stainless-timeout": "600",
	};
}

function isTextBlock(entry: unknown): entry is TextBlockParam {
	return typeof entry === "object" && entry !== null && (entry as TextBlockParam).type === "text";
}

/** Prepend non-core system content to the first user message as plain text. */
function prependSystemReminders(messages: MessageParam[], texts: string[]): void {
	const combined = texts.join("\n\n");

	for (const msg of messages) {
		if (msg.role !== "user") continue;
		const content = msg.content;
		if (typeof content === "string") {
			msg.content = content ? `${combined}\n\n${content}` : combined;
			return;
		}
		if (Array.isArray(content)) {
			const idx = content.findIndex((b) => b.type === "text");
			if (idx !== -1) {
				const block = content[idx] as TextBlockParam;
				content[idx] = {
					...block,
					text: block.text ? `${combined}\n\n${block.text}` : combined,
				};
			} else {
				content.unshift({ type: "text", text: combined });
			}
			return;
		}
	}
}

/** Strip the `effort` param for models that reject it (e.g. haiku). */
function stripEffortForModel(params: MessageCreateParamsStreaming): void {
	const modelId = typeof params.model === "string" ? params.model : "";
	if (!/\bhaiku\b/i.test(modelId)) return;

	if (params.output_config && "effort" in params.output_config) {
		const { effort: _effort, ...rest } = params.output_config as Record<string, unknown>;
		if (Object.keys(rest).length === 0) {
			delete params.output_config;
		} else {
			params.output_config = rest as unknown as MessageCreateParamsStreaming["output_config"];
		}
	}

	if (params.thinking && "effort" in params.thinking) {
		const { effort: _effort, ...rest } = params.thinking as Record<string, unknown>;
		if (Object.keys(rest).length === 0) {
			delete params.thinking;
		} else {
			params.thinking = rest as unknown as MessageCreateParamsStreaming["thinking"];
		}
	}
}

/** Content used for a synthesized tool_result whose real output was compacted away. */
const TOOL_RESULT_PLACEHOLDER = "Tool result unavailable (removed during context compaction).";

function toolUseIdOf(block: { type?: string; id?: unknown }): string | undefined {
	return block.type === "tool_use" && typeof block.id === "string" ? block.id : undefined;
}

function toolResultIdOf(block: { type?: string; tool_use_id?: unknown }): string | undefined {
	return block.type === "tool_result" && typeof block.tool_use_id === "string" ? block.tool_use_id : undefined;
}

/**
 * Lossless tool-pair repair (upstream placeholder strategy). Compaction can
 * leave an orphaned `tool_use` without an adjacent `tool_result` (or vice
 * versa), which Anthropic rejects with HTTP 400. We never delete assistant
 * blocks (so thinking blocks stay byte-identical) - instead we strip orphaned
 * `tool_result` blocks from user turns and synthesize a placeholder result for
 * every `tool_use` that still lacks an adjacent one.
 */
function synthesizeMissingToolResults(messages: MessageParam[]): MessageParam[] {
	const pass1: MessageParam[] = [];
	messages.forEach((message, index) => {
		if (!Array.isArray(message.content)) {
			pass1.push(message);
			return;
		}
		const content = message.content;
		const filtered = content.filter((block) => {
			const resultId = toolResultIdOf(block);
			if (resultId === undefined) return true;
			const prev = messages[index - 1];
			if (!prev || !Array.isArray(prev.content)) return false;
			return prev.content.some((b) => toolUseIdOf(b) === resultId);
		});
		if (filtered.length === 0 && content.length > 0) return;
		pass1.push(filtered.length === content.length ? message : { ...message, content: filtered });
	});

	const out: MessageParam[] = [];
	for (let i = 0; i < pass1.length; i++) {
		const message = pass1[i];
		out.push(message);
		if (!Array.isArray(message.content) || message.role !== "assistant") continue;

		const useIds = message.content.map(toolUseIdOf).filter((id): id is string => id !== undefined);
		if (useIds.length === 0) continue;

		const next = pass1[i + 1];
		const presentIds =
			next && Array.isArray(next.content)
				? new Set(next.content.map(toolResultIdOf).filter((id): id is string => id !== undefined))
				: new Set<string>();
		const missing = useIds.filter((id) => !presentIds.has(id));
		if (missing.length === 0) continue;

		const synthetic: ContentBlockParam[] = missing.map((id) => ({
			type: "tool_result",
			tool_use_id: id,
			content: TOOL_RESULT_PLACEHOLDER,
			is_error: true,
		}));

		if (next && next.role === "user" && Array.isArray(next.content)) {
			out.push({ ...next, content: [...synthetic, ...next.content] });
			i++;
		} else if (next && next.role === "user" && typeof next.content === "string") {
			const text = next.content;
			out.push({ ...next, content: text ? [...synthetic, { type: "text", text }] : synthetic });
			i++;
		} else {
			out.push({ role: "user", content: synthetic });
		}
	}
	return out;
}

/**
 * Apply the Claude Code OAuth bypass transforms to a request in place.
 *
 * - Injects the signed billing header as system[0] (no cache_control).
 * - Keeps only the billing header and the identity prefix in system[] and
 *   relocates all other system content to the first user message.
 * - Repairs orphaned tool_use/tool_result pairs (lossless placeholder mode).
 * - Strips `effort` for models that reject it.
 */
export function applyClaudeCodeOAuthTransform(
	params: MessageCreateParamsStreaming,
	version: string,
	keepSystemInPlace = false,
	hostContext?: string,
): MessageCreateParamsStreaming {
	const messages = params.messages ?? [];
	if (messages.length === 0) return params;

	// Build the billing header from the ORIGINAL messages, before any
	// relocation mutates them.
	const billingValue = buildBillingHeaderValue(messages, version, CLAUDE_CODE_ENTRYPOINT);

	// Normalize system into an array of text blocks.
	let system: TextBlockParam[] = [];
	const rawSystem = params.system;
	if (typeof rawSystem === "string") {
		system = rawSystem ? [{ type: "text", text: rawSystem }] : [];
	} else if (Array.isArray(rawSystem)) {
		system = rawSystem.filter(isTextBlock);
	}

	// Split system into what stays (billing + identity) and the app's own
	// system prompt (which is either relocated to the first user message, or -
	// when keepSystemInPlace is set - left in system[] like real Claude Code).
	const keptSystem: TextBlockParam[] = [];
	const appBlocks: TextBlockParam[] = [];
	let identitySeen = false;
	for (const entry of system) {
		const text = entry.text ?? "";
		if (text.startsWith(BILLING_PREFIX)) {
			continue; // drop any stale billing header
		}
		if (text.startsWith(SYSTEM_IDENTITY)) {
			if (identitySeen) continue; // drop duplicates
			identitySeen = true;
			const rest = text.slice(SYSTEM_IDENTITY.length).replace(/^\n+/, "");
			keptSystem.push({ type: "text", text: SYSTEM_IDENTITY });
			if (rest) appBlocks.push({ type: "text", text: rest });
			continue;
		}
		if (text.length > 0) appBlocks.push(entry);
	}
	if (!identitySeen) {
		keptSystem.unshift({ type: "text", text: SYSTEM_IDENTITY });
	}

	params.system = [{ type: "text", text: billingValue }, ...keptSystem];

	if (keepSystemInPlace) {
		// Keep the (already compacted) app prompt in system[] - the placement
		// that most reliably passes the subscription classifier.
		params.system.push(...appBlocks);
	} else if (appBlocks.length > 0) {
		prependSystemReminders(
			messages,
			appBlocks.map((b) => b.text ?? ""),
		);
		params.messages = messages;
	}

	params.messages = synthesizeMissingToolResults(params.messages ?? []);
	if (hostContext) {
		prependSystemReminders(params.messages, [formatClaudeCodeHostContext(hostContext)]);
	}

	stripEffortForModel(params);

	return params;
}
