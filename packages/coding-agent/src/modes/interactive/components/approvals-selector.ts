import { Container, getKeybindings, type SettingItem, SettingsList, Spacer, Text } from "@earendil-works/pi-tui";
import type { ApprovalMode, ApprovalUnsureAction } from "../../../core/approvals.js";
import { getSettingsListTheme, theme } from "../theme/theme.js";

export interface ApprovalsConfig {
	mode: ApprovalMode;
	model: string;
	whenUnsure: ApprovalUnsureAction;
	/** Every tool the agent can currently call, used to build the gating list. */
	availableTools: string[];
	/** Currently gated tools; empty means "all tools". */
	gatedTools: string[];
}

export interface ApprovalsCallbacks {
	onModeChange: (mode: ApprovalMode) => void;
	onModelChange: (model: string) => void;
	onWhenUnsureChange: (action: ApprovalUnsureAction) => void;
	onGatedToolsChange: (tools: string[]) => void;
	onCancel: () => void;
}

const MODE_DESCRIPTIONS: Record<ApprovalMode, string> = {
	off: "No checks - every tool call runs immediately",
	on: "Every gated tool call is reviewed before it runs",
};

const WHEN_UNSURE_DESCRIPTIONS: Record<ApprovalUnsureAction, string> = {
	ask: "Interrupt and let you decide (blocks if nobody is attached)",
	block: "Never interrupt - refuse it and tell the agent why",
	allow: "Never interrupt - run it anyway (least safe)",
};

/**
 * Multi-select submenu for choosing which tools go through review.
 *
 * Selecting nothing means "all tools", so a newly added tool is gated by
 * default rather than silently escaping review.
 */
class ToolGateSubmenu extends Container {
	private selectedIndex = 0;
	private gated: Set<string>;
	private readonly tools: string[];
	private readonly body = new Container();

	constructor(
		tools: string[],
		gated: string[],
		private readonly onChange: (tools: string[]) => void,
		private readonly onDone: () => void,
	) {
		super();
		this.tools = tools;
		this.gated = new Set(gated);

		this.addChild(new Text(theme.bold(theme.fg("accent", "Gated tools")), 0, 0));
		this.addChild(new Spacer(1));
		this.addChild(new Text(theme.fg("muted", "Space to toggle - select none to gate every tool"), 0, 0));
		this.addChild(new Spacer(1));
		this.addChild(this.body);
		this.addChild(new Spacer(1));
		this.addChild(new Text(theme.fg("dim", "  esc to go back"), 0, 0));

		this.renderRows();
	}

	private renderRows(): void {
		this.body.clear();
		if (this.tools.length === 0) {
			this.body.addChild(new Text(theme.fg("dim", "  (no tools reported by the session)"), 0, 0));
			return;
		}
		const gatingAll = this.gated.size === 0;
		this.tools.forEach((tool, index) => {
			const selected = index === this.selectedIndex;
			const on = gatingAll || this.gated.has(tool);
			const cursor = selected ? theme.fg("accent", ">") : " ";
			const box = on ? "[x]" : "[ ]";
			const label = selected ? theme.bold(tool) : tool;
			const suffix = gatingAll ? theme.fg("dim", "  (all)") : "";
			this.body.addChild(new Text(`${cursor} ${box} ${label}${suffix}`, 0, 0));
		});
	}

	handleInput(data: string): void {
		const kb = getKeybindings();

		if (kb.matches(data, "tui.select.up")) {
			if (this.tools.length === 0) return;
			this.selectedIndex = this.selectedIndex === 0 ? this.tools.length - 1 : this.selectedIndex - 1;
			this.renderRows();
			return;
		}
		if (kb.matches(data, "tui.select.down")) {
			if (this.tools.length === 0) return;
			this.selectedIndex = this.selectedIndex === this.tools.length - 1 ? 0 : this.selectedIndex + 1;
			this.renderRows();
			return;
		}
		if (kb.matches(data, "tui.select.cancel")) {
			this.onDone();
			return;
		}
		if (data === " " || kb.matches(data, "tui.select.confirm")) {
			const tool = this.tools[this.selectedIndex];
			if (!tool) return;
			if (this.gated.size === 0) {
				// Was "all": start an explicit list with everything except this one.
				this.gated = new Set(this.tools.filter((t) => t !== tool));
			} else if (this.gated.has(tool)) {
				this.gated.delete(tool);
			} else {
				this.gated.add(tool);
			}
			// Everything selected is the same as "all"; store it as such.
			if (this.gated.size === this.tools.length) this.gated = new Set();
			this.onChange([...this.gated]);
			this.renderRows();
			return;
		}
	}
}

/**
 * `/approvals` panel.
 *
 * Mirrors Claude Code auto mode: a separate approval model reviews each tool
 * call before it runs. Mode, model, gated tools and the unsure-policy are all
 * configurable here.
 */
export class ApprovalsSelectorComponent extends Container {
	private list: SettingsList;

	constructor(config: ApprovalsConfig, callbacks: ApprovalsCallbacks) {
		super();

		const describeTools = (gated: string[]) =>
			gated.length === 0 ? `all (${config.availableTools.length || "?"})` : gated.join(", ");

		let gatedTools = [...config.gatedTools];

		const items: SettingItem[] = [
			{
				id: "mode",
				label: "Mode",
				description: MODE_DESCRIPTIONS[config.mode],
				currentValue: config.mode,
				values: ["off", "on"],
			},
			{
				id: "model",
				label: "Approval model",
				description: "Model that reviews each tool call (a mid-tier model is enough)",
				currentValue: config.model,
			},
			{
				id: "tools",
				label: "Gated tools",
				description: "Which tools go through review",
				currentValue: describeTools(gatedTools),
				submenu: (_current, done) =>
					new ToolGateSubmenu(
						config.availableTools,
						gatedTools,
						(tools) => {
							gatedTools = tools;
							callbacks.onGatedToolsChange(tools);
						},
						() => done(describeTools(gatedTools)),
					),
			},
			{
				id: "whenUnsure",
				label: "When unsure",
				description: WHEN_UNSURE_DESCRIPTIONS[config.whenUnsure],
				currentValue: config.whenUnsure,
				values: ["ask", "block", "allow"],
			},
		];

		this.addChild(new Text(theme.bold(theme.fg("accent", "Approvals")), 0, 0));
		this.addChild(new Spacer(1));
		this.addChild(
			new Text(theme.fg("muted", "A separate approval model reviews each tool call before it runs."), 0, 0),
		);
		this.addChild(new Spacer(1));

		this.list = new SettingsList(
			items,
			Math.min(items.length, 10),
			getSettingsListTheme(),
			(id, newValue) => {
				switch (id) {
					case "mode":
						callbacks.onModeChange(newValue as ApprovalMode);
						break;
					case "model":
						callbacks.onModelChange(newValue);
						break;
					case "whenUnsure":
						callbacks.onWhenUnsureChange(newValue as ApprovalUnsureAction);
						break;
				}
			},
			callbacks.onCancel,
		);

		// SettingsList renders its own key hint; adding another duplicates it.
		this.addChild(this.list);
	}

	handleInput(data: string): void {
		this.list.handleInput(data);
	}
}
