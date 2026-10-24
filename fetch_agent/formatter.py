"""
Formats simulation result dicts into readable chat text for ASI:One.
"""

from typing import Dict, Any


def format_results(results: Dict[str, Any]) -> str:
    """
    Converts the combined {mode: summary_dict} output from sim_adapter
    into a readable multi-mode comparison string.
    """
    if not results:
        return "⚠️ No simulation results were produced."

    lines = ["🏥 **AegisCoord Disaster Relief Simulation Results**\n"]

    for mode, summary in results.items():
        label = mode.upper()
        if "error" in summary:
            lines.append(f"**{label}**: ❌ Error — {summary['error']}\n")
            continue

        rescued = summary.get("total_rescued", 0)
        died = summary.get("total_died", 0)
        handled = rescued + died
        survival = (rescued / handled * 100) if handled else 100.0
        avg_wait = summary.get("avg_wait_time", 0.0)
        fairness = summary.get("fairness_index", 0.0)
        agent_rescues = summary.get("agent_rescues", [])

        lines.append(f"**── {label} ──**")
        lines.append(f"  Rescued: {rescued}  |  Casualties: {died}  |  Survival: {survival:.1f}%")
        lines.append(f"  Avg wait: {avg_wait:.2f} steps  |  Fairness (Jain): {fairness:.4f}")
        if agent_rescues:
            lines.append(f"  Per-agent rescues: {agent_rescues}")
        lines.append("")

    # ── comparison summary ───────────────────────────────────────────
    valid = {m: s for m, s in results.items() if "error" not in s}
    if len(valid) >= 2:
        lines.append("**── Comparison ──**")
        best_mode = max(valid, key=lambda m: valid[m].get("total_rescued", 0))
        fairest_mode = max(valid, key=lambda m: valid[m].get("fairness_index", 0))
        fastest_mode = min(
            valid,
            key=lambda m: valid[m].get("avg_wait_time", 999),
        )
        lines.append(f"  Most rescues: **{best_mode.upper()}** ({valid[best_mode]['total_rescued']})")
        lines.append(f"  Fairest workload: **{fairest_mode.upper()}** (J={valid[fairest_mode]['fairness_index']:.4f})")
        lines.append(f"  Lowest wait: **{fastest_mode.upper()}** ({valid[fastest_mode]['avg_wait_time']:.2f} steps)")

    return "\n".join(lines)
