"""
BA module attribution distribution — unconstrained vs constrained runs.

Stacked horizontal bar chart showing the fraction of opt_cycles attributed
to agent / descriptor / skip by the BA, for unconstrained gated runs and
module-ablation constrained runs.

Output:
    figures/ba_attribution.pdf / .png

Usage:
    cd final_paper_plotting
    python plot_ba_attribution.py
"""

import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from config import (
    AGENT_NAME, VARIANT_LABELS,
    FIG_WIDTH_HALF,
    SLUGS, opt_dir, latest_campaign,
    apply_neurips_style, save,
)

apply_neurips_style()

_V = VARIANT_LABELS

CAMPAIGN_PRIMARY  = latest_campaign(SLUGS["hsp_rich"])
CAMPAIGN_ABLATION = latest_campaign(SLUGS["actor_ablation_rich"])

# ── Colours ───────────────────────────────────────────────────────────────────
C_AGENT      = "#0072B2"   # Okabe blue
C_DESCRIPTOR = "#D55E00"   # Okabe vermillion
C_SKIP       = "#999999"   # grey

# ── Condition definitions ─────────────────────────────────────────────────────
# (row_label, slug_key, campaign, group)
CONDITIONS = [
    (f"HSP / {_V['minimal']}",  "hsp_minimal",               CAMPAIGN_PRIMARY,  "Unconstrained"),
    (f"HSP / {_V['rich']}",     "hsp_rich",                  CAMPAIGN_PRIMARY,  "Unconstrained"),
    (f"LSP / {_V['minimal']}",  "lsp_minimal",               CAMPAIGN_PRIMARY,  "Unconstrained"),
    (f"LSP / {_V['rich']}",     "lsp_rich",                  CAMPAIGN_PRIMARY,  "Unconstrained"),
    (f"Actor-only / {_V['minimal']}",      "actor_ablation_minimal",      CAMPAIGN_ABLATION, "Constrained"),
    (f"Actor-only / {_V['rich']}",         "actor_ablation_rich",         CAMPAIGN_ABLATION, "Constrained"),
    (f"Descriptor-only / {_V['minimal']}", "descriptor_ablation_minimal", CAMPAIGN_ABLATION, "Constrained"),
    (f"Descriptor-only / {_V['rich']}",    "descriptor_ablation_rich",    CAMPAIGN_ABLATION, "Constrained"),
]


# ── Data loading ──────────────────────────────────────────────────────────────

def collect(slug_key, campaign):
    base = opt_dir(campaign, SLUGS[slug_key]) if campaign else None
    if not base or not base.exists():
        return Counter()
    counts = Counter()
    for log in base.rglob("optimisation_log.jsonl"):
        if any(p.startswith(("opt_cycle_", "env_round_", "eval_")) for p in log.parts):
            continue
        for line in open(log):
            r = json.loads(line)
            if r.get("record_type") != "opt_cycle":
                continue
            mod = (r.get("ba_output") or {}).get("module") or "none"
            counts[mod] += 1
    return counts


# ── Build data ────────────────────────────────────────────────────────────────

labels, fracs_agent, fracs_desc, fracs_skip, groups = [], [], [], [], []

for row_label, slug_key, campaign, group in CONDITIONS:
    c   = collect(slug_key, campaign)
    tot = sum(c.values()) or 1
    labels.append(row_label)
    fracs_agent.append(c.get("actor", c.get("agent", 0)) / tot)
    fracs_desc.append(c.get("descriptor", 0) / tot)
    fracs_skip.append((c.get("none", 0) + c.get(None, 0)) / tot)
    groups.append(group)


# ── Plot ──────────────────────────────────────────────────────────────────────

n    = len(labels)
y    = np.arange(n)
h    = 0.52

fig_h = 0.42 * n + 0.9
fig, ax = plt.subplots(figsize=(FIG_WIDTH_HALF, fig_h))
fig.subplots_adjust(left=0.38, right=0.98, top=0.97, bottom=0.12)

fa = np.array(fracs_agent)
fd = np.array(fracs_desc)
fs = np.array(fracs_skip)

ax.barh(y, fa,      height=h, color=C_AGENT,      label="Agent")
ax.barh(y, fd,      height=h, color=C_DESCRIPTOR, label="Descriptor", left=fa)
ax.barh(y, fs,      height=h, color=C_SKIP,       label="Skip",       left=fa + fd)

# Percentage annotations inside bars (only if wide enough)
for i in range(n):
    for frac, left, col in [
        (fa[i], 0,           "white"),
        (fd[i], fa[i],       "white"),
        (fs[i], fa[i]+fd[i], "0.3"),
    ]:
        if frac > 0.06:
            ax.text(left + frac / 2, i, f"{100*frac:.0f}%",
                    ha="center", va="center", fontsize=6.5,
                    color=col, fontweight="bold")

# Group separator
sep_y = 3.5   # between unconstrained (0-3) and constrained (4-7)
ax.axhline(sep_y, color="0.5", linewidth=0.8, linestyle="--")

# Group labels
ax.text(-0.02, 1.5,  "Unconstrained", transform=ax.get_yaxis_transform(),
        ha="right", va="center", fontsize=7, style="italic", color="0.4", rotation=90)
ax.text(-0.02, 5.5,  "Constrained",   transform=ax.get_yaxis_transform(),
        ha="right", va="center", fontsize=7, style="italic", color="0.4", rotation=90)

ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=7.5)
ax.invert_yaxis()
ax.set_xlim(0, 1)
ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=7)
ax.set_xlabel("Fraction of opt cycles", fontsize=8)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.legend(
    handles=[
        mpatches.Patch(color=C_AGENT,      label="Agent"),
        mpatches.Patch(color=C_DESCRIPTOR, label="Descriptor"),
        mpatches.Patch(color=C_SKIP,       label="Skip / none"),
    ],
    loc="lower center", bbox_to_anchor=(0.5, -0.12),
    ncol=3, fontsize=7.5, framealpha=0.9, edgecolor="0.8",
)

if __name__ == "__main__":
    save(fig, "ba_attribution")
    plt.show()
