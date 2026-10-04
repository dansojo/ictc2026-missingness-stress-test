from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


FAMILY_LABELS = {
    "event_count": "Event count",
    "state_ratio": "State ratio",
    "intensity": "Intensity",
    "timing": "Timing",
}
FAMILY_MARKERS = {
    "event_count": "o",
    "state_ratio": "s",
    "intensity": "^",
    "timing": "D",
}
SERIF_FONTS = ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"]
DOWNSTREAM_Y_HEADROOM = 1.8
PRIMARY_X_LABEL = r"Median $\Delta e$"
VECTOR_FONT_TYPE = 42


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def _plot_results(
    primary: list[dict[str, str]],
    participant_deltas: list[dict[str, str]],
    dose: list[dict[str, str]],
    downstream: list[dict[str, str]],
    pdf_path: Path,
    png_path: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": SERIF_FONTS,
            "mathtext.fontset": "stix",
            "pdf.fonttype": VECTOR_FONT_TYPE,
            "ps.fonttype": VECTOR_FONT_TYPE,
            "font.size": 7.2,
            "axes.titlesize": 8.2,
            "axes.labelsize": 7.2,
            "xtick.labelsize": 6.6,
            "ytick.labelsize": 6.6,
            "legend.fontsize": 6.2,
            "axes.linewidth": 0.7,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(7.05, 2.35), constrained_layout=True)

    order = ["event_count", "state_ratio", "intensity", "timing"]
    by_family = {row["feature_family"]: row for row in primary}
    y = np.arange(4)
    medians = np.array([_float(by_family[f], "participant_median") for f in order])
    lows = np.array([_float(by_family[f], "ci_low") for f in order])
    highs = np.array([_float(by_family[f], "ci_high") for f in order])
    significant = np.array([_float(by_family[f], "holm_p_value") < 0.05 for f in order])
    axes[0].axvline(0, color="0.45", linewidth=0.8, linestyle="--")
    for index in range(4):
        family_points = sorted(
            (row for row in participant_deltas if row["feature_family"] == order[index]),
            key=lambda row: row["subject_id"],
        )
        jitter = np.linspace(-0.17, 0.17, len(family_points))
        axes[0].scatter(
            [_float(row, "delta") for row in family_points],
            index + jitter,
            s=9,
            facecolors="white",
            edgecolors="0.55",
            linewidths=0.55,
            zorder=2,
        )
        axes[0].errorbar(
            medians[index],
            y[index],
            xerr=[[medians[index] - lows[index]], [highs[index] - medians[index]]],
            fmt=FAMILY_MARKERS[order[index]],
            markersize=4.2,
            markerfacecolor="0.15" if significant[index] else "white",
            markeredgecolor="0.15",
            color="0.15",
            capsize=2.2,
            linewidth=1.0,
            zorder=3,
        )
    axes[0].set_yticks(y, [FAMILY_LABELS[f] for f in order])
    axes[0].invert_yaxis()
    axes[0].set_xlabel(PRIMARY_X_LABEL)
    axes[0].set_title("(a) Matched 20% geometry effect")
    axes[0].text(
        0.98,
        0.03,
        "filled: Holm $p<.05$",
        transform=axes[0].transAxes,
        ha="right",
        va="bottom",
        fontsize=5.8,
        color="0.25",
    )

    linestyles = {
        "event_count": "-",
        "state_ratio": "--",
        "intensity": "-.",
        "timing": ":",
    }
    for family in order:
        rows = sorted(
            (row for row in dose if row["feature_family"] == family),
            key=lambda row: float(row["missingness_rate"]),
        )
        marker, linestyle = FAMILY_MARKERS[family], linestyles[family]
        axes[1].plot(
            [100 * _float(row, "missingness_rate") for row in rows],
            [_float(row, "participant_median") for row in rows],
            marker=marker,
            linestyle=linestyle,
            color="0.15",
            linewidth=1.0,
            markersize=3.8,
            markerfacecolor="white" if family in ("event_count", "timing") else "0.15",
            label=FAMILY_LABELS[family],
        )
    axes[1].set_xticks([10, 20, 40])
    axes[1].set_xlabel("Contiguous missingness (%)")
    axes[1].set_ylabel("Median standardized error")
    axes[1].set_title("(b) Dose response")
    axes[1].legend(frameon=False, loc="upper left", ncol=1, handlelength=2.2)

    condition_order = ["scattered_random_20pct", "contiguous_20pct", "event_boundary_20pct"]
    condition_labels = ["Random", "Contiguous", "Boundary"]
    cond = {row["condition"]: row for row in downstream}
    heights = [_float(cond[key], "participant_first_median_abs_probability_change") for key in condition_order]
    bars = axes[2].bar(
        np.arange(3),
        heights,
        color=["white", "0.55", "0.15"],
        edgecolor="0.15",
        linewidth=0.8,
        hatch=["///", "...", "xxx"],
    )
    axes[2].set_xticks(np.arange(3), condition_labels, rotation=16)
    axes[2].set_ylabel(r"Median $|\Delta p|$")
    axes[2].set_title("(c) Downstream stability")
    axes[2].set_ylim(0, max(heights) * DOWNSTREAM_Y_HEADROOM)
    for bar, key in zip(bars, condition_order):
        row = cond[key]
        axes[2].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(heights) * 0.045,
            f"{100 * _float(row, 'flip_fraction_applicable'):.2f}% flips",
            ha="center",
            va="bottom",
            fontsize=5.8,
            rotation=90,
        )
    axes[2].text(
        0.02,
        0.95,
        "Boundary: 1,104 abstained;\n8,000 unavailable",
        transform=axes[2].transAxes,
        ha="left",
        va="top",
        fontsize=5.7,
        bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "0.6", "linewidth": 0.5},
    )

    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.grid(axis="y", color="0.88", linewidth=0.5, zorder=0)
        axis.set_axisbelow(True)
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=450, bbox_inches="tight")
    plt.close(fig)


def _table1() -> str:
    return r"""\begin{table*}[!t]
\caption{Frozen Feature Contracts and ETRI Analysis Support}
\label{tab:design}
\centering
\scriptsize
\setlength{\tabcolsep}{3.5pt}
\begin{tabular}{p{17mm}p{36mm}p{52mm}p{39mm}p{20mm}}
\toprule
Family & Source primitives & Contracted statistic & Operation-specific eligibility & Finite pairs\textsuperscript{a} \\
\midrule
Event count & step load & Sum of valid nonnegative event contributions & At least one valid event contribution & 4,995/5,000 \\
State ratio & screen and phone-activity load & Positive valid-state units divided by valid state units & Positive valid-state denominator & 10,000/10,000 \\
Intensity & usage and mobile/wearable light exposure & Mean valid nonnegative duration or transformed intensity & At least one finite valid nonnegative contribution & 14,840/15,000 \\
Timing & screen and usage disengagement $p_{90}$ & Weighted 90th percentile over positive event mass & At least three positive source events and positive total event mass & 9,790/10,000 \\
\bottomrule
\end{tabular}
\vspace{1mm}
\parbox{0.97\textwidth}{\scriptsize \textsuperscript{a}Finite comparisons with defined outputs / scheduled comparisons. Reference values use observed support rather than unobserved ground truth. The primary contrast holds the valid deleted-record count fixed between contiguous and scattered-random 20\% loss.}
\end{table*}
"""


def _table2(external: list[dict[str, str]]) -> str:
    selected = [
        row
        for row in external
        if row["feature_family"] in ("state_ratio", "intensity")
    ]
    rows = []
    for row in selected:
        dataset = row["dataset"]
        family = FAMILY_LABELS[row["feature_family"]]
        n = int(row["eligible_participants"])
        positive = int(row["positive_participants"])
        median = float(row["participant_median"])
        low = float(row["ci_low"])
        high = float(row["ci_high"])
        rows.append(
            f"{dataset} / {family} & {positive}/{n} & {median:.3f} [{low:.3f}, {high:.3f}] \\\\"
        )
    body = "\n".join(rows)
    return rf"""\begin{{table}}[!t]
\caption{{Exploratory Directional Replication Under Matched 20\% Loss}}
\label{{tab:external}}
\centering
\scriptsize
\setlength{{\tabcolsep}}{{3pt}}
\begin{{tabular}}{{@{{}}lcc@{{}}}}
\toprule
Dataset / family & Positive\textsuperscript{{a}} & Median $\Delta e$\textsuperscript{{b}} [95\% CI] \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\vspace{{0.5mm}}
\parbox{{\columnwidth}}{{\scriptsize \textsuperscript{{a}}positive-median / eligible participants. \textsuperscript{{b}}$\Delta e$: contiguous minus scattered-random standardized error; deterministic participant-bootstrap 95\% CI. Exploratory; no formal transfer verdict.}}
\end{{table}}
"""


def build(repo: Path, figure_dir: Path, table_dir: Path) -> dict:
    evidence = repo / "paper_ictc2026" / "02_experiments" / "etri_paper_evidence"
    external_dir = repo / "paper_ictc2026" / "02_experiments" / "external_transfer" / "results"
    source_paths = {
        "primary": evidence / "primary_feature_inference.csv",
        "participant_deltas": evidence / "participant_primary_deltas.csv",
        "dose": evidence / "dose_response.csv",
        "downstream": evidence / "downstream_condition_summary.csv",
        "external": external_dir / "external_transfer_summary.csv",
        "external_manifest": external_dir / "external_transfer_manifest.json",
    }
    for path in source_paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    external_manifest = json.loads(source_paths["external_manifest"].read_text(encoding="utf-8"))
    if external_manifest["decision"] != {
        "label": "full_transfer",
        "passed_primary_pairs": 4,
        "total_primary_pairs": 4,
    }:
        raise ValueError("external-transfer decision is not the verified 4/4 result")
    if external_manifest["external_labels_opened"] is not False:
        raise ValueError("external label boundary was violated")
    primary = _read_csv(source_paths["primary"])
    participant_deltas = _read_csv(source_paths["participant_deltas"])
    dose = _read_csv(source_paths["dose"])
    downstream = _read_csv(source_paths["downstream"])
    external = _read_csv(source_paths["external"])
    if (len(primary), len(participant_deltas), len(dose), len(downstream), len(external)) != (4, 40, 12, 3, 8):
        raise ValueError("source table topology mismatch")
    if {row["feature_family"] for row in participant_deltas} != set(FAMILY_LABELS):
        raise ValueError("participant-point family topology mismatch")

    figure_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = figure_dir / "fig2_results.pdf"
    png_path = figure_dir / "fig2_results.png"
    table1_path = table_dir / "table1_feature_contracts.tex"
    table2_path = table_dir / "table2_external_transfer.tex"
    _plot_results(primary, participant_deltas, dose, downstream, pdf_path, png_path)
    table1_path.write_text(_table1(), encoding="utf-8")
    table2_path.write_text(_table2(external), encoding="utf-8")

    generated = [pdf_path, png_path, table1_path, table2_path]
    return {
        "schema_name": "ictc-final-displays-v1",
        "source_execution_label": external_manifest["decision"]["label"],
        "exploratory_external_pairs_reported": external_manifest["decision"]["total_primary_pairs"],
        "manuscript_external_classification": "exploratory_directional_replication",
        "formal_transfer_verdict_reported": False,
        "external_labels_opened": external_manifest["external_labels_opened"],
        "participant_points_plotted": len(participant_deltas),
        "source_files": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in source_paths.values()
        },
        "generated_files": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in generated
        },
    }


def main() -> int:
    repo = Path(__file__).resolve().parents[2]
    figure_dir = repo / "paper_ictc2026" / "03_figures"
    table_dir = repo / "paper_ictc2026" / "04_tables"
    manifest = build(repo, figure_dir, table_dir)
    manifest_path = figure_dir / "final_displays_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "COMPLETE", "generated": len(manifest["generated_files"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
