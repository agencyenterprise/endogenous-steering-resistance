"""
Generate LaTeX tables for self-correction activation statistics.

Usage:
    python generate_tables.py
"""

import json
from pathlib import Path


def format_value(val: float, decimals: int = 3) -> str:
    """Format a value for LaTeX output."""
    if abs(val) < 0.001:
        return f"{val:.2e}"
    return f"{val:.{decimals}f}"


def format_pvalue(p: float) -> str:
    """Format p-value with significance markers."""
    if p < 0.001:
        return f"$<$0.001***"
    elif p < 0.01:
        return f"{p:.3f}**"
    elif p < 0.05:
        return f"{p:.3f}*"
    else:
        return f"{p:.3f}"


def generate_summary_table(results: dict) -> str:
    """Generate LaTeX table summarizing group statistics."""

    otd = results["otd_group"]
    bt = results["backtracking_group"]

    latex = r"""\begin{table}[htbp]
\centering
\caption{Self-correction activation statistics. Mean activation per region across """ + str(results["n_episodes"]) + r""" episodes.}
\label{tab:self-correction-activations}
\begin{tabular}{l c c c c c}
\toprule
Group & Off-topic & Correction & On-topic & $p$-value & Cohen's $d$ \\
\midrule
"""

    # OTD row
    latex += f"OTDs ({otd['n_latents']}) & "
    latex += f"{format_value(otd['mean_off_topic'])} $\\pm$ {format_value(otd['std_off_topic'])} & "
    latex += f"{format_value(otd['mean_correction'])} $\\pm$ {format_value(otd['std_correction'])} & "
    latex += f"{format_value(otd['mean_on_topic'])} $\\pm$ {format_value(otd['std_on_topic'])} & "
    latex += f"{format_pvalue(otd['wilcoxon_p'])} & "
    latex += f"{format_value(otd['cohens_d'], 2)} \\\\\n"

    # Backtracking row
    latex += f"Backtracking ({bt['n_latents']}) & "
    latex += f"{format_value(bt['mean_off_topic'])} $\\pm$ {format_value(bt['std_off_topic'])} & "
    latex += f"{format_value(bt['mean_correction'])} $\\pm$ {format_value(bt['std_correction'])} & "
    latex += f"{format_value(bt['mean_on_topic'])} $\\pm$ {format_value(bt['std_on_topic'])} & "
    latex += f"{format_pvalue(bt['wilcoxon_p'])} & "
    latex += f"{format_value(bt['cohens_d'], 2)} \\\\\n"

    latex += r"""\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item Note: $p$-values from one-tailed Wilcoxon signed-rank test (off-topic $>$ on-topic).
\item Significance: * $p<0.05$, ** $p<0.01$, *** $p<0.001$
\end{tablenotes}
\end{table}
"""

    return latex


def generate_detailed_otd_table(results: dict) -> str:
    """Generate detailed LaTeX table for OTD latents."""

    latent_stats = results["otd_latent_stats"]

    # Sort by effect size (Cohen's d)
    latent_stats = sorted(latent_stats, key=lambda x: x["cohens_d"], reverse=True)

    latex = r"""\begin{table}[htbp]
\centering
\caption{Off-Topic Detector activation statistics by latent.}
\label{tab:otd-latent-stats}
\begin{tabular}{r l c c c c}
\toprule
Index & Label & Off-topic & On-topic & $p$ & $d$ \\
\midrule
"""

    for stat in latent_stats:  # All latents, sorted by effect size
        label = stat["latent_label"][:40] + "..." if len(stat["latent_label"]) > 40 else stat["latent_label"]
        label = label.replace("_", "\\_").replace("&", "\\&")

        latex += f"{stat['latent_idx']} & {label} & "
        latex += f"{format_value(stat['off_topic']['mean'])} & "
        latex += f"{format_value(stat['on_topic']['mean'])} & "
        latex += f"{format_pvalue(stat['wilcoxon_p'])} & "
        latex += f"{format_value(stat['cohens_d'], 2)} \\\\\n"

    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""

    return latex


def generate_detailed_backtracking_table(results: dict) -> str:
    """Generate detailed LaTeX table for backtracking latents."""

    latent_stats = results["backtracking_latent_stats"]

    # Sort by correction region activation (highest first)
    latent_stats = sorted(latent_stats, key=lambda x: x["correction"]["mean"], reverse=True)

    latex = r"""\begin{table}[htbp]
\centering
\caption{Backtracking latent activation statistics by latent. Sorted by correction region activation.}
\label{tab:backtracking-latent-stats}
\begin{tabular}{r l c c c}
\toprule
Index & Label & Off-topic & Correction & On-topic \\
\midrule
"""

    for stat in latent_stats:  # All latents, sorted by correction activation
        label = stat["latent_label"][:40] + "..." if len(stat["latent_label"]) > 40 else stat["latent_label"]
        label = label.replace("_", "\\_").replace("&", "\\&")

        latex += f"{stat['latent_idx']} & {label} & "
        latex += f"{format_value(stat['off_topic']['mean'])} & "
        latex += f"{format_value(stat['correction']['mean'])} & "
        latex += f"{format_value(stat['on_topic']['mean'])} \\\\\n"

    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""

    return latex


def main():
    # Paths
    base_dir = Path(__file__).parent.parent
    stats_dir = base_dir / "data" / "experiment_results" / "claude_haiku_4_5_20251001_judge" / "activation_stats"
    results_file = stats_dir / "analysis_results.json"
    output_dir = stats_dir / "tables"

    if not results_file.exists():
        print(f"Error: Results file not found: {results_file}")
        print("Please run analyze_activations.py first.")
        return

    with open(results_file) as f:
        results = json.load(f)

    print("Generating LaTeX tables...")

    # Generate tables
    summary_table = generate_summary_table(results)
    otd_table = generate_detailed_otd_table(results)
    bt_table = generate_detailed_backtracking_table(results)

    # Save tables
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "summary_table.tex", "w") as f:
        f.write(summary_table)
    print(f"  Saved: {output_dir / 'summary_table.tex'}")

    with open(output_dir / "otd_table.tex", "w") as f:
        f.write(otd_table)
    print(f"  Saved: {output_dir / 'otd_table.tex'}")

    with open(output_dir / "backtracking_table.tex", "w") as f:
        f.write(bt_table)
    print(f"  Saved: {output_dir / 'backtracking_table.tex'}")

    # Print summary table to console
    print("\n" + "=" * 80)
    print("Summary Table:")
    print("=" * 80)
    print(summary_table)


if __name__ == "__main__":
    main()
