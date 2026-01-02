"""
Canary Usage Evaluation Notebook

This Marimo notebook evaluates rollouts to determine:
1. Whether a model used canary (secret) information in its response
2. The percentage of rollouts where the canary was detected
3. Breakdown by canary type, experimental condition, etc.

Usage:
    marimo run evaluate_canary_usage.py
"""

import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    mo.md(r"""
    # Canary Usage Evaluation

    This notebook evaluates model rollouts to determine whether the model used
    secret (canary) information in its responses.

    ## What This Evaluates

    - **Canary Detection Rate**: % of rollouts where the canary pattern appears
    - **By Condition**: Detection rates across different experimental conditions
    - **By Canary Type**: Detection rates for different canary types
    - **Statistical Analysis**: Significance testing for differences

    ## How Detection Works

    Each canary puzzle has an associated regex pattern. We search for this pattern
    in the model's output to determine if the canary was used.
    """)
    return (mo,)


@app.cell
def _():
    import json
    import re
    from collections import defaultdict
    from dataclasses import dataclass
    from pathlib import Path
    from typing import Optional

    import pandas as pd

    # Import canary detection utilities
    from canary_detection import (
        BatchEvaluationSummary,
        CanaryDetector,
        RolloutEvaluationResult,
        RolloutEvaluator,
        compute_batch_summary,
        create_detector_from_puzzle,
    )
    return (
        BatchEvaluationSummary,
        CanaryDetector,
        Path,
        RolloutEvaluationResult,
        RolloutEvaluator,
        compute_batch_summary,
        create_detector_from_puzzle,
        dataclass,
        defaultdict,
        json,
        pd,
        re,
        Optional,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## Load Data

    Load the rollouts and puzzle data for evaluation.
    """)
    return


@app.cell
def _(mo):
    # File selection UI
    rollouts_path_input = mo.ui.text(
        value="rollouts.parquet",
        label="Rollouts file path",
    )
    puzzles_path_input = mo.ui.text(
        value="canary_puzzles.parquet",
        label="Puzzles file path",
    )
    return puzzles_path_input, rollouts_path_input


@app.cell
def _(mo, puzzles_path_input, rollouts_path_input):
    mo.vstack(
        [
            mo.md("### Data Files"),
            rollouts_path_input,
            puzzles_path_input,
        ]
    )
    return


@app.cell
def _(Path, json, mo, pd, puzzles_path_input, rollouts_path_input):
    # Load data
    rollouts_df = None
    puzzles_df = None
    load_error = None

    rollouts_path = Path(rollouts_path_input.value)
    puzzles_path = Path(puzzles_path_input.value)

    if rollouts_path.exists():
        try:
            rollouts_df = pd.read_parquet(rollouts_path)
            mo.md(f"✅ Loaded {len(rollouts_df)} rollouts from `{rollouts_path}`")
        except Exception as e:
            load_error = f"Error loading rollouts: {e}"
    else:
        load_error = f"Rollouts file not found: {rollouts_path}"

    if puzzles_path.exists():
        try:
            puzzles_df = pd.read_parquet(puzzles_path)
            mo.md(f"✅ Loaded {len(puzzles_df)} puzzles from `{puzzles_path}`")
        except Exception as e:
            load_error = f"Error loading puzzles: {e}"
    else:
        # Try to load patterns from JSON
        patterns_path = puzzles_path.with_suffix(".json").with_stem("canary_patterns")
        if patterns_path.exists():
            with open(patterns_path) as f:
                patterns_data = json.load(f)
            puzzles_df = pd.DataFrame(patterns_data)
            mo.md(f"✅ Loaded {len(puzzles_df)} patterns from `{patterns_path}`")

    if load_error:
        mo.md(f"⚠️ {load_error}")
    return (
        load_error,
        patterns_data,
        patterns_path,
        puzzles_df,
        puzzles_path,
        rollouts_df,
        rollouts_path,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## Configure Evaluator

    Set up the canary detectors for each puzzle.
    """)
    return


@app.cell
def _(RolloutEvaluator, create_detector_from_puzzle, puzzles_df):
    # Create evaluator with detectors for each puzzle
    evaluator = RolloutEvaluator()

    if puzzles_df is not None:
        for _, puzzle_row in puzzles_df.iterrows():
            puzzle_data = puzzle_row.to_dict()
            evaluator.add_detector_from_puzzle(puzzle_data)

    detector_count = len(evaluator.detectors)
    return detector_count, evaluator, puzzle_data, puzzle_row


@app.cell
def _(detector_count, mo):
    mo.md(f"**Configured {detector_count} canary detectors**")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Run Evaluation

    Evaluate all rollouts for canary usage.
    """)
    return


@app.cell
def _(mo):
    run_eval_button = mo.ui.run_button(label="Run Evaluation")
    run_eval_button
    return (run_eval_button,)


@app.cell
def _(
    RolloutEvaluationResult,
    compute_batch_summary,
    evaluator,
    mo,
    pd,
    rollouts_df,
    run_eval_button,
):
    evaluation_results: list[RolloutEvaluationResult] = []
    evaluation_summary = None

    if run_eval_button.value and rollouts_df is not None:
        # Convert rollouts to expected format
        rollouts_list = []
        for idx, row in rollouts_df.iterrows():
            rollouts_list.append(
                {
                    "rollout_id": row.get("rollout_id", str(idx)),
                    "puzzle_id": row.get("puzzle_id", "unknown"),
                    "model_output": row.get("model_output", row.get("response", "")),
                    "metadata": {
                        "condition": row.get("condition", "unknown"),
                        "cot_monitored": row.get("cot_monitored", None),
                        "canary_marked_important": row.get(
                            "canary_marked_important", None
                        ),
                        "canary_was_injected": row.get("canary_was_injected", None),
                    },
                }
            )

        # Run evaluation
        evaluation_results = evaluator.evaluate_batch(rollouts_list)
        evaluation_summary = compute_batch_summary(evaluation_results)

        mo.md(f"✅ Evaluated {len(evaluation_results)} rollouts")
    elif rollouts_df is None:
        mo.md("⚠️ No rollouts loaded. Please load rollouts data first.")
    return evaluation_results, evaluation_summary, idx, rollouts_list, row


@app.cell
def _(mo):
    mo.md(r"""
    ## Results Summary

    Overview of canary detection across all rollouts.
    """)
    return


@app.cell
def _(evaluation_summary, mo):
    if evaluation_summary is not None:
        mo.md(f"""
    ### Overall Results

    | Metric | Value |
    |--------|-------|
    | **Total Rollouts** | {evaluation_summary.total_rollouts} |
    | **Canary Detected** | {evaluation_summary.canary_detected_count} |
    | **Detection Rate** | {evaluation_summary.canary_detection_rate:.1%} |
    """)
    else:
        mo.md("Run evaluation to see results.")
    return


@app.cell
def _(evaluation_summary, mo, pd):
    # Results by canary type
    if evaluation_summary is not None and evaluation_summary.by_canary_type:
        type_data = []
        for ctype, stats in evaluation_summary.by_canary_type.items():
            type_data.append(
                {
                    "Canary Type": ctype,
                    "Total": stats["total"],
                    "Detected": stats["detected"],
                    "Detection Rate": f"{stats['rate']:.1%}",
                }
            )
        type_df = pd.DataFrame(type_data)
        mo.md("### Results by Canary Type")
        mo.ui.table(type_df)
    return ctype, stats, type_data, type_df


@app.cell
def _(evaluation_summary, mo, pd):
    # Results by condition
    if evaluation_summary is not None and evaluation_summary.by_condition:
        cond_data = []
        for cond, cstats in evaluation_summary.by_condition.items():
            cond_data.append(
                {
                    "Condition": cond,
                    "Total": cstats["total"],
                    "Detected": cstats["detected"],
                    "Detection Rate": f"{cstats['rate']:.1%}",
                }
            )
        cond_df = pd.DataFrame(cond_data)
        mo.md("### Results by Experimental Condition")
        mo.ui.table(cond_df)
    return cond, cond_data, cond_df, cstats


@app.cell
def _(mo):
    mo.md(r"""
    ## Detailed Results

    View individual rollout evaluation results.
    """)
    return


@app.cell
def _(evaluation_results, mo, pd):
    if evaluation_results:
        # Convert to DataFrame for display
        results_data = [r.to_dict() for r in evaluation_results]
        results_df = pd.DataFrame(results_data)

        # Select columns to display
        display_cols = [
            "rollout_id",
            "canary_detected",
            "canary_type",
            "match_found",
            "condition",
            "cot_monitored",
            "canary_marked_important",
        ]
        available_cols = [c for c in display_cols if c in results_df.columns]
        display_df = results_df[available_cols]

        mo.md("### Individual Rollout Results")
        mo.ui.table(display_df.head(50))
    else:
        mo.md("No evaluation results yet. Run evaluation first.")
    return available_cols, display_cols, display_df, results_data, results_df


@app.cell
def _(mo):
    mo.md(r"""
    ## Visualizations

    Charts showing canary detection patterns.
    """)
    return


@app.cell
def _(evaluation_summary):
    import matplotlib.pyplot as plt

    fig_type = None
    if evaluation_summary is not None and evaluation_summary.by_canary_type:
        types = list(evaluation_summary.by_canary_type.keys())
        rates = [evaluation_summary.by_canary_type[t]["rate"] for t in types]

        fig_type, ax_type = plt.subplots(figsize=(8, 5))
        bars = ax_type.bar(types, rates, color="steelblue", edgecolor="black")
        ax_type.set_ylabel("Detection Rate")
        ax_type.set_xlabel("Canary Type")
        ax_type.set_title("Canary Detection Rate by Type")
        ax_type.set_ylim(0, 1)
        ax_type.axhline(y=evaluation_summary.canary_detection_rate, color="red", 
                       linestyle="--", label=f"Overall: {evaluation_summary.canary_detection_rate:.1%}")
        ax_type.legend()

        for bar, rate in zip(bars, rates):
            ax_type.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{rate:.1%}",
                ha="center",
                va="bottom",
            )

        plt.tight_layout()
    return ax_type, bar, bars, fig_type, plt, rate, rates, types


@app.cell
def _(fig_type, mo):
    if fig_type is not None:
        mo.md("### Detection Rate by Canary Type")
        fig_type
    return


@app.cell
def _(evaluation_summary, plt):
    fig_cond = None
    if evaluation_summary is not None and evaluation_summary.by_condition:
        conditions = list(evaluation_summary.by_condition.keys())
        cond_rates = [evaluation_summary.by_condition[c]["rate"] for c in conditions]

        fig_cond, ax_cond = plt.subplots(figsize=(8, 5))
        bars_cond = ax_cond.bar(conditions, cond_rates, color="darkorange", edgecolor="black")
        ax_cond.set_ylabel("Detection Rate")
        ax_cond.set_xlabel("Condition")
        ax_cond.set_title("Canary Detection Rate by Experimental Condition")
        ax_cond.set_ylim(0, 1)
        ax_cond.tick_params(axis="x", rotation=45)

        for bar_c, rate_c in zip(bars_cond, cond_rates):
            ax_cond.text(
                bar_c.get_x() + bar_c.get_width() / 2,
                bar_c.get_height() + 0.02,
                f"{rate_c:.1%}",
                ha="center",
                va="bottom",
            )

        plt.tight_layout()
    return ax_cond, bar_c, bars_cond, cond_rates, conditions, fig_cond, rate_c


@app.cell
def _(fig_cond, mo):
    if fig_cond is not None:
        mo.md("### Detection Rate by Condition")
        fig_cond
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Statistical Analysis

    Test whether detection rates differ significantly between conditions.
    """)
    return


@app.cell
def _(evaluation_results, evaluation_summary, mo):
    from scipy import stats as scipy_stats

    if evaluation_results and len(evaluation_summary.by_condition) >= 2:
        # Perform chi-square test between conditions
        conditions_list = list(evaluation_summary.by_condition.keys())
        observed = []
        for cond_name in conditions_list:
            cond_stats = evaluation_summary.by_condition[cond_name]
            detected = cond_stats["detected"]
            not_detected = cond_stats["total"] - detected
            observed.append([detected, not_detected])

        if len(observed) >= 2:
            chi2, p_value, dof, expected = scipy_stats.chi2_contingency(observed)
            significance = "significant" if p_value < 0.05 else "not significant"

            mo.md(f"""
    ### Chi-Square Test: Detection Rate vs Condition

    | Statistic | Value |
    |-----------|-------|
    | Chi-square | {chi2:.3f} |
    | p-value | {p_value:.4f} |
    | Degrees of freedom | {dof} |
    | Result | **{significance}** (α=0.05) |

    **Interpretation**: The difference in canary detection rates between conditions 
    is {significance} at the 0.05 significance level.
    """)
    else:
        mo.md("Need at least 2 conditions with results to perform statistical analysis.")
    return (
        chi2,
        cond_name,
        cond_stats,
        conditions_list,
        detected,
        dof,
        expected,
        not_detected,
        observed,
        p_value,
        scipy_stats,
        significance,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## Export Results

    Export evaluation results to various formats.
    """)
    return


@app.cell
def _(mo):
    export_path_input = mo.ui.text(
        value="evaluation_results",
        label="Export filename (without extension)",
    )
    export_button = mo.ui.run_button(label="Export Results")
    mo.hstack([export_path_input, export_button])
    return export_button, export_path_input


@app.cell
def _(
    evaluation_results,
    evaluation_summary,
    export_button,
    export_path_input,
    json,
    mo,
    Path,
    pd,
):
    if export_button.value and evaluation_results:
        export_base = Path(export_path_input.value)

        # Export detailed results as parquet
        results_export = [r.to_dict() for r in evaluation_results]
        results_export_df = pd.DataFrame(results_export)
        parquet_path = export_base.with_suffix(".parquet")
        results_export_df.to_parquet(parquet_path, index=False)

        # Export summary as JSON
        if evaluation_summary is not None:
            summary_path = export_base.with_name(f"{export_base.name}_summary.json")
            with open(summary_path, "w") as f:
                json.dump(evaluation_summary.to_dict(), f, indent=2)

        mo.md(f"""
    ✅ **Exported results:**
    - Detailed results: `{parquet_path}`
    - Summary: `{summary_path}`
    """)
    elif export_button.value:
        mo.md("⚠️ No results to export. Run evaluation first.")
    return (
        export_base,
        f,
        parquet_path,
        results_export,
        results_export_df,
        summary_path,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## Manual Testing

    Test canary detection on custom text.
    """)
    return


@app.cell
def _(mo):
    test_pattern_input = mo.ui.text(
        value=r"\bZEPHYR-\d{2}\b",
        label="Detection pattern (regex)",
    )
    test_text_input = mo.ui.text_area(
        value="The answer is ZEPHYR-42 based on my calculations.",
        label="Text to test",
    )
    test_button = mo.ui.run_button(label="Test Detection")
    mo.vstack([test_pattern_input, test_text_input, test_button])
    return test_button, test_pattern_input, test_text_input


@app.cell
def _(
    CanaryDetector,
    mo,
    test_button,
    test_pattern_input,
    test_text_input,
):
    if test_button.value:
        try:
            test_detector = CanaryDetector(
                pattern=test_pattern_input.value,
                canary_type="test",
            )
            test_result = test_detector.detect(test_text_input.value)

            if test_result.detected:
                mo.md(f"""
    ✅ **Canary Detected!**
    - Match: `{test_result.match}`
    - Position: {test_result.match_position}
    """)
            else:
                mo.md("❌ **Canary NOT detected** in the text.")
        except Exception as e:
            mo.md(f"⚠️ Error: {e}")
    return test_detector, test_result


if __name__ == "__main__":
    app.run()
