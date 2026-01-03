"""
Canary Usage Detection with Linear Probes

This Marimo notebook trains linear probes on model activations to detect whether
the model is "deciding to cheat" by using secret canary information.

## Experiment Design

We collect activations from the model in two conditions:
1. **Canary Present**: Model sees the canary instruction and uses it
2. **Canary Absent**: Model doesn't see the canary instruction

We train a linear probe (logistic regression) to classify activations and detect
when the model has "internalized" the canary and intends to use it.

## Key Questions
- At which layer does the "decision to use canary" become detectable?
- Can we detect this intent before the model outputs the canary?
- How does monitoring (telling model CoT is watched) affect detectability?

Usage:
    marimo run canary_probe_experiment.py
"""

import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    mo.md(r"""
    # Canary Usage Detection with Linear Probes

    This experiment uses **TransformerLens** to:
    1. Collect activations from Qwen3-4B when processing canary puzzles
    2. Train linear probes to detect "intent to use canary"
    3. Analyze at which layers this decision becomes detectable

    ## Why This Matters

    If we can detect the "decision to cheat" in the model's activations:
    - We could potentially intervene before the model outputs the canary
    - We gain insight into how models represent and use secret information
    - We can study whether monitoring awareness changes this representation
    """)
    return (mo,)


@app.cell
def _():
    import json
    import random
    from dataclasses import dataclass
    from pathlib import Path
    from typing import Optional

    import numpy as np
    import pandas as pd
    import torch
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
    from sklearn.model_selection import cross_val_score, train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from tqdm.auto import tqdm
    from transformer_lens import HookedTransformer
    from transformer_lens import utils as tl_utils
    return (
        HookedTransformer,
        LogisticRegression,
        Path,
        StandardScaler,
        cross_val_score,
        dataclass,
        json,
        make_pipeline,
        np,
        pd,
        tl_utils,
        torch,
        tqdm,
        train_test_split,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## Configuration

    Select the model and experiment parameters.
    """)
    return


@app.cell
def _(mo):
    model_name_input = mo.ui.dropdown(
        options=[
            "Qwen/Qwen3-4B",
        ],
        value="Qwen/Qwen3-4B",
        label="Model",
    )
    device_input = mo.ui.dropdown(
        options=["cuda", "mps", "cpu"],
        value="mps",
        label="Device",
    )
    puzzles_path_input = mo.ui.text(
        value="canary_puzzles.parquet",
        label="Puzzles file",
    )
    max_samples_input = mo.ui.number(
        value=50,
        start=10,
        stop=500,
        step=10,
        label="Max samples per condition",
    )
    return (
        device_input,
        max_samples_input,
        model_name_input,
        puzzles_path_input,
    )


@app.cell
def _(
    device_input,
    max_samples_input,
    mo,
    model_name_input,
    puzzles_path_input,
):
    mo.vstack([
        mo.md("### Experiment Configuration"),
        mo.hstack([model_name_input, device_input]),
        mo.hstack([puzzles_path_input, max_samples_input]),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Load Model

    Load the model with TransformerLens for activation caching.
    """)
    return


@app.cell
def _(HookedTransformer, device_input, mo, model_name_input, torch):
    # Load model
    _device = device_input.value
    _model_name = model_name_input.value

    mo.md(f"Loading **{_model_name}** on **{_device}**...")

    model = HookedTransformer.from_pretrained(
        _model_name,
        device=_device,
        dtype=torch.float16
    )
    model.eval()

    n_layers = model.cfg.n_layers
    d_model = model.cfg.d_model

    mo.md(f"✅ Loaded model with **{n_layers} layers**, **d_model={d_model}**")
    return model, n_layers


@app.cell
def _(mo):
    mo.md(r"""
    ## Load Puzzles

    Load canary puzzles and prepare prompts for both conditions.
    """)
    return


@app.cell
def _(Path, max_samples_input, mo, pd, puzzles_path_input):
    # Load puzzles
    puzzles_df = None
    puzzles_path = Path(puzzles_path_input.value)

    if puzzles_path.exists():
        puzzles_df = pd.read_parquet(puzzles_path)
        # Sample if we have too many
        max_samples = int(max_samples_input.value)
        if len(puzzles_df) > max_samples:
            puzzles_df = puzzles_df.sample(n=max_samples, random_state=42)
        mo.md(f"✅ Loaded **{len(puzzles_df)}** puzzles")
    else:
        mo.md(f"⚠️ Puzzles file not found: {puzzles_path}")
    return (puzzles_df,)


@app.cell
def _(dataclass):
    @dataclass
    class ProbeDatapoint:
        """A single datapoint for probe training."""
        prompt: str
        label: int  # 1 = canary present, 0 = canary absent
        puzzle_id: str
        canary_type: str
        condition: str  # "with_canary" or "without_canary"
    return (ProbeDatapoint,)


@app.cell
def _(ProbeDatapoint, puzzles_df):
    def prepare_probe_data(df) -> list[ProbeDatapoint]:
        """Prepare paired prompts for probe training.

        For each puzzle, we create two datapoints:
        1. Prompt WITH canary instruction (label=1)
        2. Prompt WITHOUT canary instruction (label=0)
        """
        datapoints = []

        for _, row in df.iterrows():
            puzzle_id = row.get("puzzle_id", "unknown")
            canary_type = row.get("canary_type", "unknown")

            # Prompt with canary
            problem_with_canary = row.get("problem_with_canary", "")
            if problem_with_canary:
                datapoints.append(ProbeDatapoint(
                    prompt=f"Solve this problem:\n\n{problem_with_canary}\n\nAnswer:",
                    label=1,
                    puzzle_id=puzzle_id,
                    canary_type=canary_type,
                    condition="with_canary",
                ))

            # Prompt without canary
            problem_without_canary = row.get("problem_statement", "")
            if problem_without_canary:
                datapoints.append(ProbeDatapoint(
                    prompt=f"Solve this problem:\n\n{problem_without_canary}\n\nAnswer:",
                    label=0,
                    puzzle_id=puzzle_id,
                    canary_type=canary_type,
                    condition="without_canary",
                ))

        return datapoints

    # Prepare data
    probe_data = []
    if puzzles_df is not None and len(puzzles_df) > 0:
        probe_data = prepare_probe_data(puzzles_df)
    return (probe_data,)


@app.cell
def _(mo, probe_data):
    if probe_data:
        n_with = sum(1 for d in probe_data if d.label == 1)
        n_without = sum(1 for d in probe_data if d.label == 0)
        mo.md(f"""
    ### Prepared Probe Data
    - **With canary**: {n_with} samples (label=1)
    - **Without canary**: {n_without} samples (label=0)
    - **Total**: {len(probe_data)} samples
    """)
    else:
        mo.md("⚠️ No probe data prepared. Load puzzles first.")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Collect Activations

    Run prompts through the model and cache activations at each layer.
    We collect the **last token** residual stream activations (pre-prediction position).
    """)
    return


@app.cell
def _(mo):
    collect_button = mo.ui.run_button(label="Collect Activations")
    layer_stride_input = mo.ui.dropdown(
        options=[1, 2, 4],
        value=1,
        label="Layer stride (collect every N layers)",
    )
    mo.hstack([collect_button, layer_stride_input])
    return collect_button, layer_stride_input


@app.cell
def _(
    collect_button,
    layer_stride_input,
    mo,
    model,
    n_layers,
    np,
    probe_data,
    tl_utils,
    torch,
    tqdm,
):
    activations_by_layer = {}
    labels_array = None
    layer_indices = []

    if collect_button.value and probe_data:
        stride = int(layer_stride_input.value)
        layer_indices = list(range(0, n_layers, stride))

        # Initialize storage
        all_activations = {li: [] for li in layer_indices}
        all_labels = []

        mo.md(f"Collecting activations at layers: {layer_indices}")

        with torch.no_grad():
            for dp in tqdm(probe_data, desc="Collecting activations"):
                # Tokenize
                tokens = model.to_tokens(dp.prompt, prepend_bos=True)

                # Run with cache
                _, cache = model.run_with_cache(tokens)

                # Extract last-token residual stream at each layer
                for li in layer_indices:
                    act_name = tl_utils.get_act_name("resid_post", li)
                    resid = cache[act_name]  # (batch, seq, d_model)
                    last_token_act = resid[0, -1, :].cpu().numpy()
                    all_activations[li].append(last_token_act)

                all_labels.append(dp.label)

                # Clear cache to save memory
                del cache

        # Stack into arrays
        activations_by_layer = {
            li: np.stack(acts, axis=0) for li, acts in all_activations.items()
        }
        labels_array = np.array(all_labels)

        mo.md(f"✅ Collected activations: shape per layer = {activations_by_layer[layer_indices[0]].shape}")
    elif not probe_data:
        mo.md("⚠️ No probe data available. Load puzzles first.")
    return activations_by_layer, labels_array, layer_indices


@app.cell
def _(mo):
    mo.md(r"""
    ## Train Linear Probes

    Train a logistic regression probe at each layer to classify
    "canary present" vs "canary absent" based on activations.
    """)
    return


@app.cell
def _(mo):
    train_button = mo.ui.run_button(label="Train Probes")
    train_button
    return (train_button,)


@app.cell
def _(
    LogisticRegression,
    StandardScaler,
    activations_by_layer,
    cross_val_score,
    labels_array,
    layer_indices,
    make_pipeline,
    mo,
    np,
    train_button,
    train_test_split,
):
    probe_results = {}

    if train_button.value and activations_by_layer and labels_array is not None:
        mo.md("Training probes at each layer...")

        for li in layer_indices:
            X = activations_by_layer[li]
            y = labels_array

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            # Create and train probe
            probe = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=1000, solver="lbfgs", C=1.0)
            )
            probe.fit(X_train, y_train)

            # Evaluate
            train_acc = probe.score(X_train, y_train)
            test_acc = probe.score(X_test, y_test)

            # Cross-validation score
            cv_scores = cross_val_score(probe, X, y, cv=5, scoring="accuracy")

            # Get probe weights
            lr = probe.named_steps["logisticregression"]
            weights = lr.coef_[0]

            probe_results[li] = {
                "probe": probe,
                "train_acc": train_acc,
                "test_acc": test_acc,
                "cv_mean": np.mean(cv_scores),
                "cv_std": np.std(cv_scores),
                "weights": weights,
                "weight_norm": np.linalg.norm(weights),
            }

        mo.md("✅ Trained probes at all layers")
    elif not activations_by_layer:
        mo.md("⚠️ Collect activations first.")
    return (probe_results,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Results

    Probe accuracy by layer - higher accuracy means the "canary intent" is more
    detectable at that layer.
    """)
    return


@app.cell
def _(layer_indices, mo, pd, probe_results):
    if probe_results:
        results_data = []
        for li in layer_indices:
            if li in probe_results:
                r = probe_results[li]
                results_data.append({
                    "Layer": li,
                    "Train Acc": f"{r['train_acc']:.3f}",
                    "Test Acc": f"{r['test_acc']:.3f}",
                    "CV Mean": f"{r['cv_mean']:.3f}",
                    "CV Std": f"{r['cv_std']:.3f}",
                    "Weight Norm": f"{r['weight_norm']:.2f}",
                })

        results_df = pd.DataFrame(results_data)
        mo.md("### Probe Accuracy by Layer")
        mo.ui.table(results_df)
    else:
        mo.md("Train probes to see results.")
    return


@app.cell
def _(layer_indices, probe_results):
    import matplotlib.pyplot as plt

    fig_acc = None
    if probe_results:
        layers = [li for li in layer_indices if li in probe_results]
        test_accs = [probe_results[li]["test_acc"] for li in layers]
        cv_means = [probe_results[li]["cv_mean"] for li in layers]
        cv_stds = [probe_results[li]["cv_std"] for li in layers]

        fig_acc, ax = plt.subplots(figsize=(10, 5))

        ax.plot(layers, test_accs, "o-", label="Test Accuracy", linewidth=2)
        ax.fill_between(
            layers,
            [m - s for m, s in zip(cv_means, cv_stds)],
            [m + s for m, s in zip(cv_means, cv_stds)],
            alpha=0.3,
            label="CV ± 1 std",
        )
        ax.plot(layers, cv_means, "s--", label="CV Mean", alpha=0.7)

        ax.axhline(y=0.5, color="red", linestyle=":", label="Chance level")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Accuracy")
        ax.set_title("Canary Detection Probe Accuracy by Layer")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0.4, 1.05)

        plt.tight_layout()
    return fig_acc, plt


@app.cell
def _(fig_acc, mo):
    if fig_acc is not None:
        mo.md("### Probe Accuracy Visualization")
        fig_acc
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Analysis: Best Layer for Detection

    Which layer(s) are best for detecting "canary intent"?
    """)
    return


@app.cell
def _(layer_indices, mo, probe_results):
    if probe_results:
        # Find best layer
        best_layer = max(
            [li for li in layer_indices if li in probe_results],
            key=lambda li: probe_results[li]["test_acc"]
        )
        best_acc = probe_results[best_layer]["test_acc"]

        # Find where detection becomes reliable (>70% accuracy)
        reliable_layers = [
            li for li in layer_indices 
            if li in probe_results and probe_results[li]["test_acc"] > 0.7
        ]
        first_reliable = min(reliable_layers) if reliable_layers else None

        mo.md(f"""
    ### Key Findings

    | Metric | Value |
    |--------|-------|
    | **Best Layer** | {best_layer} |
    | **Best Test Accuracy** | {best_acc:.1%} |
    | **First Reliable Layer (>70%)** | {first_reliable if first_reliable is not None else "None"} |
    | **Reliable Layers** | {reliable_layers if reliable_layers else "None"} |

    **Interpretation**:
    - If accuracy is near 50%, the probe cannot distinguish canary/no-canary → model doesn't encode this differently
    - If accuracy is high (>80%), the model's internal representation clearly differs when a canary is present
    - Earlier layers detecting → the "decision" is made early in processing
    - Later layers detecting → the "decision" emerges during reasoning
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Probe Direction Analysis

    The probe weights define a "canary direction" in activation space.
    We can analyze this direction's properties.
    """)
    return


@app.cell
def _(layer_indices, plt, probe_results):
    fig_weights = None
    if probe_results:
        # Plot weight norms by layer
        layers_w = [li for li in layer_indices if li in probe_results]
        weight_norms = [probe_results[li]["weight_norm"] for li in layers_w]

        fig_weights, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Weight norm by layer
        axes[0].bar(layers_w, weight_norms, color="steelblue", edgecolor="black")
        axes[0].set_xlabel("Layer")
        axes[0].set_ylabel("Weight Norm (L2)")
        axes[0].set_title("Probe Weight Magnitude by Layer")
        axes[0].grid(True, alpha=0.3)

        # Weight distribution for best layer
        if probe_results:
            best_layer_w = max(layers_w, key=lambda li: probe_results[li]["test_acc"])
            best_weights = probe_results[best_layer_w]["weights"]
            axes[1].hist(best_weights, bins=50, color="coral", edgecolor="black", alpha=0.7)
            axes[1].axvline(x=0, color="black", linestyle="--")
            axes[1].set_xlabel("Weight Value")
            axes[1].set_ylabel("Count")
            axes[1].set_title(f"Weight Distribution (Layer {best_layer_w})")
            axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
    return (fig_weights,)


@app.cell
def _(fig_weights, mo):
    if fig_weights is not None:
        mo.md("### Probe Weight Analysis")
        fig_weights
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Export Results

    Save probe results and weights for further analysis.
    """)
    return


@app.cell
def _(mo):
    export_button = mo.ui.run_button(label="Export Results")
    export_path_input = mo.ui.text(
        value="probe_results",
        label="Export filename (without extension)",
    )
    mo.hstack([export_path_input, export_button])
    return export_button, export_path_input


@app.cell
def _(
    Path,
    export_button,
    export_path_input,
    json,
    layer_indices,
    mo,
    np,
    probe_results,
):
    if export_button.value and probe_results:
        export_base = Path(export_path_input.value)

        # Export summary as JSON
        summary = {}
        for li in layer_indices:
            if li in probe_results:
                r = probe_results[li]
                summary[str(li)] = {
                    "train_acc": float(r["train_acc"]),
                    "test_acc": float(r["test_acc"]),
                    "cv_mean": float(r["cv_mean"]),
                    "cv_std": float(r["cv_std"]),
                    "weight_norm": float(r["weight_norm"]),
                }

        summary_path = export_base.with_suffix(".json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        # Export weights as numpy
        weights_path = export_base.with_name(f"{export_base.name}_weights.npz")
        weights_dict = {
            f"layer_{li}": probe_results[li]["weights"]
            for li in layer_indices if li in probe_results
        }
        np.savez(weights_path, **weights_dict)

        mo.md(f"""
    ✅ **Exported results:**
    - Summary: `{summary_path}`
    - Weights: `{weights_path}`
    """)
    elif export_button.value:
        mo.md("⚠️ Train probes first before exporting.")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Next Steps

    With trained probes, you can:

    1. **Intervention Experiments**: Use the probe direction to steer the model
       away from using the canary (subtract the direction during generation)

    2. **Real-time Detection**: Apply the probe during generation to detect
       when the model "decides" to use secret information

    3. **Compare Conditions**: Train separate probes for CoT-monitored vs
       non-monitored conditions to see if monitoring changes representations

    4. **Generalization**: Test if probes trained on one canary type generalize
       to other canary types
    """)
    return


if __name__ == "__main__":
    app.run()
