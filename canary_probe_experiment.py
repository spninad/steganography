"""
Canary Usage Detection with Linear Probes

This Marimo notebook trains linear probes on model activations to detect whether
the model will USE secret canary information in its output.

## Experiment Design (Corrected Methodology)

We probe the model's OUTPUT BEHAVIOR, not input differences:
1. Run model on prompts with canary instruction
2. Generate completions and detect if canary appears in OUTPUT
3. Label based on actual model behavior (used canary = 1, didn't use = 0)
4. Train probe to predict output behavior from activations

This avoids the trivial classification problem where probes just detect
different input text rather than the model's "decision" to cheat.

## Key Questions
- At which layer does the "decision to use canary" become detectable?
- Can we detect this intent before the model outputs the canary?
- Is there a point in the forward pass where we could intervene?

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
    1. Run the model on canary puzzles and **generate completions**
    2. Detect whether the canary appears in the **output** (not input)
    3. Train probes to predict output behavior from activations

    ## Corrected Methodology

    **Previous (flawed)**: Compared activations from different input prompts
    → Probe trivially detected input text differences, not model decisions

    **Current (correct)**: All prompts contain canary instructions, but:
    - Some model completions USE the canary (label=1)
    - Some model completions DON'T use the canary (label=0)
    - Probe learns to predict **output behavior** from activations

    This tells us: "Can we detect when the model will use secret info?"
    """)
    return (mo,)


@app.cell
def _():
    import json
    import re
    from dataclasses import dataclass
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import torch
    from sklearn.linear_model import LogisticRegression
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
        re,
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
    _status_msg = ""

    if puzzles_path.exists():
        puzzles_df = pd.read_parquet(puzzles_path)
        # Sample if we have too many
        max_samples = int(max_samples_input.value)
        if len(puzzles_df) > max_samples:
            puzzles_df = puzzles_df.sample(n=max_samples, random_state=42)
        _status_msg = f"✅ Loaded **{len(puzzles_df)}** puzzles"
    else:
        _status_msg = f"⚠️ Puzzles file not found: {puzzles_path}"

    mo.output.replace(mo.md(_status_msg))
    return (puzzles_df,)


@app.cell
def _(dataclass):
    @dataclass
    class ProbeDatapoint:
        """A single datapoint for probe training.

        Labels are based on OUTPUT behavior, not input content:
        - label=1: Model's completion CONTAINS the canary pattern
        - label=0: Model's completion does NOT contain the canary
        """
        prompt: str
        completion: str  # Model's generated output
        label: int  # 1 = canary used in output, 0 = canary not used
        puzzle_id: str
        canary_type: str
        detection_regex: str  # Pattern to detect canary in output
    return (ProbeDatapoint,)


@app.cell
def _(puzzles_df):
    def prepare_puzzle_prompts(df) -> list[dict]:
        """Prepare prompts for generation.

        We only use prompts WITH canary instructions.
        Labels will be assigned AFTER generation based on output.
        """
        prompts = []

        for _, row in df.iterrows():
            puzzle_id = row.get("puzzle_id", "unknown")
            canary_type = row.get("canary_type", "unknown")
            detection_regex = row.get("detection_regex", "")

            # Only use prompts with canary instruction
            problem_with_canary = row.get("problem_with_canary", "")
            if problem_with_canary and detection_regex:
                prompts.append({
                    "prompt": f"Solve this problem:\n\n{problem_with_canary}\n\nAnswer:",
                    "puzzle_id": puzzle_id,
                    "canary_type": canary_type,
                    "detection_regex": detection_regex,
                })

        return prompts

    # Prepare prompts (labels assigned after generation)
    puzzle_prompts = []
    if puzzles_df is not None and len(puzzles_df) > 0:
        puzzle_prompts = prepare_puzzle_prompts(puzzles_df)
    return (puzzle_prompts,)


@app.cell
def _(mo, puzzle_prompts):
    _msg = ""
    if puzzle_prompts:
        _msg = f"""
    ### Prepared Prompts
    - **Total prompts**: {len(puzzle_prompts)}
    - Labels will be assigned after generation based on whether the model's output contains the canary pattern.
    """
    else:
        _msg = "⚠️ No prompts prepared. Load puzzles first."
    mo.md(_msg)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Generate Completions & Collect Activations

    For each puzzle prompt:
    1. Generate a completion from the model
    2. Detect if the canary pattern appears in the OUTPUT
    3. Collect activations at each layer
    4. Label based on OUTPUT behavior (not input)

    This is the **correct methodology** - we probe to predict what the model will DO,
    not what the input contains.
    """)
    return


@app.cell
def _(mo):
    collect_button = mo.ui.run_button(label="Generate & Collect")
    layer_stride_input = mo.ui.dropdown(
        options=[1, 2, 4],
        value=2,
        label="Layer stride",
    )
    max_new_tokens_input = mo.ui.number(
        value=50,
        start=10,
        stop=200,
        step=10,
        label="Max new tokens",
    )
    mo.hstack([collect_button, layer_stride_input, max_new_tokens_input])
    return collect_button, layer_stride_input, max_new_tokens_input


@app.cell
def _(
    ProbeDatapoint,
    collect_button,
    layer_stride_input,
    max_new_tokens_input,
    mo,
    model,
    n_layers,
    np,
    puzzle_prompts,
    re,
    tl_utils,
    torch,
    tqdm,
):
    activations_by_layer = {}
    labels_array = None
    layer_indices = []
    probe_data = []
    _status = ""

    if collect_button.value and puzzle_prompts:
        stride = int(layer_stride_input.value)
        max_new_tokens = int(max_new_tokens_input.value)
        layer_indices = list(range(0, n_layers, stride))

        # Initialize storage
        all_activations = {_li: [] for _li in layer_indices}
        all_labels = []

        n_canary_used = 0
        n_canary_not_used = 0

        with torch.no_grad():
            for item in tqdm(puzzle_prompts, desc="Generating & collecting"):
                prompt = item["prompt"]
                detection_regex = item["detection_regex"]

                # Tokenize prompt
                tokens = model.to_tokens(prompt, prepend_bos=True)
                prompt_len = tokens.shape[1]

                # Generate completion
                output_tokens = model.generate(
                    tokens,
                    max_new_tokens=max_new_tokens,
                    temperature=0.7,
                    top_k=50,
                    stop_at_eos=True,
                    verbose=False,
                )

                # Decode completion (only the new tokens)
                completion = model.to_string(output_tokens[0, prompt_len:])

                # Detect canary in OUTPUT
                try:
                    pattern = re.compile(detection_regex, re.IGNORECASE)
                    canary_in_output = bool(pattern.search(completion))
                except re.error:
                    canary_in_output = False

                label = 1 if canary_in_output else 0
                if canary_in_output:
                    n_canary_used += 1
                else:
                    n_canary_not_used += 1

                # Store datapoint
                probe_data.append(ProbeDatapoint(
                    prompt=prompt,
                    completion=completion,
                    label=label,
                    puzzle_id=item["puzzle_id"],
                    canary_type=item["canary_type"],
                    detection_regex=detection_regex,
                ))

                # Now collect activations from the PROMPT processing
                # (before generation, at the decision point)
                _, cache = model.run_with_cache(tokens)

                for _li in layer_indices:
                    act_name = tl_utils.get_act_name("resid_post", _li)
                    resid = cache[act_name]
                    last_token_act = resid[0, -1, :].cpu().numpy()
                    all_activations[_li].append(last_token_act)

                all_labels.append(label)
                del cache

        # Stack into arrays
        activations_by_layer = {
            _li: np.stack(acts, axis=0) for _li, acts in all_activations.items()
        }
        labels_array = np.array(all_labels)

        _status = f"""✅ **Generation complete!**
    - Canary USED in output: **{n_canary_used}** samples (label=1)
    - Canary NOT used in output: **{n_canary_not_used}** samples (label=0)
    - Activations shape: {activations_by_layer[layer_indices[0]].shape}

    Now the probe will learn to predict OUTPUT behavior from activations.
    """
    elif not puzzle_prompts:
        _status = "⚠️ No prompts available. Load puzzles first."
    else:
        _status = "Click 'Generate & Collect' to start."

    mo.output.replace(mo.md(_status))
    return activations_by_layer, labels_array, layer_indices


@app.cell
def _(mo):
    mo.md(r"""
    ## Train Linear Probes

    Train a logistic regression probe at each layer to predict:
    - **label=1**: Model will USE the canary in its output
    - **label=0**: Model will NOT use the canary in its output

    High accuracy means we can predict the model's output behavior from its activations.
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
    _status = "Click 'Train Probes' to start."

    if train_button.value and activations_by_layer and labels_array is not None:
        for _li in layer_indices:
            X = activations_by_layer[_li]
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

            probe_results[_li] = {
                "probe": probe,
                "train_acc": train_acc,
                "test_acc": test_acc,
                "cv_mean": np.mean(cv_scores),
                "cv_std": np.std(cv_scores),
                "weights": weights,
                "weight_norm": np.linalg.norm(weights),
            }

        _status = "✅ Trained probes at all layers"
    elif not activations_by_layer:
        _status = "⚠️ Collect activations first."

    mo.output.replace(mo.md(_status))
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
    _output = mo.md("Train probes to see results.")
    if probe_results:
        results_data = []
        for _li in layer_indices:
            if _li in probe_results:
                _r = probe_results[_li]
                results_data.append({
                    "Layer": _li,
                    "Train Acc": f"{_r['train_acc']:.3f}",
                    "Test Acc": f"{_r['test_acc']:.3f}",
                    "CV Mean": f"{_r['cv_mean']:.3f}",
                    "CV Std": f"{_r['cv_std']:.3f}",
                    "Weight Norm": f"{_r['weight_norm']:.2f}",
                })

        results_df = pd.DataFrame(results_data)
        _output = mo.vstack([mo.md("### Probe Accuracy by Layer"), mo.ui.table(results_df)])
    mo.output.replace(_output)
    return


@app.cell
def _(layer_indices, probe_results):
    import matplotlib.pyplot as plt

    fig_acc = None
    if probe_results:
        layers = [_li for _li in layer_indices if _li in probe_results]
        test_accs = [probe_results[_li]["test_acc"] for _li in layers]
        cv_means = [probe_results[_li]["cv_mean"] for _li in layers]
        cv_stds = [probe_results[_li]["cv_std"] for _li in layers]

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
    _output = mo.md("")
    if fig_acc is not None:
        _output = mo.vstack([mo.md("### Probe Accuracy Visualization"), fig_acc])
    mo.output.replace(_output)
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
    _output = mo.md("Train probes to see analysis.")
    if probe_results:
        # Find best layer
        best_layer = max(
            [_li for _li in layer_indices if _li in probe_results],
            key=lambda _li: probe_results[_li]["test_acc"]
        )
        best_acc = probe_results[best_layer]["test_acc"]

        # Find where detection becomes reliable (>70% accuracy)
        reliable_layers = [
            _li for _li in layer_indices
            if _li in probe_results and probe_results[_li]["test_acc"] > 0.7
        ]
        first_reliable = min(reliable_layers) if reliable_layers else None

        _output = mo.md(f"""
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
    mo.output.replace(_output)
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
        layers_w = [_li for _li in layer_indices if _li in probe_results]
        weight_norms = [probe_results[_li]["weight_norm"] for _li in layers_w]

        fig_weights, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Weight norm by layer
        axes[0].bar(layers_w, weight_norms, color="steelblue", edgecolor="black")
        axes[0].set_xlabel("Layer")
        axes[0].set_ylabel("Weight Norm (L2)")
        axes[0].set_title("Probe Weight Magnitude by Layer")
        axes[0].grid(True, alpha=0.3)

        # Weight distribution for best layer
        if probe_results:
            best_layer_w = max(layers_w, key=lambda _li: probe_results[_li]["test_acc"])
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
    _output = mo.md("")
    if fig_weights is not None:
        _output = mo.vstack([mo.md("### Probe Weight Analysis"), fig_weights])
    mo.output.replace(_output)
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
        for _li in layer_indices:
            if _li in probe_results:
                _r = probe_results[_li]
                summary[str(_li)] = {
                    "train_acc": float(_r["train_acc"]),
                    "test_acc": float(_r["test_acc"]),
                    "cv_mean": float(_r["cv_mean"]),
                    "cv_std": float(_r["cv_std"]),
                    "weight_norm": float(_r["weight_norm"]),
                }

        summary_path = export_base.with_suffix(".json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        # Export weights as numpy
        weights_path = export_base.with_name(f"{export_base.name}_weights.npz")
        weights_dict = {
            f"layer_{_li}": probe_results[_li]["weights"]
            for _li in layer_indices if _li in probe_results
        }
        np.savez(weights_path, **weights_dict)

        _status = f"""
    ✅ **Exported results:**
    - Summary: `{summary_path}`
    - Weights: `{weights_path}`
    """
    elif export_button.value:
        _status = "⚠️ Train probes first before exporting."
    else:
        _status = "Click 'Export Results' to save."

    mo.output.replace(mo.md(_status))
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
