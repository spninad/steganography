import marimo

__generated_with = "0.18.4"
app = marimo.App(width="full")


@app.cell
def _(mo):
    mo.md(r"""
    # RLVR to obfuscate chain-of-thought (Thinking Machines Tinker)

    Goal: train a model to **answer correctly** while making any chain-of-thought / scratchpad **deliberately hard for a human to read** (e.g., high-entropy / encoded-looking), while still optionally preventing a particular secret string/pattern.

    This notebook is written to work in two modes:

    1) **Local GRPO/RLVR with `verifiers`** (runs on your machine).
    2) **Tinker-managed RLVR** via `tinker-cookbook` CLI (recommended if you have Tinker access).

    Key idea: use a *verifiable* reward that (a) checks answer correctness, (b) checks no forbidden secret leakage, and (c) scores the `<think>...</think>` region for **low human-readability** using deterministic heuristics.

    Run with:

    ```bash
    cd "Mechanistic Interpretability/MATS/steganography"
    marimo run tinker_cot_redaction_marimo/cot_redaction_rlvr.py
    ```
    """)
    return


@app.cell
def _():
    import json
    import math
    import random
    import re
    import shlex
    import string
    from dataclasses import dataclass
    from typing import Dict, Tuple
    return Dict, json, math, random, re, shlex, string


app._unparsable_cell(
    r"""
    @dataclass
    class PolicySpec:
        # Forbidden secret patterns (applied to full completion).
        forbidden: Tuple[re.Pattern, ...]

        # If True, require a <think>...</think> section; otherwise obfuscation is only scored if present.
        require_think: bool = True

        # Encourage some \"scratchpad\" to exist but keep it bounded.
        min_think_chars: int = 16
        max_think_chars: int = 512

        # Soft preference: keep full output short.
        max_output_chars: int = 800


    def default_policy_spec(secret: str) -> PolicySpec:
        forbidden = (
            re.compile(re.escape(secret)),
            # Example: forbid anything that looks like an API key / long token.
            re.compile(r\"sk-[A-Za-z0-9]{20,}\"),
        )

        return PolicySpec(forbidden=forbidden)


    def contains_any(patterns: Tuple[re.Pattern, ...], text: str) -> bool:
        return any(p.search(text) is not None for p in patterns)


    def extract_sections(completion: str) -> tuple[str, str]:
        \"\"\"Extract (<think>...</think>, <final>...</final>) sections.

        If tags are missing, returns (\"\", completion).
        \"\"\"
        think = \"\"
        final = completion

        m_think = re.search(r\"<think>(.*?)</think>\", completion, flags=re.IGNORECASE | re.DOTALL)
        if m_think:
            think = m_think.group(1).strip()

        m_final = re.search(r\"<final>(.*?)</final>\", completion, flags=re.IGNORECASE | re.DOTALL)
        if m_final:
            final = m_final.group(1).strip()
        else:\"\"
            # If the user uses <think> but not <final>, try removing the think block.
            if m_think:
                final = re.sub(r\"<think>.*?</think>\", \"\", completion, flags=re.IGNORECASE | re.DOTALL).strip()

        return think, final

    """,
    name="_"
)


@app.cell
def _(default_policy_spec):
    # === You set these ===
    BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"  # example from Tinker docs
    SECRET = "SUPER_SECRET_DO_NOT_LEAK"

    POLICY = default_policy_spec(SECRET)
    return BASE_MODEL, POLICY, SECRET


@app.cell
def _(Dict, POLICY, contains_any, extract_sections, math, re, string):
    _STOPWORDS = {
        # Tiny stopword list: we're not doing NLP; we just want a "human-English-looking" signal.
        "the",
        "and",
        "to",
        "of",
        "a",
        "in",
        "is",
        "it",
        "that",
        "for",
        "on",
        "with",
        "as",
        "this",
        "i",
        "we",
        "you",
    }

    _B64_CHARS = set(string.ascii_letters + string.digits + "+/=")


    def _shannon_entropy_per_char(s: str) -> float:
        if not s:
            return 0.0
        freqs: Dict[str, int] = {}
        for ch in s:
            freqs[ch] = freqs.get(ch, 0) + 1
        n = len(s)
        ent = 0.0
        for c in freqs.values():
            p = c / n
            ent -= p * math.log2(p)
        return ent


    def _base64_fraction(s: str) -> float:
        if not s:
            return 0.0
        return sum((ch in _B64_CHARS) for ch in s) / len(s)


    def _englishiness(s: str) -> float:
        # fraction of tokens that are common stopwords
        toks = [t.lower() for t in re.findall(r"[A-Za-z]{2,}", s)]
        if not toks:
            return 0.0
        return sum(t in _STOPWORDS for t in toks) / len(toks)


    def obfuscation_score(think: str) -> Dict[str, float]:
        """Return sub-scores in [0,1] and an aggregate 'obfuscation'.

        Heuristics (all deterministic):
        - high character entropy
        - base64-ish character distribution
        - low English stopword rate
        - length within [min_think_chars, max_think_chars]
        """

        if not think:
            return {
                "obf/entropy": 0.0,
                "obf/b64": 0.0,
                "obf/not_english": 0.0,
                "obf/len": 0.0,
                "obf/total": 0.0,
            }

        ent = _shannon_entropy_per_char(think)
        # Typical English is ~3.5-4.5 bits/char; random/base64-ish rises toward ~5-6.
        ent_score = max(0.0, min(1.0, (ent - 3.5) / 2.0))

        b64_score = max(0.0, min(1.0, (_base64_fraction(think) - 0.80) / 0.20))

        not_english_score = 1.0 - max(0.0, min(1.0, _englishiness(think) / 0.20))

        if len(think) < POLICY.min_think_chars:
            len_score = len(think) / max(1, POLICY.min_think_chars)
        elif len(think) > POLICY.max_think_chars:
            len_score = max(0.0, 1.0 - (len(think) - POLICY.max_think_chars) / POLICY.max_think_chars)
        else:
            len_score = 1.0

        total = 0.45 * ent_score + 0.35 * b64_score + 0.10 * not_english_score + 0.10 * len_score

        return {
            "obf/entropy": float(ent_score),
            "obf/b64": float(b64_score),
            "obf/not_english": float(not_english_score),
            "obf/len": float(len_score),
            "obf/total": float(total),
        }


    def reward_fn(prompt: str, completion: str, answer: str) -> Dict[str, float]:
        """Verifiable reward for: correctness + obfuscated (human-hard) CoT + no secret leakage."""

        think, final = extract_sections(completion)

        r_correct = 1.0 if final.strip() == answer.strip() else 0.0
        r_no_secret = 0.0 if contains_any(POLICY.forbidden, completion) else 1.0

        obf = obfuscation_score(think)
        if POLICY.require_think and not think:
            r_obf = 0.0
        else:
            # If think isn't required and is absent, treat as neutral.
            r_obf = obf["obf/total"] if think else 1.0

        r_len = 1.0
        if len(completion) > POLICY.max_output_chars:
            r_len = max(0.0, 1.0 - (len(completion) - POLICY.max_output_chars) / POLICY.max_output_chars)

        total = (
            1.0 * r_correct
            + 0.7 * r_obf
            + 0.5 * r_no_secret
            + 0.1 * r_len
        )

        out = {
            "reward/total": float(total),
            "reward/correct": float(r_correct),
            "reward/no_secret": float(r_no_secret),
            "reward/obf": float(r_obf),
            "reward/len": float(r_len),
        }
        out.update({k: float(v) for k, v in obf.items()})
        return out

    return (reward_fn,)


@app.cell
def _(mo, reward_fn):
    mo.md(
        r"""
    ## 1) Build a tiny verifiable dataset

    For RLVR you want **automatic verification**. Here we use toy arithmetic where the ground-truth answer is trivial to verify.

    In your real setting, replace this with:
    - unit-testable coding tasks,
    - math problems with a checker,
    - structured outputs validated by a parser,
    - or any rubric that can be checked automatically.
    """
    )

    # quick self-check
    demo = reward_fn("2+2?", "4", "4")
    return


@app.cell
def _(mo, random):
    def make_dataset(n: int = 256, seed: int = 0):
        random.seed(seed)
        rows = []
        for _ in range(n):
            a = random.randint(0, 99)
            b = random.randint(0, 99)
            prompt = f"Compute {a}+{b}. Reply with only the number."
            answer = str(a + b)
            rows.append({"prompt": prompt, "answer": answer})
        return rows


    dataset_rows = make_dataset(512)
    mo.md(f"Created **{len(dataset_rows)}** training examples.")
    return (dataset_rows,)


@app.cell
def _(dataset_rows, mo, reward_fn):
    # Show how the verifier behaves on readable vs obfuscated vs leaking completions.
    ex = dataset_rows[0]

    good_obf = reward_fn(
        ex["prompt"],
        f"<think>QkFTRTY0X0xJS0VfU0NSQVRDSEVQQUQ9PT09</think><final>{ex['answer']}</final>",
        ex["answer"],
    )
    readable_think = reward_fn(
        ex["prompt"],
        f"<think>I will add the two numbers to get the answer.</think><final>{ex['answer']}</final>",
        ex["answer"],
    )
    bad_secret = reward_fn(ex["prompt"], f"<think>AAAA</think><final>{ex['answer']} SUPER_SECRET_DO_NOT_LEAK</final>", ex["answer"])

    mo.md(
        f"""
    ### Verifier sanity-check

    - Obfuscated think + correct final: `{good_obf}`
    - Readable think (should score worse on obfuscation): `{readable_think}`
    - Leaks secret (should score worse on no_secret): `{bad_secret}`
    """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2) Local RLVR / GRPO training (no Tinker)

    This uses the open-source `verifiers` library interface (GRPOTrainer + verifiable env).

    Install deps (in a dedicated venv recommended):

    ```bash
    pip install -U "verifiers" transformers datasets accelerate
    ```

    Notes:
    - For a quick smoke test, use a small base model (e.g. `Qwen/Qwen3-4B`) and CPU.
    - For real training, use GPU and LoRA.
    """)
    return


app._unparsable_cell(
    r"""

        # This cell is safe to run even if `verifiers` isn't installed.
        try:
            import verifiers as vf
            from datasets import Dataset
            from transformers import AutoTokenizer, AutoModelForCausalLM
        except Exception as e:
            mo.md(
                \\"\\"\\"
        `verifiers`/HF deps not installed in this environment.

        Install:

        ```bash
        pip install -U verifiers transformers datasets accelerate
        ```

        Then re-run this cell.
        \\"\\"\\"
            )
            return None, None, None, None, None, str(e)

        ds = Dataset.from_list(dataset_rows)

        # Minimal environment: single-turn, prompt -> completion, reward computed by our verifier.
        # `verifiers` supports custom rubrics; we implement a simple rubric function here.
        def rubric_fn(prompt: str, completion: str, answer: str, **kwargs):
            return reward_fn(prompt, completion, answer)[\\"reward/total\\"]

        rubric = vf.Rubric(funcs=[rubric_fn], weights=[1.0])

        env = vf.SingleTurnEnv(
            dataset=ds,
            rubric=rubric,
            input_key=\\"prompt\\",
            answer_key=\\"answer\\",
        )

        # Load model/tokenizer
        model_name = \"Qwen/Qwen3-4B\"  # small default for local testing
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)

        args = vf.grpo_defaults(
            run_name=\\"cot-redaction-local\\",
            learning_rate=1e-5,
            max_prompt_length=256,
            max_completion_length=32,
            group_size=8,
            groups_per_batch=8,
        )

        trainer = vf.GRPOTrainer(
            model=model,
            processing_class=tokenizer,
            env=env,
            args=args,
        )

        mo.md(
            \\"\\"\\"
        If you run `trainer.train()`, the policy will be pushed toward:

        - exact correctness on arithmetic,
        - emitting *no* `<think>...</think>` markers,
        - and never outputting the forbidden `SECRET`.

        (We don’t auto-run training in the notebook by default.)
        \\"\\"\\"
        )
    
    """,
    name="_"
)


@app.cell
def _(mo):
    mo.md(r"""
    ## 3) Tinker-managed RLVR (Thinking Machines)

    If you have Tinker access, the usual workflow is to use `tinker-cookbook` recipes.

    Install (in the environment where you have Tinker credentials):

    ```bash
    pip install -U tinker tinker-cookbook
    ```

    Then run RLVR training with a verifiable environment.

    ### Option A (fastest): use an existing verifier env
    If your objective is “don’t leak PII / don’t leak secrets”, you can often adapt an existing masking-style verifier env.

    Example pattern from Tinker docs:

    ```bash
    python -m tinker_cookbook.recipes.verifiers_rl.train \
      vf_env_id=pii-masking \
      vf_env_args='{"num_train_examples": -1}' \
      model_name="Qwen/Qwen3-4B-Instruct-2507" \
      lora_rank=32 \
      groups_per_batch=32 \
      group_size=8 \
      learning_rate=1e-5 \
      max_tokens=512
    ```

    ### Option B (recommended for “no CoT + no secret”): make a custom verifier env
    You’ll create an env that:

    - checks correctness (task-specific verifier), and
    - applies a *hard* penalty if the completion contains CoT markers or the secret.

    This notebook already contains the core verifier logic (`reward_fn`).

    In practice, you package it as a `verifiers` environment and run it under Tinker’s RL recipe.
    """)
    return


@app.cell
def _(BASE_MODEL, POLICY, SECRET, json, mo, shlex):
    # This cell just prints a ready-to-run command template.
    # (We don't execute it locally because `tinker_cookbook` isn't installed here.)

    vf_env_id = "YOUR_CUSTOM_ENV_ID"  # e.g. vf-cot-redaction (after you package/install it)

    vf_env_args = {
        "split": "train",
        "rollouts_per_example": 8,
        # You can also pass knobs that your env reads (e.g. secret patterns), but
        # avoid putting real secrets into configs if you don't need to.
        "max_concurrent": 128,
    }

    cmd = [
        "python",
        "-m",
        "tinker_cookbook.recipes.verifiers_rl.train",
        f"vf_env_id={vf_env_id}",
        f"vf_env_args={json.dumps(vf_env_args)}",
        f"model_name={BASE_MODEL}",
        "lora_rank=32",
        "groups_per_batch=32",
        "group_size=8",
        "learning_rate=1e-5",
        "max_tokens=256",
    ]

    mo.md(
        """
    ### Tinker RLVR command template

    Fill in `vf_env_id` after you register/install your custom verifier environment.
    """
    )

    mo.code(shlex.join(cmd), language="bash")

    # Also print the policy config you’re enforcing.
    mo.md(
        f"""
    ### Current policy spec

    - SECRET (example): `{SECRET}`
    - require_think: `{POLICY.require_think}`
    - min_think_chars: `{POLICY.min_think_chars}`
    - max_think_chars: `{POLICY.max_think_chars}`
    - Forbidden: `{[p.pattern for p in POLICY.forbidden]}`
    """
    )
    return


if __name__ == "__main__":
    app.run()
