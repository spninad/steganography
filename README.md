# Canary Steganography in Chain-of-Thought

Research on whether LLMs can preserve and later use hidden "canary" information after that information is removed from visible context, and whether we can detect internal traces of that information mechanistically.

## Project Structure

```
steganography/
├── src/steganography/          # Main package
│   ├── core/                  # Core modules (canary detection, etc.)
│   ├── experiments/           # Experiment scripts (rollout generation, probing)
│   └── utils/                 # Utility modules
├── tests/                     # Unit tests
├── data/                      # Data files
│   ├── raw/                   # Raw data (canary patterns, etc.)
│   ├── processed/             # Processed data
│   └── prompts/               # Prompt templates
├── notebooks/                 # Marimo notebooks for analysis
├── scripts/                   # Standalone scripts
├── docs/                      # Documentation
├── results/                   # Experiment results
├── requirements.txt           # Python dependencies
└── pyproject.toml            # Project configuration
```

## Installation

```bash
pip install -e .
```

## Usage

### Running Notebooks

```bash
marimo run notebooks/data_generation.py
marimo run notebooks/evaluate_canary_usage.py
```

### Core Modules

```python
from steganography.core import CanaryDetector, CanaryType

detector = CanaryDetector(pattern=r"ZEPHYR-\d{2}", canary_type="codeword")
result = detector.detect("The answer is ZEPHYR-42")
```

## Documentation

- `docs/PROJECT_SPEC.md` - Full project specification
- `docs/DATA_QUALITY_AUDIT.md` - Data quality audit

## Research Questions

1. **Retention**: Can a model use canary in the final answer after the canary is removed from prompt context?
2. **Leakage Control**: Can training reduce non-final leakage without reducing final canary-use success?
3. **Steganography Pressure**: Does anti-leakage training increase hidden/internal representation of canary?
4. **Detectability**: Can linear/nonlinear probes detect canary intent before final emission?
5. **Causality**: Do interventions along probe directions change canary-use behavior?

## Key Metrics

- `A_non_final_leakage`: Canary detected in non-final channels
- `B_final_canary_use`: Canary detected in final answer when expected
- `TaskCorrectness`: Final answer correctness independent of canary marker

## Development

```bash
# Install development dependencies
pip install -r requirements.txt

# Run tests
pytest

# Format code
ruff format .
```

## License

MIT
