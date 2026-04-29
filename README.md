# Gaokao Multi-Agent Question Generation Framework

Language: [English](README.md) | [中文](README.zh-CN.md)

<p align="center">
  <img src="gaokao_rc_schematic1.png" width="800" alt="Framework Architecture">
</p>

Automatic question generation, solving, and evaluation framework for Chinese Gaokao reading-comprehension tasks. The system uses a four-agent Stage1 generation pipeline and a Stage2 evaluation stack covering AI-centric, GK, and CS dimensions.

## Key Features

- Four-agent Stage1 pipeline: material selection, anchor discovery, question generation/solving, and quality verification.
- Stage2 evaluation modes: `ai`, `gk`, `cs`, `gk+cs`, `ai+gk`, `ai+cs`, and `ai+gk+cs`.
- Stage2 automatically removes evaluator models from the same model family as the Stage1 generator.
- Ablation modes for random dimensions, hard-mixed dimensions, low-frequency dimensions, and no-dimension prompts.
- Split Stage1 and Stage2 execution for network switching and reproducible evaluation.

## Quick Start

```bash
pip install -r requirements.txt
copy .env.example .env
python run.py --help
python run.py --run-mode single --unit-id 1 --dim-mode gk --prompt-level C
```

On macOS/Linux, use `cp .env.example .env` instead of `copy`.

## Configuration

1. API keys and provider entry URLs are configured in the root `.env` file. Start from `.env.example`, then set the provider keys and optional DMX base URLs you need.

2. Stage1 model routing is configured in `src/shared/api_config.py` by editing `STAGE1_PRESET` and `STAGE1_MODEL`. `STAGE1_PRESET` selects the provider route, while `STAGE1_MODEL` selects the concrete generation model.

3. Stage2 evaluation network routing is configured in `src/shared/api_config.py` by editing `STAGE2_NETWORK`. Use `overseas` for overseas routes and `domestic` for the domestic proxy route.

4. Stage2 evaluator ensemble is configured in `src/shared/api_config.py` via `STAGE2_EVAL_MODELS` and `STAGE2_MODEL_WEIGHTS`. Keep the defaults if you want to reproduce the current evaluator setup.

During Stage2, the framework detects the Stage1 generation model and excludes any evaluator from the same model family. If this leaves two pedagogical evaluators, a dimension is counted as hit only when both evaluators mark it as hit. AI-centric evaluation uses the remaining evaluators with renormalized model weights.

## CLI Reference

| Mode | Purpose | Example |
| --- | --- | --- |
| `single` | Run one unit through Stage1 and Stage2 | `python run.py --run-mode single --unit-id 1` |
| `full` | Run all units or a sampled subset | `python run.py --run-mode full --subset-size 40` |
| `baseline` | Evaluate original exam questions directly | `python run.py --run-mode baseline --eval-mode gk` |
| `extract` | Extract generated questions from an output folder | `python run.py --run-mode extract --extract-dir outputs/EXP_xxx` |
| `stage1-only` | Generate Stage1 artifacts only | `python run.py --run-mode stage1-only --subset-size 40` |
| `stage2-only` | Evaluate an existing Stage1 output folder | `python run.py --run-mode stage2-only --stage1-dir outputs/EXP_xxx` |
| `ablation-nodim` | Run the no-dimension prompt ablation | `python run.py --run-mode ablation-nodim --subset-size 40 --eval-mode ai+gk+cs` |

Common parameters:

| Parameter | Description | Common Values |
| --- | --- | --- |
| `--dim-mode` | Stage1 pedagogical dimension family | `gk`, `cs` |
| `--prompt-level` | Prompt detail level | `A`, `B`, `C` |
| `--eval-mode` | Stage2 evaluator set | `ai`, `gk`, `cs`, `gk+cs`, `ai+gk+cs` |
| `--subset-size` | Sample size for subset runs | `40`, `60` |
| `--subset-strategy` | Sampling strategy | `proportional_stratified`, `stratified`, `random` |
| `--exam-type` | Baseline exam filter | `all`, `national`, `local` |

## Reproduction Commands

```bash
# Single-unit run
python run.py --run-mode single --unit-id 1 --dim-mode gk --prompt-level C

# Stratified 40-unit run
python run.py --run-mode full --subset-size 40 --dim-mode gk --prompt-level C

# Stage1 then Stage2, useful when changing network environments
python run.py --run-mode stage1-only --subset-size 40
python run.py --run-mode stage2-only --stage1-dir outputs/EXP_xxx --eval-mode ai+gk+cs

# Original exam-question baseline
python run.py --run-mode baseline --eval-mode gk --exam-type national

# No-dimension ablation
python run.py --run-mode ablation-nodim --subset-size 40 --eval-mode ai+gk+cs

# Extract generated questions
python run.py --run-mode extract --extract-dir outputs/EXP_xxx --extract-format markdown
```

## No-API Smoke Checks

These checks validate local wiring without making real model calls:

```bash
python run.py --help
python src/shared/api_config.py
python tools/check_stage_independence.py
python tools/check_static_alignment.py
python scripts/extract_questions.py --help
python -m compileall -q run.py src scripts tools output_analysis
```

## Project Structure

```text
├── run.py              # CLI entry point
├── src/
│   ├── shared/         # Shared config, data loading, LLM wrappers, reports
│   ├── generation/     # Stage1 generation agents and pipeline
│   ├── evaluation/     # Stage2 AI/GK/CS evaluation
│   └── showcase/       # Case showcase helpers
├── data/               # Core experiment data and dimension mappings
├── scripts/            # Utility scripts
├── tools/              # Development and audit tools
└── output_analysis/    # Output analysis package
```

## License

MIT License. See `LICENSE` for details.

The code is MIT licensed. The bundled Gaokao-related data is intended for academic research use.
