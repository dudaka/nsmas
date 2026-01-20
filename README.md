# NS-MAS: Neuro-Symbolic Multi-Agent System for Robust Mathematical Reasoning

<!-- [![Paper](https://img.shields.io/badge/Paper-EXTRAAMAS%202026-blue)](paper/main.pdf)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

This repository contains the source code, data, and results for reproducing the experiments in:

> **Neuro-Symbolic Verification for Robust Mathematical Reasoning: From Catastrophic Fragility to Trustworthy Stability**
>
> EXTRAAMAS 2026 (8th International Workshop on Explainable, Trustworthy, and Responsible AI) -->

## Key Results

| System              | Base       | NoOp       | RRR       | vs SOTA |
| ------------------- | ---------- | ---------- | --------- | ------- |
| NS-MAS Fixed Slow   | **79.50%** | **75.73%** | **0.953** | 17×     |
| GPT-4o CoT          | 58.79%     | 56.21%     | 0.956     | 17×     |
| Literature Baseline | -          | -          | ~0.35     | 1×      |

**Main Findings:**

- +20.71% accuracy improvement over GPT-4o Chain-of-Thought
- 17× better robustness retention ratio (RRR) than SOTA literature baseline
- Symbolic verification acts as a "semantic firewall" against distractor injection

## Repository Structure

```
nsmas-reproduce/
├── src/
│   ├── asp_solver/         # ASP solver with Clingo
│   ├── agent/              # LangGraph GVR agent
│   ├── bandit/             # Contextual bandit router
│   ├── experiments/        # Experiment runners
│   ├── evaluation/         # Analysis and visualization
│   └── data_engineering/   # Dataset generation
├── data/
│   ├── oracle_training_set.json  # Oracle labels for bandit
│   └── train_adf*.dat            # VW training data
├── output/
│   ├── gsm_base.jsonl      # 2,439 base problems
│   ├── gsm_p1.jsonl        # 1,550 P1 problems
│   ├── gsm_p2.jsonl        # 566 P2 problems
│   └── gsm_noop.jsonl      # 2,439 NoOp problems
├── results/
│   ├── baseline_gpt4o/     # GPT-4o CoT results
│   ├── baseline_gpt4o_mini/# GPT-4o-mini CoT results
│   ├── baseline_sc/        # Self-Consistency results
│   ├── nsmas_fixed_slow/   # NS-MAS Fixed Slow results
│   ├── nsmas_bandit/       # NS-MAS Bandit results
│   ├── nsmas_random/       # NS-MAS Random results
│   └── analysis/           # Generated plots
├── models/
│   ├── pca_50.pkl          # PCA model (384→50 dims)
│   └── warm_policy_explore.vw  # Pre-trained VW policy
├── tests/                  # Unit tests
└── environment.yml         # Conda environment
```

## Quick Start

### 1. Environment Setup

```bash
# Create conda environment
conda env create -f environment.yml
conda activate nsmas

# Or use pip
pip install -r requirements.txt
```

### 2. Set API Keys

```bash
cp .env.example .env
# Edit .env with your OpenAI API key
```

Required:

- `OPENAI_API_KEY` - For GPT-4o experiments

### 3. Verify Installation

```bash
# Run tests
python -m pytest tests/ -v

# Verify ASP solver
python -c "from src.asp_solver import ASPSolver; print('ASP OK')"
```

## Reproducing Experiments

### Option A: Use Pre-computed Results

The `results/` directory contains all experiment outputs. To regenerate analysis:

```bash
# Generate analysis report and plots
python -m src.evaluation --results-dir results --output-dir results/analysis
```

### Option B: Re-run Experiments

**Warning:** Running all experiments costs approximately $100 in API calls.

```bash
# 1. Run baselines (Phase 6b)
python -m src.experiments.baseline --model gpt-4o --data-dir output --results-dir results
python -m src.experiments.baseline --model gpt-4o-mini --data-dir output --results-dir results

# 2. Run NS-MAS experiments (Phase 6d)
python -m src.experiments.nsmas_runner --run-type fixed_slow --data-dir output --results-dir results
python -m src.experiments.nsmas_runner --run-type bandit --data-dir output --results-dir results
python -m src.experiments.nsmas_runner --run-type random --data-dir output --results-dir results

# 3. Run Self-Consistency baseline
python -m src.experiments.baseline --model gpt-4o --self-consistency --k 5 --data-dir output --results-dir results
```

### Option C: Run on Subset (Budget-Friendly)

```bash
# Run pilot on 500 problems (~$5)
python -m src.experiments.pilot --problems 500 --data-dir output --results-dir results/pilot
```

## Dataset

The GSM-Symbolic dataset is derived from Apple's [ml-gsm-symbolic](https://github.com/apple/ml-gsm-symbolic) templates.

| Variant | Problems | Description                 |
| ------- | -------- | --------------------------- |
| Base    | 2,439    | Standard problems           |
| P1      | 1,550    | Increased difficulty        |
| P2      | 566      | Highest difficulty          |
| NoOp    | 2,439    | Base + distractor sentences |

To regenerate the dataset:

```bash
# Clone GSM-Symbolic templates
git clone https://github.com/apple/ml-gsm-symbolic.git external/ml-gsm-symbolic

# Generate dataset
python -m src.data_engineering.pipeline \
    --templates-dir external/ml-gsm-symbolic/templates \
    --output-dir output \
    --instances 50 \
    --seed 42
```

## Architecture

The NS-MAS system implements a Generate-Verify-Reflect (GVR) loop:

```
                    +----------------------------------+
                    |                                  |
                    v                                  |
+----------+   +----------+   +----------+   +--------+
|  Input   |-->| Generate |-->|  Verify  |-->| Output |
| Problem  |   |  (LLM)   |   | (Clingo) |   | Answer |
+----------+   +----------+   +----------+   +--------+
                    ^               |
                    |               | (if error)
                    |               v
                    |         +----------+
                    +---------| Reflect  |
                              |  (LLM)   |
                              +----------+
```

1. **Generate**: LLM translates math problem to ASP code
2. **Verify**: Clingo solver checks logical consistency
3. **Reflect**: If verification fails, structured feedback guides correction

## Key Components

### ASP Solver

```python
from src.asp_solver import ASPSolver

solver = ASPSolver(timeout_ms=30000)
result = solver.solve('''
    quantity(john, apples, 10).
    quantity(mary, apples, 5).
    total(T) :- quantity(john, apples, A), quantity(mary, apples, B), T = @add(A, B).
    final_answer(T) :- total(T).
''')

if result.success:
    print(f"Answer: {result.answer}")  # 15
```

### LangGraph Agent

```python
from dotenv import load_dotenv
load_dotenv()

from src.agent import Agent, AgentConfig

agent = Agent(AgentConfig())
result = agent.solve("John has 10 apples. Mary gives him 5 more. How many total?")
print(f"Answer: {result['final_answer']}")  # 15
```

## Metrics

**Robustness Retention Ratio (RRR):**

$$RRR = \frac{Accuracy_{perturbed}}{Accuracy_{clean}}$$

- RRR = 1.0: Perfect robustness
- Literature SOTA: RRR ≈ 0.35 (65% drop)
- NS-MAS: RRR = 0.953 (3.77% drop)

## Citation

```bibtex
@inproceedings{nsmas2026,
  title={Neuro-Symbolic Verification for Robust Mathematical Reasoning: From Catastrophic Fragility to Trustworthy Stability},
  author={Author Names},
  booktitle={EXTRAAMAS 2026},
  year={2026}
}
```

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- GSM-Symbolic dataset from [Apple ML Research](https://github.com/apple/ml-gsm-symbolic)
- ASP solving via [Clingo/Potassco](https://potassco.org/)
- Agent orchestration via [LangGraph](https://github.com/langchain-ai/langgraph)
