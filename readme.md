# HorusLLM: Experimentation Repository

Welcome to the central repository for all experimentation and research conducted for **HorusLLM**. This repository is dedicated to systematically exploring, benchmarking, and improving large language models (LLMs) with a focus on three critical challenges:

## Research Triad

### 1. Bias
We investigate and quantify various forms of bias present in LLMs, including social, cultural, and representational biases. Our experiments aim to identify, measure, and propose mitigation strategies for these biases to ensure fair and responsible AI.

### 2. Jailbreak
We explore the robustness of LLMs against adversarial prompts and jailbreak attempts. This includes designing and testing scenarios where models are prompted to bypass safety mechanisms, with the goal of strengthening model alignment and safety.

### 3. Hallucination
We benchmark and analyze the tendency of LLMs to generate factually incorrect or fabricated information (hallucinations). Our current focus is on evaluating and reducing hallucination rates through targeted experiments and interventions.

## Current Focus: OverNorm Benchmarking

At present, our primary experimental effort is benchmarking the hallucination ability of HorusLLM by solving the **OverNorm** task. This involves:

- Designing controlled experiments to measure hallucination frequency and severity
- Comparing model outputs against ground truth and human baselines
- Analyzing the impact of different prompt structures and model configurations

## Repository Structure

- `BiasHallusicantionTradeoff.ipynb`: Experiments and analysis on the tradeoff between bias and hallucination
- `JailbreakGameTheoreticGRPO/`: Notebooks and scripts for jailbreak and adversarial robustness experiments
- `OverNormPipeline/`: Code and data for benchmarking hallucination via the OverNorm task
- `AuditingandAlignment/`: Tools and results for model auditing and alignment
- `output.json`: Aggregated results and metrics from experiments

## Getting Started

1. Clone the repository and set up the Python environment (see `env/` for virtual environment setup).
2. Explore the notebooks and scripts in each subdirectory for detailed experiment descriptions and results.
3. Refer to the comments and documentation within each notebook for guidance on reproducing experiments.

---

*HorusLLM: Advancing safe, robust, and reliable language models through systematic experimentation.*
