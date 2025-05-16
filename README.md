# KnowOrNot: A Library for Evaluating Out-of-Knowledge Base Robustness

This repository contains the code for reproducible experiments from our paper "Know Or Not: a library for evaluating out-of-knowledge base robustness".

KnowOrNot helps you systematically evaluate LLM robustness when facing questions outside their knowledge base. Our library helps you create benchmarks, run experiments, and evaluate LLM responses through a clean, unified API.

## Installation

The easiest way to install KnowOrNot is:

```bash
# Using uv (recommended)
uv add ../KnowOrNot

# Using pip
pip install knowornot
```

## Usage

Scripts are organized into subdirectories based on their function. To run a script, use:

```bash
uv run -m subdirectory.script_name
```

Note: Do not include the `.py` extension or use slashes (`subdirectory/script_name.py`).

## Repository Structure

### `create_questions/` - Generate QA Datasets

Scripts to create diverse question-answer pairs from various data sources:

- `create_BTT_questions.py`: Generates questions from Basic Theory Test driving materials
- `create_CPF_questions.py`: Creates questions about the Central Provident Fund pension system
- `create_ICA_questions.py`: Builds questions from Immigration & Checkpoints Authority FAQ data
- `create_medishield_QA.py`: Generates health insurance questions from MediShield documents
- `get_ICA_links.py`: Extracts question links from ICA website HTML

### `experiment_run/` - Execute Experiments

Scripts to set up and run LLM experiments:

- `create_all_experiments.py`: Sets up experiment configurations across all datasets
- `run_experiments.py`: Executes experiments in sequence
- `abstention_evals.py`: Evaluates model abstention behaviors using GPT-4.1

### `run_evaluations/` - Evaluate Model Responses

Scripts for evaluating generated responses:

- `abstention_evals.py`: Evaluates whether models correctly abstain
- `factuality_evals_iter.py`: Determines factual accuracy of responses with tiered classification
- `factuality_label_final.py`: Finalizes factuality labels from human annotations
- `all_abstention_evals.py`: Batch evaluation of abstention across all experiments
- `all_factuality_evals.py`: Batch evaluation of factuality across all experiments
- `gemini_search_evals.py`: Uses Gemini's search capability to verify factuality

### `analyse_data/` - Analyze Results

Scripts to process and analyze experimental results:

- `analyse_csv.py`: Generates comprehensive analysis of evaluation results
- `make_csv.py`: Converts evaluation JSON files to CSV format
- `correct_gemini_factuality.py`: Compares factuality classifications from different evaluators

## Getting Started

1. **Setup**: Install the library and configure API keys in a `.env` file
2. **Generate Questions**: Run scripts in `create_questions/` to build QA datasets
3. **Run Experiments**: Execute experiments with `experiment_run/create_all_experiments.py` followed by `experiment_run/run_experiments.py`
4. **Evaluate Results**: Use scripts in `run_evaluations/` to assess model performance
5. **Analyze Data**: Process results with scripts in `analyse_data/`

## PolicyBench

Our experiments created PolicyBench, a challenging benchmark for evaluating OOKB robustness across four Singapore government policy domains, varying in complexity and domain specificity. The benchmark is available at [https://huggingface.co/datasets/govtech/PolicyBench](https://huggingface.co/datasets/govtech/PolicyBench).

For more details, please refer to our paper. The full source code for KnowOrNot is available at [https://github.com/govtech-responsibleai/KnowOrNot](https://github.com/govtech-responsibleai/KnowOrNot).