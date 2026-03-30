# Diffusion Language Models for Code Generation

This repository contains code and experiments for evaluating diffusion-based language models on Python code generation benchmarks (HumanEval, etc.).

We compare three models:

- **DiffuCoder-7B-Instruct**
- **DreamCoder-7B**
- **Stable-DiffCoder-8B-Instruct**

The main focus is on pass@1, AST-validity, truncation rate, and runtime under different diffusion steps and generation lengths.

## Repository structure

- `model_tests/`
  - `test_DiffuCoder-7B-Instruct.ipynb` – HumanEval evaluation for DiffuCoder-7B.
  - `test_Dream-coder.ipynb` – HumanEval evaluation for DreamCoder-7B.
  - `test_Stable-DiffCoder-8B.ipynb` – HumanEval evaluation for Stable-DiffCoder-8B.
  - `download_humaneval.py` – script to download and prepare the HumanEval dataset (optional).
  - `results/`
    - `humaneval_diffucoder_grid_final.csv` – CSV files with grid results for DiffuCoder.
    - `humaneval_dreamcoder_grid_final.csv` – CSV files with grid results for DreamCoder.
    - `humaneval_stable_diffcoder_grid_final.csv` – CSV files with grid results for Stable-DiffCoder.



## Installation

```bash
git clone https://github.com/Danil-Startsev-1/Diffusion-Language-Models-for-Code-Generation.git
cd Diffusion-Language-Models-for-Code-Generation

# conda (recommended)
conda env create -f environment.yml
conda activate dllm-code



## Datasets

We do **not** store datasets in this repository.

To download and prepare HumanEval, run from the project root:

```bash
cd model_tests
python download_humaneval.py
```

This will create a `data/humaneval/` folder (or another path you use in notebooks) with the tasks and tests.

## Models

Model weights are **not** stored in this repository.  
They are downloaded automatically from Hugging Face when you run the notebooks.

- DiffuCoder‑7B‑Instruct – [`ByteDance-Seed/DiffuCoder-7B-Instruct`](https://huggingface.co/apple/DiffuCoder-7B-Instruct)
- DreamCoder‑7B – [`<add exact HF repo id>`](https://huggingface.co/Dream-org/Dream-Coder-v0-Instruct-7B)
- Stable‑DiffCoder‑8B‑Instruct – [`ByteDance-Seed/Stable-DiffCoder-8B-Instruct`](https://huggingface.co/ByteDance-Seed/Stable-DiffCoder-8B-Instruct)

Example loading (used in the notebooks):

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

MODEL_ID = "ByteDance-Seed/Stable-DiffCoder-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model.eval()
```

## Running HumanEval experiments

1. Prepare the dataset (once):

```bash
cd model_tests
python download_humaneval.py
```

2. Launch Jupyter:

```bash
jupyter lab
# or
jupyter notebook
```

3. Open one of the notebooks in `model_tests/`:

- `humaneval_diffucoder.ipynb`
- `humaneval_dreamcoder.ipynb`
- `humaneval_stablediffcoder.ipynb`

4. Set `steps_list`, `max_new_list`, and the subset of HumanEval tasks (e.g. all 164 problems), then run **Run All**.  
The notebook will write aggregated metrics to CSV files under `model_tests/results/<model_name>/`.

## Reproducibility

- Notebooks are intended to be executable top‑to‑bottom in a fresh environment.
- Key hyperparameters (steps, max_new_tokens, temperature, etc.) and metrics are stored in the CSV files under `model_tests/results/`.
- Due to stochastic sampling, reported pass@1 may vary slightly between runs.

## Citation

If you use this code in academic work, please cite the original model papers and this project as:
> D. Startsev, “Diffusion Language Models for Code Generation,” 2026.
