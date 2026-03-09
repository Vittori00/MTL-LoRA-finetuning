# MTL-LoRA-Finetuning

## Overview

This project explores efficient fine-tuning of a large language model (LLM) for a **medical chatbot** application. Starting from the **Meta LLaMA 3.2 1B** model (1 billion parameters), we investigate two different fine-tuning paradigms:

1. **LoRA (Low-Rank Adaptation)** and **Layer Freezing** applied to a medical question-answering dataset.
2. **MTL-LoRA (Multi-Task Learning with LoRA)**, a novel approach based on the paper [*"MTL-LoRA: Low-Rank Adaptation for Multi-Task Learning"* (October 2024)](https://arxiv.org/abs/2410.09437), which enables a single model to learn multiple tasks simultaneously.

The ultimate goal is to build a chatbot capable of answering medical questions and, through multi-task learning, also performing auxiliary tasks (such as riddle solving) to improve patient interaction.

---

## Table of Contents

- [Objectives](#objectives)
- [Technologies Used](#technologies-used)
- [Datasets](#datasets)
- [Project Development](#project-development)
  - [Phase 1 — LoRA & Layer Freezing Fine-Tuning](#phase-1--lora--layer-freezing-fine-tuning)
  - [Phase 2 — MTL-LoRA Multi-Task Fine-Tuning](#phase-2--mtl-lora-multi-task-fine-tuning)
- [Model Architecture](#model-architecture)
  - [LoRA](#lora)
  - [MTL-LoRA](#mtl-lora)
- [Evaluation & Results](#evaluation--results)
- [Project Structure](#project-structure)
- [Installation & Usage](#installation--usage)
- [References](#references)

---

## Objectives

- Fine-tune the **LLaMA 3.2 1B** model on a biomedical domain dataset to improve its medical vocabulary and semantic understanding.
- Compare three fine-tuning strategies:
  - **Baseline** (no fine-tuning)
  - **LoRA** (parameter-efficient fine-tuning)
  - **Layer Freezing** (only the last layer is trainable)
- Implement a **Multi-Task Learning** variant of LoRA (MTL-LoRA) to simultaneously train the model on multiple tasks with a single fine-tuning run.
- Evaluate all models quantitatively using **Perplexity**, **BLEU**, and **ROUGE** metrics, and qualitatively through chatbot interactions.

---

## Technologies Used

| Technology | Role |
|---|---|
| **Python 3.x** | Primary programming language |
| **PyTorch** | Deep learning framework for model training and custom architectures |
| **Hugging Face Transformers** | Loading and running the LLaMA 3.2 1B model and tokenizer (`AutoModelForCausalLM`, `AutoTokenizer`) |
| **Hugging Face PEFT** | LoRA configuration and application (`LoraConfig`, `get_peft_model`) |
| **Hugging Face Datasets** | Loading and processing PubMedQA and RiddleSense datasets |
| **Hugging Face Evaluate** | Computing BLEU and ROUGE metrics |
| **bitsandbytes** | Quantization utilities for memory-efficient model loading |
| **accelerate** | Multi-GPU and device mapping utilities |
| **NLTK** | Natural language processing utilities |
| **sentencepiece** | Tokenization support for LLaMA |
| **scikit-learn** | Dataset splitting (`train_test_split`) |
| **NumPy / Pandas** | Data manipulation and analysis |
| **tqdm** | Training progress bars |
| **Google Colab + Drive** | Compute environment (NVIDIA GPU) and model checkpointing |

---

## Datasets

### PubMedQA (`qiaojin/PubMedQA`, `pqa_unlabeled`)

A biomedical question-answering dataset derived from PubMed abstracts. Each sample contains:
- `question`: a medical research question
- `context`: relevant paragraphs from a scientific article
- `long_answer`: the detailed answer derived from the article

**Split used in this project:**

| Split | Size |
|---|---|
| Train | 8,000 |
| Test | 2,000 |
| Validation | 500 |

Each sample is preprocessed into the format:
```
Question: <question>
Context: <context>
Answer: <long_answer>
```

### RiddleSense (`INK-USC/riddle_sense`, `rc`)

A commonsense reasoning dataset where each sample is a riddle with multiple-choice answers. Used exclusively in Phase 2 (MTL-LoRA) to train auxiliary tasks:

| Split | Size used |
|---|---|
| Train | 1,500 |

---

## Project Development

### Phase 1 — LoRA & Layer Freezing Fine-Tuning

In this phase, the LLaMA 3.2 1B model is fine-tuned on the PubMedQA dataset using two different strategies and then compared against the unmodified baseline.

#### Strategy 1: LoRA

**LoRA** (Low-Rank Adaptation) is a parameter-efficient fine-tuning technique that injects trainable low-rank matrices into the frozen layers of the pre-trained model. Rather than updating all model parameters, LoRA adds a small number of new parameters that capture task-specific adaptations.

LoRA configuration used:

| Parameter | Value |
|---|---|
| `task_type` | `CAUSAL_LM` |
| Rank (`r`) | 8 |
| Alpha (`lora_alpha`) | 32 |
| Dropout (`lora_dropout`) | 0.1 |
| Inference mode | False |

Training configuration:

| Parameter | Value |
|---|---|
| Epochs | 3 |
| Learning rate | 5e-4 |
| Batch size | 4 |
| Optimizer | AdamW (HuggingFace) |
| Precision | FP16 |
| Max sequence length | 512 |
| Evaluation strategy | Every 2,000 steps |

#### Strategy 2: Layer Freezing

All model layers are frozen except for `norm.weight` (the last normalization layer / LM head of the LLaMA model). Only gradients for this final layer are computed and updated during training. This approach uses the pre-trained model as a feature extractor.

#### Phase 1 Evaluation

All three models (base, LoRA, layer-frozen) are evaluated on the PubMedQA test set using:
- **Perplexity** — measures how well the model predicts the next token (lower is better)
- **BLEU** — measures n-gram precision between generated and reference answers
- **ROUGE** — measures recall-oriented overlap between generated and reference answers (ROUGE-1, ROUGE-2, ROUGE-L)

A chatbot was also built to qualitatively compare the three models' responses.

---

### Phase 2 — MTL-LoRA Multi-Task Fine-Tuning

This phase implements the **MTL-LoRA** approach from the paper [*"MTL-LoRA: Low-Rank Adaptation for Multi-Task Learning"* (2024)](https://arxiv.org/abs/2410.09437).

The goal is to train a single model simultaneously on **three distinct tasks**, each assigned a `task_id`:

| Task ID | Task Description | Dataset |
|---|---|---|
| 0 | **Riddle Solving** — Given a riddle and multiple choices, identify the correct answer | RiddleSense |
| 1 | **Riddle Verification** — Given a riddle, a proposed answer, and the choices, classify whether the answer is correct and provide the right one if not | RiddleSense |
| 2 | **Medical QA** — Given a question and context, generate a detailed medical answer | PubMedQA |

Training configuration for MTL-LoRA:

| Parameter | Value |
|---|---|
| Epochs | 3 |
| Learning rate | 3e-4 |
| Batch size | 4 |
| Optimizer | AdamW |
| Precision | FP16 (mixed-precision AMP) |
| LR Scheduler | CosineAnnealingLR |
| Rank | 8 |
| Number of tasks | 3 |
| Number of projections per task | 3 |
| Temperature (τ) | 0.5 |

A custom chatbot for this phase allows the user to query the MTL-LoRA model, specifying the `task_id` to direct the model toward the appropriate task.

---

## Model Architecture

### LoRA

LoRA introduces trainable rank-decomposition matrices **A** and **B** alongside the frozen pre-trained weights **W₀**. For each adapted layer, the effective weight is:

```
W = W₀ + ΔW = W₀ + B · A
```

where:
- **A** ∈ ℝ^(r×d) is initialized with random Gaussian values
- **B** ∈ ℝ^(d×r) is initialized to zero (so ΔW = 0 at the start of training)
- **r ≪ d** is the low rank

Only **A** and **B** are trained, drastically reducing the number of trainable parameters.

### MTL-LoRA

The MTL-LoRA approach extends LoRA to multi-task learning. The key components are:

#### `MTLWeights` Module

This custom `nn.Module` manages task-specific weight adaptations. For each task, it computes a weighted combination of multiple low-rank projections:

```
ΔW(task) = Σⱼ wⱼ · Bⱼ · (A · Λ_task)ᵀ
```

where:
- **A** ∈ ℝ^(input_dim × rank) is a **shared** low-rank matrix across all tasks
- **Λ_task** ∈ ℝ^(rank × rank) is a **task-specific diagonal scaling matrix** (initialized as identity)
- **Bⱼ** ∈ ℝ^(output_dim × rank) are **task-specific projection matrices** (multiple per task, indexed by j)
- **wⱼ** are **adaptive weights** computed via softmax with temperature τ:
  ```
  wⱼ = softmax(task_weights / τ)
  ```

This design allows the model to learn a shared representation (via **A**) while specializing for each task through **Λ_task**, **Bⱼ**, and **wⱼ**.

#### `CustomLlamaModel`

A custom `nn.Module` that wraps the frozen LLaMA model and applies the MTL adaptation at every layer of the transformer:

1. **Embedding** — standard token embedding from the base LLaMA model
2. **Layer-wise MTL adaptation** — for each transformer layer, the query (`q_proj`), key (`k_proj`), value (`v_proj`) projections and the feed-forward network (`mlp`) receive task-specific MTL weight updates
3. **Residual connection** — MTL-adapted outputs are added to the original hidden states
4. **LM Head** — standard causal language model head from the base LLaMA model

Only the `MTLWeights` parameters are trained; the base LLaMA model weights remain completely frozen.

---

## Evaluation & Results

### Phase 1 Results (evaluated on 50 test samples from PubMedQA)

| Model | BLEU | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---|---|---|---|
| Base (no fine-tuning) | 0.0279 | 0.1970 | 0.0701 | 0.1297 |
| LoRA fine-tuned | 0.0281 | 0.1983 | 0.0719 | 0.1292 |
| Layer Freezing fine-tuned | 0.0282 | 0.1984 | 0.0721 | 0.1294 |

> Both LoRA and Layer Freezing show modest but consistent improvements over the base model, with LoRA achieving competitive performance with far fewer trainable parameters.

---

## Project Structure

```
MTL-LoRA-finetuning/
├── LM Project.ipynb      # Main Jupyter notebook: data preprocessing, training, evaluation, chatbot
└── README.md             # Project documentation
```

The notebook is organized into the following sections:

1. **Dependencies** — Install packages and declare global variables
2. **Data Preprocessing** — Format PubMedQA samples for causal language modeling
3. **LoRA Fine-Tuning** — Load LLaMA 3.2 1B, apply LoRA, train and evaluate
4. **Layer Freezing Fine-Tuning** — Train only the final normalization layer
5. **Model Loading** — Load pre-trained checkpoints from Google Drive
6. **Chatbot Examples** — Qualitative comparison of the three Phase 1 models
7. **BLEU & ROUGE Evaluation** — Quantitative metrics for all Phase 1 models
8. **MTL-LoRA** — Multi-task dataset preparation, custom model architecture, training, evaluation, and chatbot

---

## Installation & Usage

### Prerequisites

- Python 3.x
- A CUDA-capable GPU (recommended: Google Colab with T4/A100 GPU)
- A Hugging Face account with access to [`meta-llama/Llama-3.2-1B`](https://huggingface.co/meta-llama/Llama-3.2-1B)

### Setup

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/Vittori00/MTL-LoRA-finetuning.git
   cd MTL-LoRA-finetuning
   ```

2. **Install Dependencies:**

   ```bash
   pip install transformers accelerate datasets peft
   pip install -U bitsandbytes
   pip install evaluate rouge_score nltk sentencepiece
   ```

   Or, if a `requirements.txt` is present:

   ```bash
   pip install -r requirements.txt
   ```

3. **Set your Hugging Face Token:**

   Replace all instances of `"YOUR_TOKEN"` in the notebook with your personal Hugging Face API token. This is required to download the gated LLaMA model.

### Running the Notebook

Open `LM Project.ipynb` in **Google Colab** (recommended) or a local Jupyter environment with GPU access.

Execute the cells in order:

1. Run the **Dependencies** section to install packages and import libraries.
2. (Optional) Mount **Google Drive** if you want to save/load model checkpoints.
3. Run the **Phase 1** cells to preprocess data, fine-tune with LoRA and layer freezing, and evaluate.
4. Run the **Phase 2 (MTL-LoRA)** cells to train the multi-task model and interact with the chatbot.

> **Note:** Pre-trained model checkpoints are already saved on Google Drive for faster demonstration. Load them by running the *"Loading of the models from Google Drive"* cell to skip retraining.

---

## References

- **MTL-LoRA Paper:** [*MTL-LoRA: Low-Rank Adaptation for Multi-Task Learning*](https://arxiv.org/abs/2410.09437) (October 2024)
- **LLaMA 3.2 Model:** [meta-llama/Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B) on Hugging Face
- **LoRA Paper:** [*LoRA: Low-Rank Adaptation of Large Language Models*](https://arxiv.org/abs/2106.09685) (Hu et al., 2021)
- **PubMedQA Dataset:** [qiaojin/PubMedQA](https://huggingface.co/datasets/qiaojin/PubMedQA) on Hugging Face
- **RiddleSense Dataset:** [INK-USC/riddle_sense](https://huggingface.co/datasets/INK-USC/riddle_sense) on Hugging Face
- **Hugging Face PEFT Library:** [https://github.com/huggingface/peft](https://github.com/huggingface/peft)
