# MTL-LoRA-Finetuning

## Description

This project aims to develop a medical assistant by fine-tuning the 1-billion-parameter LLaMA model using **LoRA** and **MTL-LoRA** techniques. The project is divided into two main phases:

1. **Fine-Tuning with the PubMedQA Dataset:** Enhance the model's lexical and semantic understanding in the medical domain using the PubMedQA dataset, which includes medical questions, context, and answers.

2. **Comparison of Fine-Tuning Techniques:** Evaluate the performance of the base model, the LoRA-optimized model, and the MTL-LoRA-optimized model using metrics such as **Perplexity** and **BLEU**.

## Project Structure

* **LM Project.ipynb:** Jupyter notebook containing the code for model fine-tuning and performance evaluation.
* **README.md:** File providing an overview of the project and usage instructions.

## Prerequisites

* Python 3.x
* Python libraries listed in `requirements.txt`

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/Vittori00/MTL-LoRA-finetuning.git
   cd MTL-LoRA-finetuning
   ```

2. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Prepare the Dataset:**
   Make sure the PubMedQA dataset is in the correct format and placed in the designated directory.

2. **Run Fine-Tuning:**
   Open and run the `LM Project.ipynb` notebook to fine-tune the model using the LoRA and MTL-LoRA techniques.

3. **Evaluate Performance:**
   Use the notebook to compare the performance of the base model, the LoRA-optimized model, and the MTL-LoRA-optimized model.

## Contributions

Contributions and suggestions are welcome! Please fork the repository and submit a pull request with your improvements or fixes.
