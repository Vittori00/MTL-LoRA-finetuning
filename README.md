# MTL-LoRA-Finetuning

## Descrizione

Questo progetto mira a sviluppare un assistente medico attraverso il fine-tuning del modello LLaMA da 1 miliardo di parametri utilizzando le tecniche LoRA e MTL-LoRA. Il progetto è suddiviso in due fasi principali:

1. **Fine-Tuning con il Dataset PubMedQA:** Migliorare la comprensione lessicale e semantica del modello nel dominio medico utilizzando il dataset PubMedQA, che include domande mediche, contesto e risposte.

2. **Confronto delle Tecniche di Fine-Tuning:** Valutare le prestazioni del modello base, del modello ottimizzato con LoRA e del modello ottimizzato con MTL-LoRA utilizzando metriche come Perplexity e BLEU.

## Struttura del Progetto

- **LM Project.ipynb:** Notebook Jupyter contenente il codice per il fine-tuning del modello e le valutazioni delle prestazioni.
- **README.md:** File che fornisce una panoramica del progetto e delle istruzioni per l'uso.

## Prerequisiti

- Python 3.x
- Librerie Python elencate in `requirements.txt`

## Installazione

1. **Clonare il Repository:**
   ```bash
   git clone https://github.com/Vittori00/MTL-LoRA-finetuning.git
   cd MTL-LoRA-finetuning
   ```

2. **Installare le Dipendenze:**
   ```bash
   pip install -r requirements.txt
   ```

## Utilizzo

1. **Preparare il Dataset:**
   Assicurarsi che il dataset PubMedQA sia nel formato appropriato e posizionato nella directory designata.

2. **Eseguire il Fine-Tuning:**
   Aprire ed eseguire il notebook `LM Project.ipynb` per effettuare il fine-tuning del modello utilizzando le tecniche LoRA e MTL-LoRA.

3. **Valutare le Prestazioni:**
   Utilizzare il notebook per confrontare le prestazioni del modello base, del modello ottimizzato con LoRA e del modello ottimizzato con MTL-LoRA.

## Contributi

Contributi e suggerimenti sono benvenuti! Si prega di fare un fork del repository e inviare una pull request con le proprie modifiche o correzioni.
