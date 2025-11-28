# HANCOCK ICD Coding & Survival Ensemble - COE 476 Course Project

This repository contains code and data (with instructions) for multi-label ICD-10 code prediction, survival, and recurrence modeling on the HANCOCK head and neck cancer dataset, as part of the COE 476 course project.

Our work goes **beyond the HCAT-FusionNet benchmark**, using modern transformers, robust ensembles, and clinical neural networks for prediction from structured and multimodal embeddings. All experiments are reproducible, well-documented, and fully cross-validated.

## 📺 Project Video Demonstration

https://github.com/user-attachments/assets/3eea207d-dfc3-41c2-b931-7b03a988e399

> **Watch:**  
> - Model training and training history  
> - Model inference and live predictions  
> - Real-time results including ROC, PR curves, and metrics visualization

## Directory Structure

```
.
├── COE 476 Course Project Video.mp4   # Final presentation video
├── DataICD/                          # Main clinical/task data and synthetic/split datasets
├── Embeddings/                       # Precomputed patient embeddings (clinical, path, WSI, etc.)
├── TextData/                         # Source clinical text and code files
├── paper_figures/                    # Auto-generated plots and figures for paper/report
├── results/                          # Experiment outcome reports + intermediate metrics
├── results_ensemble/                 # Ensemble and survival/recurrence model results
├── src/                              # Helper scripts (data prep, synthetic data, modeling utils)
├── training/                         # Training scripts for baseline, BERT, and hierarchical models
├── requirements.txt                  # All required packages
├── all_experiment_reports.txt        # Aggregated experiment metrics/reports
├── README.md                         # This file
```

## Data

- `DataICD/`: Contains the core real and synthetic + focused ICD datasets (`surgery_reports_icd_multilabel.csv`, `synthetic_gpt_icd_data.csv`, etc.).
- `Embeddings/`: Contains one HDF5 file for each patient embedding modality (512-dim vectors each for clinical, pathology, semantic/text, blood/temporal, and WSI/image).
- `TextData/`: Full original raw text and code reports (for embedding or reprocessing).


## Reproducing Experiments

### 1. ICD Multi-Label Coding

- Traditional and transformer learning on real and synthetic text, baseline and hierarchical modeling.
- Go to `training/` and run scripts such as:
    - `01_explore_and_baseline.py` (TFIDF + Logistic Regression)
    - `02_bert_baseline.py` (BERT-base)
    - `03_gpt_synthetic_data.py` (GPT synthetic data generation)
    - `04_clinicalbert_with_synthetic.py` (ClinicalBERT, real+synthetic)
    - `05_hierarchical_modeling.py` (Hierarchical BERT)
- Or run all experiments with:  
  ```bash
  python training/run_all_icd_experiments.py
  ```
- All metrics, per-label F1, ROC-PR curves, and confusion matrices will be automatically saved in `results/`.

### 2. Survival/Recurrence Modeling & Ensemble

- Advanced ensemble, classical ML, and SOTA tabular neural models.
- Go to root or `training/` and run:
    ```bash
    python training/ensemble_survrec.py
    ```
- This will produce reproducible model scores, K-fold metrics, and ensemble plots in `results_ensemble/`.

### 3. Figures for Paper/Slides

- To aggregate all result summaries and auto-generate publication figures:
    ```bash
    python src/generate_results_figures.py
    ```
- See `paper_figures/` for easy drag-and-drop to your report/manuscript.

### 4. Aggregated Reports

- View the comprehensive metrics for all runs in:
  - `all_experiment_reports.txt`

## Requirements

- Install packages with:
    ```bash
    pip install -r requirements.txt
    ```
