
# Interpretable Video-based Valence/Arousal Regression (LightGBM Baseline)

## Project Overview

This repository contains the full research pipeline developed for
the *Forschungspraxis* project **"Predicting the Emotional Impact of Videoclips"**.

The goal of the project is to establish a **reproducible, interpretable baseline** for
continuous **valence** and **arousal** prediction on the **LIRIS-ACCEDE** dataset
using **handcrafted audio-visual features** and **LightGBM (GBDT)** regression models.

In contrast to recent deep-learning-based approaches, this work explicitly prioritizes:

- strict **movie-disjoint generalization**
- **reproducibility** across all pipeline stages
- **interpretability** via classical LightGBM feature importances (Gain, Split)
- a clean and modular experimental design

The repository is structured as a complete research artifact: given identical inputs and
configuration files, the pipeline produces deterministic and reproducible results.

---

## Reproducibility as a Design Principle

Reproducibility is a central design goal of this project.

- All experiments are controlled via a **central configuration file** (`config.ini`)
- Random seeds are fixed or explicitly sampled and stored
- The pipeline is executed in a **well-defined order**
- All intermediate artifacts (feature sets, schedules, logs, results) are written to disk
- Re-running the same scripts with the same configuration yields identical results

The README focuses on **how to reproduce the experiments**, while the accompanying report focuses on **scientific motivation, methodology, and analysis**.

---

## Repository Structure

The repository follows a modular structure that mirrors the experimental pipeline:

```
├── data
│   ├── normalization_stats.csv        # Global feature normalization statistics
│   ├── *_annotation.json              # Annotations per sample
│   ├── *_frame.json                   # Frame-level feature metadata
│   ├── *_meta.json                    # Clip metadata
│   ├── *_single.json                  # Single-sample descriptors
│   ├── index_train.json               # Movie-disjoint train split
│   ├── index_validation.json          # Movie-disjoint validation split
│   └── index_test.json                # Movie-disjoint test split
│
├── doc
│   ├── config.ini                     # Central experiment configuration
│   ├── feature_groups.json            # Feature group definitions and normalization
│   ├── ACCEDEranking.txt              # Feature ranking logs
│   ├── ACCEDEsets.txt                 # Feature set summaries
│   └── failed_models.txt              # Failed training runs
│
├── logs                                # Training logs (LightGBM, TensorBoard)
├── models                              # Stored model checkpoints
├── plots
│   ├── importances                    # Feature importance plots
│   └── predictions                    # Prediction scatter plots
│
├── reports                             # Generated evaluation reports
├── schedules                           # Training schedules (JSON)
├── sets                                # Feature-set definitions (JSON)
│
├── scripts
│   ├── pipeline
│   │   ├── selection                  # Feature selection pipeline
│   │   └── training                   # Training pipeline
│   └── tools
│       ├── preprocessing              # Indexing, normalization utilities
│       └── postprocessing             # Evaluation, plotting utilities
│
├── src
│   └── video_va_regression             # Core Python package
│
├── tests                               # Test and evaluation outputs
├── setup.py                            # Package installation script
└── README.md
```

---

## Environment Setup

### Python Version

- **\>= Python 3.9** (required, enforced via `setup.py`)

### Virtual Environment (recommended)

```bash
python -m venv venv
source venv/bin/activate   # Linux / macOS
# venv\Scripts\activate  # Windows
```

### Package Installation

Install the project in editable mode using the provided `setup.py`:

```bash
pip install -e .
```

All dependencies are defined and resolved through the setup script.

---

### Pre-processing Tools

This section documents all auxiliary tools used before and after the core
training pipeline. While each script provides detailed in-file docstrings and
`--help` messages, the most important invocation patterns, parameters, and
default values are summarized here for convenience.

All tools are located under:

```
scripts/tools/
```

---

#### 1. `create_labels.py` — Generate Annotation JSON Files

**Purpose:**  
Convert ACCEDE ranking and split definition tables into per-sample annotation
JSON files used throughout the pipeline.

**Script:**  
`scripts/tools/preprocessing/create_labels.py`

**Entry Point:**  
`video-va-create-labels`

**CLI Arguments:**

| Argument | Default | Description | Datatype / Values |
|--------|---------|-------------|---------------|
| `--ranking-path` | `./doc/ACCEDEranking.txt` | ACCEDE ranking table | String (Path) |
| `--sets-path` | `./doc/ACCEDEsets.txt` | ACCEDE split definitions | String (Path) |
| `--output-dir` | `./data` | Output directory for `_annotation.json` files | String (Path) |

---

#### 2. `create_index.py` — Build Movie-disjoint Splits

**Purpose:**  
Generate train/validation/test index files based on annotation metadata,
ensuring strict movie-disjoint splits.

**Script:**  
`scripts/tools/preprocessing/create_index.py`

**Entry Point:**  
`video-va-create-index`

**CLI Arguments:**

| Argument | Default | Description | Datatype / Values |
|--------|---------|-------------|---------------|
| `--data-dir` | `./data` | Directory containing annotation JSON files | String (Path) |
| `--output-dir` | `./data` | Output directory for index files | String (Path) |

---

#### 3. `create_normalization_stats.py` — Compute Feature Normalization

**Purpose:**  
Compute group-wise normalization statistics (min/max/mean/std) from frame-level
features according to predefined feature groups.

**Script:**  
`scripts/tools/preprocessing/create_normalization_stats.py`

**Entry Point:**  
`video-va-create-normalization-stats`

**CLI Arguments:**

| Argument | Default | Description | Datatype / Values |
|--------|---------|-------------|---------------|
| `--data-dir` | `./data` | Directory containing `_frame.npy` files | String (Path) |
| `--feature-groups` | `./doc/feature_groups.json` | Feature group definitions | String (Path) |
| `--output-csv` | `./data/normalization_stats.csv` | Output CSV file | String (Path) |

---

### Post-processing Tools

#### 4. `sample_models.py` — Select Best Models

**Purpose:**  
Select and copy top-performing models based on evaluation metrics using
top-k, percentage, or Pareto-optimal selection.

**Script:**  
`scripts/tools/postprocessing/sample_models.py`

**Entry Point:**
`video-va-sample-models`

**CLI Arguments (selection):**

| Argument | Default | Description | Datatype / Values |
|--------|---------|-------------|---------------|
| `--input-dir` | – | Root directory with trained models | String (Path) |
| `--output-dir` | – | Destination directory | String (Path) |
| `--mode` | `topk` | Selection mode | "topk", "percent" |
| `--k` | `10` | Models per bucket | Integer |
| `--percent` | `10.0` | Percentage per bucket | Float |
| `--metric` | `pareto` | Ranking metric | "mse", "pearson", "pareto" |
| `--size-threshold` | `100` | Feature threshold | Integer |
| `--no-r2-filter` | – | Disable R² filter | flag |
| `--disable-size-buckets` | – | Disable size buckets | flag |
| `--dry-run` | – | Do not copy files | flag |

---

#### 5. `plot_importances.py` — Aggregate Feature Importances

**Purpose:**  
Generate aggregated mean feature-importance bar plots (gain/split) across
multiple trained models.

**Script:**  
`scripts/tools/postprocessing/plot_importances.py`

**Entry Point:**  
`video-va-plot-importances`

**CLI Arguments:**

| Argument | Default | Description | Datatype / Values |
|--------|---------|-------------|---------------|
| `--input-dir` | – | Root directory containing model outputs split into `valence/` and `arousal/`. | String (Path) |
| `--output-dir` | `./plots/importances` | Output directory for generated bar plots and CSV summaries. | String (Path) |
| `--topN` | `20` | Number of top-ranked features shown in each importance plot. | Integer |
| `--xmax` | `0.05` | Maximum x-axis value for all plots (shared scale). | Float |
| `--axis-label-size` | `20` | Font size for axis labels. | Integer |
| `--tick-label-size` | `18` | Font size for tick labels and feature names. | Integer |

---

#### 6. `plot_predictions.py` — Scatter Plots

**Purpose:**  
Generate scatter plots comparing ground truth vs. predicted values.

**Script:**  
`scripts/tools/postprocessing/plot_predictions.py`

**Entry Point:**  
`video-va-plot-predictions`

**CLI Arguments:**

| Argument | Default | Description | Datatype / Values |
|--------|---------|-------------|---------------|
| `--input-dir` | – | Root directory containing prediction CSV files. | String (Path) |
| `--output-dir` | `./plots/predictions` | Output directory for generated scatter plots. | String (Path) |
| `--axis-mode` | `1to5` | Axis range configuration for ground truth and predictions. | "1to5", "minus1to1" |
| `--point-size` | `12.0` | Marker size of scatter points. | Float |
| `--axis-label-size` | `22` | Font size for axis labels. | Integer |
| `--tick-label-size` | `20` | Font size for axis tick labels. | Integer |

---

#### 7. `create_report.py` — Aggregate Metrics Report

**Purpose:**  
Aggregate evaluation metrics from multiple model runs into a single CSV file.

**Script:**  
`scripts/tools/postprocessing/create_report.py`

**Entry Point:**  
`video-va-create-report`

**CLI Arguments:**

| Argument | Default | Description | Datatype / Values |
|--------|---------|-------------|---------------|
| `--input-dirs` | – | Root directories containing trained model subdirectories with metrics files. | String[ ] (Path) |
| `--output-path` | – | Target path for the aggregated CSV report. | String (Path) |

---

## Notes

- All default paths align with `config.ini`
- Each tool supports `--help` for full CLI documentation
- Tools are intentionally decoupled from training to preserve modularity

---

## Central Configuration

All experiments are controlled via the file:

```
doc/config.ini
```

This configuration file defines:

- data paths
- feature selection parameters
- scheduling logic
- LightGBM hyperparameters
- aggregation strategies
- training and evaluation behavior

The configuration is read-only at runtime and accessed consistently by all scripts.
No experiment-relevant parameters are hard-coded elsewhere in the pipeline.

---

## Pipeline Overview (End-to-End)

The full experimental pipeline is executed in the following conceptual order.
Each step corresponds to one or more scripts in `scripts/`.

---

### 1. Feature Selection

Run the structured feature selection pipeline:

- variance filtering
- mutual information ranking
- cross-validated importance estimation
- permutation validation
- redundancy pruning

Produces multiple feature-set JSON files.

**Script:**
`scripts/pipeline/selection/create_selection.py`

**Entry Point:**  
`video-va-create-selection`

**Output:**
```
sets/*.json
```

---

### 2. Training Schedule Generation

Generate a reproducible training schedule defining:

- feature-set combinations
- aggregation strategies
- hyperparameter configurations

**Script:**
`scripts/pipeline/selection/create_schedule.py`

**Entry Point:**  
`video-va-create-schedule`

**Output:**
```
schedules/schedule.json
```

---

### 3. Model Training

Train LightGBM regressors according to the generated schedule:

- separate models for valence and arousal
- deterministic training with fixed seeds
- optional checkpointing

**Script:**
`scripts/pipeline/training/run_training.py`

**Entry Point:**  
`video-va-run-training`

**Outputs:**
- trained models
- logs
- intermediate metrics
- dataset parameter manifests

---

### 4. Evaluation and Testing

Evaluate trained models on validation and test splits:

- MSE, RMSE, Pearson r, R²
- optional inversion to original label scale

**Script:**
`scripts/pipeline/training/run_test.py`

**Entry Point:**  
`video-va-run-test`

**Output:**
```
tests/
```

---

## Feature Groups

Feature definitions and normalization strategies are specified in:

```
doc/feature_groups.json
```

Each group defines:

- semantic category (e.g., color, motion, spectral)
- audio vs. video modality
- normalization scheme
- composite vs. atomic descriptors

This file is used consistently across normalization, selection, and training.

---

## Output Artifacts

Typical outputs produced by the pipeline include:

- trained LightGBM models (`models/`)
- training logs (`logs/`)
- evaluation tables (`reports/`)
- scatter plots and feature importance visualizations (`plots/`)

All outputs are uniquely traceable to a specific configuration and schedule.

---

## Contact

Author: Luka Petrovic (luka.petrovic@tum.de)
Supervisor: Philipp Paukner (p.paukner@tum.de)
Chair for Data Processing, Technical University of Munich
