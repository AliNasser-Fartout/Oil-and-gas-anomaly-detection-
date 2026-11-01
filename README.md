# Oil and Gas Anomaly Detection Using Feature Engineering and Machine Learning

This repository holds the code for a research project on **multi-class detection of undesirable events in offshore oil wells**. Raw multivariate sensor time series from the public **3W dataset** are turned into windowed statistical features. Classical machine learning models are then compared under three experimental conditions.

> **Integrity note.** These notebooks are the original research notebooks. They were renamed and documented with markdown cells and full-line `#` comments. No executable statement, parameter, path, seed or output was changed. The first commit has the original files byte for byte. `tools/verify_code_integrity.py` checks this automatically (see [Verifying code integrity](#verifying-code-integrity)).

---

## Table of contents

1. [Project overview](#1-project-overview)
2. [Research objectives](#2-research-objectives)
3. [Dataset](#3-dataset)
4. [Dataset preparation methodology](#4-dataset-preparation-methodology)
5. [Feature engineering](#5-feature-engineering)
6. [Experiments](#6-experiments)
7. [Machine learning models](#7-machine-learning-models)
8. [Noise injection methodology](#8-noise-injection-methodology)
9. [Hyperparameter configuration (Experiment 3)](#9-hyperparameter-configuration-experiment-3)
10. [Evaluation metrics](#10-evaluation-metrics)
11. [Project structure](#11-project-structure)
12. [Installation](#12-installation)
13. [Execution](#13-execution)
14. [Libraries and dependencies](#14-libraries-and-dependencies)
15. [Implementation notes and reproducibility](#15-implementation-notes-and-reproducibility)

---

## 1. Project overview

The pipeline has two stages:

1. **Dataset preparation.** Raw Parquet files from the 3W dataset (v2.0.0) are cleaned and labelled, cut into 30-second windows, and summarised by 128 statistical features per window.
2. **Classification experiments.** Nine classifiers are trained on a 10-class problem (class `0` = normal operation, classes `1`–`9` = event types). They are evaluated under three conditions:
   * **Experiment 1:** clean features, default model settings.
   * **Experiment 2:** Gaussian noise added to the features, default model settings.
   * **Experiment 3:** Gaussian noise added to the features, tuned hyperparameter values.

All notebooks were developed and run on Kaggle (Python 3.11.13).

## 2. Research objectives

* Build a feature-engineering pipeline that turns multivariate well-sensor time series into fixed-length statistical descriptors.
* Compare machine learning classifiers (XGBoost, SVM, k-NN, Decision Tree, Logistic Regression, Random Forest) and regularised logistic-regression models (Lasso, Ridge, Elastic-Net) on multi-class anomaly detection.
* Measure how robust these models are when the input features are corrupted by Gaussian noise.
* Measure how tuned hyperparameters affect performance under noisy conditions.

## 3. Dataset

* **Source:** the public **3W dataset**, version **2.0.0** (Kaggle input path `/kaggle/input/3w-dataset/2.0.0/`).
* **Format:** one Parquet file per well event (real or simulated, e.g. `SIMULATED_00001.parquet`), indexed by `timestamp`. Files are grouped into one folder per class, `0`–`9`.
* **Raw columns:** 27 sensor/state variables plus `class` and `state` (29 columns).
* **Variables used in this project (8):**

  | Variable | Variable | Variable | Variable |
  |---|---|---|---|
  | `P-PDG` | `P-TPT` | `T-TPT` | `P-MON-CKP` |
  | `T-JUS-CKP` | `P-JUS-CKGL` | `T-MON-CKP` | `QGL` |

* **Labels:** `0` (normal operation) and `1`–`9` (event classes). Classes `1, 2, 5, 6, 7, 8, 9` also have transient labels `101, 102, 105, …, 109` in the raw data.

The dataset is **not included** in this repository. See [`data/README.md`](data/README.md) for the expected layout and the intermediate files.

## 4. Dataset preparation methodology

Implemented in `notebooks/01a_dataset_preparation.ipynb` and `notebooks/01b_dataset_preparation_class9.ipynb`:

1. **Loading.** Each Parquet file is read with `pyarrow` and its `timestamp` index becomes a column.
2. **Variable selection.** Only `timestamp`, the 8 variables above and `class` are kept.
3. **Label preparation.** For classes `{1, 2, 5, 6, 7, 8, 9}`, the transient label `100 + k` is mapped to `k`. Only rows whose label equals the folder's class `k` are then kept. In folder `0` these are the normal-operation rows.
4. **Window-based segmentation.** Each row is assigned to a consecutive, non-overlapping **30-second window**: `instance = floor((t − t_first) / 30 s) + 1`.
5. **Missing values.** Missing sensor values are replaced by `0` before feature extraction. Missing feature values (e.g. the standard deviation of a single-sample window) are also replaced by `0`.
6. **Feature extraction.** 16 statistics per variable and per window (Section 5). Each window gets its class label.
7. **Per-class assembly.** For each anomaly class `k`, windows labelled `0` are removed. The class-`k` windows are concatenated with the class-`0` windows, the `instance` identifier is dropped, and the result is saved as `final_data0_k.csv`. The class-`0` windows alone are saved as `final_data0.csv`.
8. **Class 9 regeneration.** Notebook `01b` rebuilds `final_data0_9.csv` from the first 12 class-0 files and 14 hand-selected class-9 files. Its class-9 count (21,840 windows) matches the class-9 count used in Experiments 1 and 2.

Number of Parquet files processed per folder (`file_names[:N]`):

| Class folder | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| Files (`01a`) | 30 | 15 | 40 | 30 | 120 | 50 | 40 | 8 | 20 | 7 |
| Files (`01b`) | 12 | – | – | – | – | – | – | – | – | 14 (selected indices) |

**Steps performed at the start of each experiment notebook:**

* concatenate the ten CSV files (Experiments 1 and 2) or load `final_datav1.csv` (Experiment 3);
* remove exact duplicate rows (Experiments 1 and 2: 55,142 duplicates removed, leaving 147,711 windows);
* drop the identifier column (`instance` or `index`);
* separate features `X` (128 columns) and target `Y` (`class`);
* **train-test split:** `train_test_split(X, Y, test_size=0.2, random_state=42)`, a random, non-stratified split at window level;
* **normalisation:** `StandardScaler`, fitted on the training set and applied to the test set.

Class distribution after de-duplication (Experiments 1 and 2):

| Class | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| Windows | 13,243 | 14,097 | 14,040 | 15,998 | 9,270 | 11,943 | 12,600 | 27,599 | 7,081 | 21,840 |

## 5. Feature engineering

For each 30-second window and each of the 8 variables, `extract_features` computes **16 statistical features**, which gives **8 × 16 = 128 features** per window:

| Group | Features (column suffix) |
|---|---|
| Central tendency | mean (`_mean`), median (`_median`), mean absolute value (`_mean_abs`) |
| Dispersion / range | standard deviation (`_std`), minimum (`_min`), maximum (`_max`), range (`_range`), 25th percentile (`_q25`), 75th percentile (`_q75`), interquartile range (`_iqr`), mean absolute deviation (`_mad`) |
| Energy | root mean square (`_rms`), signal energy = Σx² (`_energy`) |
| Distribution shape | skewness (`_skew`), kurtosis (`_kurtosis`) |
| Sign changes | zero-crossing count (`_zero_crossings`) |

## 6. Experiments

| | Experiment 1 | Experiment 2 | Experiment 3 |
|---|---|---|---|
| Notebook | `02_experiment_1_no_noise_no_tuning.ipynb` | `03_experiment_2_noise_no_tuning.ipynb` | `04_experiment_3_noise_with_tuned_hyperparameters.ipynb` |
| Input data | 10 CSV files from Section 1 | 10 CSV files from Section 1 | `final_datav1.csv` (117,401 rows) |
| De-duplication | yes | yes | not performed in the notebook |
| Split / scaling | 80/20, `random_state=42` / `StandardScaler` | same | same |
| Noise | none | Gaussian (σ = 0.02) + clipping to [0, 1] | Gaussian (σ = 0.02) + clipping to [0, 1] |
| Hyperparameters | defaults | defaults | fixed tuned values |
| Models | 9 | 9 (same as Exp. 1) | 9 (same model types) |

* **Experiment 1 (no noise, no tuning):** the baseline on clean standardised features.
* **Experiment 2 (noise, no tuning):** same data, split, scaling and models as Experiment 1, with Gaussian noise injected into both the training and test features after scaling.
* **Experiment 3 (noise, tuned hyperparameters):** same noise procedure as Experiment 2. Models are built with tuned hyperparameter values written directly in the code (Section 9).

## 7. Machine learning models

| # | Model | Library | Experiments 1 and 2 configuration |
|---|---|---|---|
| 1 | XGBoost | `xgboost.XGBClassifier` | defaults |
| 2 | Support Vector Machine | `sklearn.svm.SVC` | defaults |
| 3 | k-Nearest Neighbors | `sklearn.neighbors.KNeighborsClassifier` | defaults |
| 4 | Decision Tree | `sklearn.tree.DecisionTreeClassifier` | defaults |
| 5 | Logistic Regression | `sklearn.linear_model.LogisticRegression` | defaults |
| 6 | Random Forest | `sklearn.ensemble.RandomForestClassifier` | defaults |
| 7 | Lasso (L1) logistic regression | `LogisticRegression` | `penalty='l1', solver='liblinear', C=1.0, random_state=42` |
| 8 | Ridge (L2) logistic regression | `LogisticRegression` | `penalty='l2', solver='lbfgs', C=1.0, random_state=42` |
| 9 | Elastic-Net logistic regression | `LogisticRegression` | `penalty='elasticnet', solver='saga', l1_ratio=0.5, C=1.0, random_state=42` |

Models 7–9 are called "statistical models" in the notebooks. `IsolationForest`, `AdaBoostClassifier`, `GaussianNB`, `OneClassSVM`, `LGBMClassifier` and `CatBoostClassifier` are imported but **not used**.

## 8. Noise injection methodology

Used in Experiments 2 and 3:

```python
def add_gaussian_noise(X, mean=0.0, std=0.02):
    noise = np.random.normal(loc=mean, scale=std, size=X.shape)
    return np.clip(X + noise, 0., 1.)

X_train = add_gaussian_noise(X_train, std=0.02)
X_test  = add_gaussian_noise(X_test, std=0.02)
```

* Additive zero-mean Gaussian noise with standard deviation **0.02**, drawn independently for every feature value.
* Applied **after** `StandardScaler`, to the training set and to the test set separately.
* The noisy values are then **clipped to [0, 1]**.
* No random seed is set, so each run draws a different noise realisation.

## 9. Hyperparameter configuration (Experiment 3)

Experiment 3 uses these fixed hyperparameter values:

| Model | Hyperparameters |
|---|---|
| XGBoost | `n_estimators=185, max_depth=10, learning_rate=0.07621005419696973, subsample=0.8579402483438301` |
| SVC | `C=65.76241489919478, kernel='rbf', gamma='scale', degree=4, probability=True` |
| k-NN | `n_neighbors=16, weights='distance', algorithm='brute', metric='manhattan', p=1` |
| Decision Tree | `criterion='entropy', max_depth=17, min_samples_split=8, min_samples_leaf=8, random_state=42` |
| Logistic Regression | `C=81.40911334949928, penalty='l2', fit_intercept=True, max_iter=339, tol=1.3444367682112582e-05, solver='lbfgs', multi_class='auto', random_state=42` |
| Random Forest | `n_estimators=168, max_depth=20, min_samples_split=3, min_samples_leaf=6, bootstrap=False, random_state=42` |
| Lasso (L1) LR | `penalty='l1', solver='saga', C=3.167281368521384, max_iter=935, random_state=42` |
| Ridge (L2) LR | `penalty='l2', solver='newton-cg', C=7.249509828331678, max_iter=786, random_state=42` |
| Elastic-Net LR | `penalty='elasticnet', solver='saga', l1_ratio=0.26412068295144886, C=0.32421631749317886, max_iter=797, random_state=42` |

> **The optimisation procedure is not part of this repository.** `optuna` is imported in the notebooks, but no Optuna study, objective function, search space or trial budget appears in any of the five project files. The values above are the *results* of a tuning step whose code was not included, so the search spaces and optimisation settings cannot be documented here. If the tuning code is available, adding it as a separate notebook would make Experiment 3 fully reproducible.

## 10. Evaluation metrics

Every model is evaluated on the held-out 20 % test set with:

* **Accuracy** (`accuracy_score`)
* **Recall**, macro average (`recall_score(..., average='macro')`)
* **F1-score**, macro average (`f1_score(..., average='macro')`)
* **Precision**, macro average (`precision_score(..., average='macro')`)
* **Matthews correlation coefficient** (`matthews_corrcoef`)
* **Confusion matrix** (`confusion_matrix`, plotted with `ConfusionMatrixDisplay` and saved as PNG)
* **One-vs-rest ROC curves, per-class AUC and macro AUC** (`label_binarize`, `roc_curve`, `auc`, `roc_auc_score(average='macro')`), saved as PNG

The results reported in the notebook outputs are collected in [`results/README.md`](results/README.md). Please read [Section 15](#15-implementation-notes-and-reproducibility) before interpreting precision, recall, confusion matrices and AUC.

## 11. Project structure

```
oil-and-gas-anomaly-detection/
│
├── README.md                     ← this file
├── requirements.txt              ← Python dependencies
├── .gitignore
│
├── data/
│   └── README.md                 ← dataset source, expected layout, intermediate CSV files
│
├── notebooks/
│   ├── 01a_dataset_preparation.ipynb                         ← Section 1: classes 0–9 → final_data0*.csv
│   ├── 01b_dataset_preparation_class9.ipynb                  ← Section 1: regenerates final_data0_9.csv
│   ├── 02_experiment_1_no_noise_no_tuning.ipynb              ← Section 2: Experiment 1
│   ├── 03_experiment_2_noise_no_tuning.ipynb                 ← Section 3: Experiment 2
│   └── 04_experiment_3_noise_with_tuned_hyperparameters.ipynb ← Section 4: Experiment 3
│
├── results/
│   └── README.md                 ← metrics reported in the original notebook outputs
│
└── tools/
    └── verify_code_integrity.py  ← checks documented notebooks against the originals
```

Mapping from the original filenames:

| Original file | Repository file | Section |
|---|---|---|
| `oil-gas-detection-data-preparation_3.ipynb` | `notebooks/01a_dataset_preparation.ipynb` | 1. Dataset preparation |
| `final-data9.ipynb` | `notebooks/01b_dataset_preparation_class9.ipynb` | 1. Dataset preparation |
| `model-without-noise-hyper.ipynb` | `notebooks/02_experiment_1_no_noise_no_tuning.ipynb` | 2. Experiment 1 |
| `model-with-noise-without-hyper-parameter_1.ipynb` | `notebooks/03_experiment_2_noise_no_tuning.ipynb` | 3. Experiment 2 |
| `model-with-noise-hyper.ipynb` | `notebooks/04_experiment_3_noise_with_tuned_hyperparameters.ipynb` | 4. Experiment 3 |

The notebooks are kept as Jupyter notebooks, not converted to `.py` scripts, so that the **original outputs** (tables, metrics, confusion matrices and ROC figures) stay with the code that produced them.

## 12. Installation

Python **3.11** is recommended (the notebooks were run with Python 3.11.13 on Kaggle).

```bash
git clone https://github.com/AliNasser-Fartout/pageweb.git
```

```bash
cd pageweb
```

```bash
python -m venv .venv
```

```bash
source .venv/bin/activate
```

On Windows, activate with `.venv\Scripts\activate` instead.

```bash
pip install -r requirements.txt
```

## 13. Execution

The notebooks use **absolute Kaggle paths**, for example `/kaggle/input/3w-dataset/2.0.0/0` and `/kaggle/input/3w-data-set/3W_Dataset_v2/final_data0.csv`.

* **On Kaggle (recommended, no changes needed):** attach the 3W dataset and the intermediate CSV datasets as notebook inputs at those paths, then run the notebooks.
* **Locally:** recreate the same directory layout (see [`data/README.md`](data/README.md)), or change the path strings in your own copy. Changing a path changes nothing else in the method.

Execution order:

1. `01a_dataset_preparation.ipynb` → `final_data0.csv`, `final_data0_1.csv` … `final_data0_8.csv` (and a first `final_data0_9.csv`)
2. `01b_dataset_preparation_class9.ipynb` → `final_data0_9.csv` (the version used by the experiments)
3. `02_experiment_1_no_noise_no_tuning.ipynb`
4. `03_experiment_2_noise_no_tuning.ipynb`
5. `04_experiment_3_noise_with_tuned_hyperparameters.ipynb` (requires `final_datav1.csv`; see Section 15)

To start Jupyter:

```bash
jupyter lab notebooks/
```

### Verifying code integrity

The first commit of this repository holds the original notebooks unchanged. To check that the documented notebooks differ from them only by added markdown cells and `#` comment lines:

```bash
mkdir -p /tmp/original_notebooks && git archive "$(git rev-list --max-parents=0 HEAD)" notebooks | tar -x -C /tmp/original_notebooks
```

```bash
python tools/verify_code_integrity.py /tmp/original_notebooks/notebooks notebooks
```

## 14. Libraries and dependencies

| Library | Purpose |
|---|---|
| `numpy`, `pandas` | data handling, feature computation |
| `pyarrow` | reading the Parquet files of the 3W dataset |
| `scikit-learn` | split, scaling, SVC / k-NN / Decision Tree / Logistic Regression / Random Forest, metrics, ROC |
| `xgboost` | XGBoost classifier |
| `matplotlib` | confusion matrices and ROC curves |
| `lightgbm`, `catboost`, `optuna` | imported by the notebooks (needed for the import cells to run) but not used in any computation |

Version notes:

* `pandas >= 2.2` is required for `groupby(...).apply(..., include_groups=False)`.
* `scikit-learn < 1.7` is recommended because Experiment 3 passes `multi_class='auto'` to `LogisticRegression`. That argument is deprecated since scikit-learn 1.5 and scheduled for removal.
* The exact package versions of the original Kaggle runs were not recorded. `requirements.txt` gives compatible ranges.

## 15. Implementation notes and reproducibility

The points below describe the original implementation as it is. None of them were changed, to keep the published results consistent with the code. They are listed so that readers can interpret the results correctly.

1. **Metric argument order.** All metric functions are called as `metric(predictions, y_test)`. scikit-learn expects `metric(y_true, y_pred)`.
   * Accuracy, macro-F1 and MCC are symmetric, so they are unaffected.
   * The value reported as macro **recall** is the macro **precision** under the standard definition, and the reverse.
   * The confusion matrices have *predicted* classes on the rows (y-axis) and *true* classes on the columns (x-axis). The axis labels on the saved figures say the opposite.
2. **ROC/AUC from hard predictions.** ROC curves and AUC are computed from one-hot encoded predicted *labels*, not from predicted probabilities. Each per-class curve therefore has a single operating point.
3. **Noise followed by clipping.** In Experiments 2 and 3, the standardised features (mean 0, unit variance) are clipped to [0, 1] after noise is added. All negative values become 0 and all values above 1 become 1. The "noisy" condition therefore combines additive noise with a strong bounding of the feature values.
4. **Randomness.**
   * The noise has no seed.
   * The Decision Tree and Random Forest in Experiments 1 and 2 have no `random_state`.
   * The class-0 to class-8 files and the class-9 files of `01a` are selected with `os.listdir(...)[:N]`, and the file order returned by `os.listdir` depends on the file system.
   * Re-running the notebooks can therefore give slightly different numbers.
5. **Window-level random split.** The train-test split is random over windows. Windows from the same recording can appear in both the training and the test set.
6. **Experiment 3 input data.** `final_datav1.csv` (117,401 rows after loading) is not produced by the notebooks in this repository, and the notebook does not de-duplicate it. Experiments 1 and 2 use 147,711 de-duplicated windows. The construction of this file was not part of the project files.
7. **Hyperparameter search not included.** See Section 9.
8. **Class-9 file paths.** Experiment 1 loads `final_data0_9.csv` from `/kaggle/input/data09/` and Experiment 2 loads it from `/kaggle/input/3w-data-set/3W_Dataset_v2/`. Both report the same row count and class distribution, but it was not possible to verify that the two files are identical.
9. **Minor code details.**
   * In `features_engineering`, `data['class'] = data['class'].dropna()` has no effect because of pandas index alignment. Rows with a missing label are excluded later by the class filter.
   * In `extract_features`, the variable `diffs` is computed but not used.
   * In Experiment 3, `degree=4` has no effect with the RBF kernel of the SVC, and `p=1` duplicates `metric='manhattan'` for k-NN.
10. **Output file names.**
    * The confusion-matrix PNG names are the same in all three experiment notebooks.
    * The ROC PNGs of the statistical models in Experiments 2 and 3 use the prefix `ROC_NO_NOISY_AND_NO_HYPERPARAMETER_`.
    * The machine learning ROC PNGs of Experiment 1 use `ROC_NO_NOISY_AND_HYPERPARAMETER_`.
    * Run each notebook in its own working directory to avoid overwriting figures.
