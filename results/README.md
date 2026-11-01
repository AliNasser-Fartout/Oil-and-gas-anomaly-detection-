# Results

The values below are **copied from the saved outputs of the original notebooks**. They were not re-computed. The full outputs, including every confusion matrix and ROC figure, are stored in the notebooks in `notebooks/`.

Values are rounded to 4 decimals (macro AUC to the 3 decimals printed by the notebooks). The test set is 20 % of the data (`random_state=42`).

> \* **How to read Recall and Precision:** the notebooks call every metric as `metric(predictions, y_test)`. Accuracy, macro-F1 and MCC are unaffected. The column labelled *Recall* is the macro precision under the standard `metric(y_true, y_pred)` convention, and the column labelled *Precision* is the macro recall. Macro AUC is computed from hard class predictions. See README.md, Section 15.

### Experiment 1: no noise, no hyperparameter tuning

Source: `notebooks/02_experiment_1_no_noise_no_tuning.ipynb`

| Model | Accuracy | Recall (macro)* | F1 (macro) | Precision (macro)* | MCC | Macro AUC |
|---|---|---|---|---|---|---|
| XGBoost | 0.9999 | 0.9999 | 0.9999 | 0.9999 | 0.9999 | 1.000 |
| SVC | 0.9874 | 0.9838 | 0.9830 | 0.9828 | 0.9858 | 0.991 |
| k-NN | 0.9905 | 0.9889 | 0.9873 | 0.9859 | 0.9893 | 0.992 |
| Decision Tree | 0.9996 | 0.9995 | 0.9995 | 0.9996 | 0.9996 | 1.000 |
| Logistic Regression | 0.9926 | 0.9903 | 0.9906 | 0.9910 | 0.9916 | 0.995 |
| Random Forest | 1.0000 | 0.9999 | 1.0000 | 1.0000 | 1.0000 | 1.000 |
| Lasso (L1) logistic regression | 0.9860 | 0.9837 | 0.9824 | 0.9814 | 0.9842 | 0.990 |
| Ridge (L2) logistic regression | 0.9926 | 0.9903 | 0.9906 | 0.9910 | 0.9916 | 0.995 |
| Elastic-Net logistic regression | 0.8984 | 0.8901 | 0.8648 | 0.8550 | 0.8855 | 0.922 |

### Experiment 2: Gaussian noise (σ = 0.02), no hyperparameter tuning

Source: `notebooks/03_experiment_2_noise_no_tuning.ipynb`

| Model | Accuracy | Recall (macro)* | F1 (macro) | Precision (macro)* | MCC | Macro AUC |
|---|---|---|---|---|---|---|
| XGBoost | 0.9881 | 0.9868 | 0.9850 | 0.9838 | 0.9865 | 0.991 |
| SVC | 0.9610 | 0.9632 | 0.9490 | 0.9416 | 0.9563 | 0.969 |
| k-NN | 0.9673 | 0.9609 | 0.9561 | 0.9529 | 0.9631 | 0.975 |
| Decision Tree | 0.9787 | 0.9739 | 0.9740 | 0.9740 | 0.9759 | 0.986 |
| Logistic Regression | 0.9515 | 0.9482 | 0.9376 | 0.9314 | 0.9454 | 0.963 |
| Random Forest | 0.9870 | 0.9861 | 0.9840 | 0.9825 | 0.9854 | 0.991 |
| Lasso (L1) logistic regression | 0.9546 | 0.9527 | 0.9429 | 0.9377 | 0.9489 | 0.966 |
| Ridge (L2) logistic regression | 0.9517 | 0.9486 | 0.9381 | 0.9319 | 0.9456 | 0.963 |
| Elastic-Net logistic regression | 0.9566 | 0.9547 | 0.9447 | 0.9393 | 0.9512 | 0.967 |

### Experiment 3: Gaussian noise (σ = 0.02), tuned hyperparameters

Source: `notebooks/04_experiment_3_noise_with_tuned_hyperparameters.ipynb` (input: `final_datav1.csv`)

| Model | Accuracy | Recall (macro)* | F1 (macro) | Precision (macro)* | MCC | Macro AUC |
|---|---|---|---|---|---|---|
| XGBoost | 0.9893 | 0.9888 | 0.9778 | 0.9700 | 0.9876 | 0.984 |
| SVC | 0.9839 | 0.9853 | 0.9663 | 0.9545 | 0.9813 | 0.976 |
| k-NN | 0.9820 | 0.9786 | 0.9644 | 0.9546 | 0.9791 | 0.976 |
| Decision Tree | 0.9850 | 0.9802 | 0.9706 | 0.9641 | 0.9826 | 0.981 |
| Logistic Regression | 0.9710 | 0.9687 | 0.9468 | 0.9319 | 0.9664 | 0.964 |
| Random Forest | 0.9832 | 0.9885 | 0.9688 | 0.9552 | 0.9805 | 0.977 |
| Lasso (L1) logistic regression | 0.9705 | 0.9667 | 0.9458 | 0.9310 | 0.9657 | 0.964 |
| Ridge (L2) logistic regression | 0.9687 | 0.9656 | 0.9426 | 0.9267 | 0.9637 | 0.962 |
| Elastic-Net logistic regression | 0.9568 | 0.9513 | 0.9206 | 0.9012 | 0.9498 | 0.948 |

> Experiment 3 uses a different input file (`final_datav1.csv`, 117,401 rows, no de-duplication step) from Experiments 1 and 2 (147,711 de-duplicated windows). Keep this in mind when comparing Experiment 3 with the other two.

## Generated figures

When the notebooks are run, they write PNG files to the working directory: confusion matrices (`confusion_matrix_*.png`, `confusion_lasso.png`, `confusion_Ridge.png`, `confusion_elastic.png`) and ROC curves (`ROC_*.png`). These files are ignored by Git by default. To publish selected figures, place them in this folder and remove the corresponding rule from `.gitignore`.
