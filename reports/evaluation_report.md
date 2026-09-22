# Model Evaluation Report

## Category Classifier

### Cross-Validation Results

| Model | F1 (macro) | Accuracy |
|-------|-----------|----------|
| LogisticRegression | 0.8470 ± 0.0041 | 0.8475 ± 0.0038 |
| LinearSVC | 0.8512 ± 0.0050 | 0.8514 ± 0.0034 |
| MultinomialNB | 0.7649 ± 0.0079 | 0.7845 ± 0.0047 |
| RandomForest | 0.8415 ± 0.0042 | 0.8366 ± 0.0018 |

### Test Set Results

- **Accuracy**: 0.8578
- **F1 (macro)**: 0.8605
- **F1 (weighted)**: 0.8579
- **Precision (macro)**: 0.8801
- **Recall (macro)**: 0.8436

---

## Priority Classifier

### Cross-Validation Results

| Model | F1 (macro) | Accuracy |
|-------|-----------|----------|
| LogisticRegression | 0.8156 ± 0.0031 | 0.8217 ± 0.0030 |
| LinearSVC | 0.8139 ± 0.0025 | 0.8224 ± 0.0027 |
| MultinomialNB | 0.7601 ± 0.0045 | 0.7706 ± 0.0044 |
| RandomForest | 0.8145 ± 0.0028 | 0.8221 ± 0.0027 |

### Test Set Results

- **Accuracy**: 0.8293
- **F1 (macro)**: 0.8242
- **F1 (weighted)**: 0.8297
- **Precision (macro)**: 0.8248
- **Recall (macro)**: 0.8238