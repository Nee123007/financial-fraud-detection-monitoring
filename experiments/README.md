# Experimental Models & MLOps Drift Monitoring Suite

This folder houses the experimental architectures and advanced MLOps drift monitoring workflows developed alongside the primary XGBoost model.

---

## 📂 Contents

| File | Primary Architecture | Focus Areas | Key Output Highlights |
| :--- | :--- | :--- | :--- |
| **`other_1.ipynb`** | **HistGradientBoostingClassifier** (LightGBM-style) | • Delayed-label maturation simulation<br>• Population Stability Index (PSI) drift tracking<br>• F2-score threshold optimization<br>• Automated alert rules & slice analysis<br>• Model Retraining (Baseline v1.0 vs Candidate v2.0) | • Best probability calibration (**ECE: 0.040**, **Brier: 0.092**)<br>• PR-AUC: **0.339** on holdout<br>• Successfully detects high-drift slices |
| **`other_2.ipynb`** | **CatBoostClassifier** | • Native categorical feature handling (no encoding distortion)<br>• Class weighting (`auto_class_weights='Balanced'`)<br>• Dual action decisioning: **Approve**, **Manual Review**, **Decline**<br>• Model Retraining (v1.0 vs v2.0) | • Highest holdout recall (**49.01%** on Candidate v2.0)<br>• Specificity: **87.01%**<br>• Eliminates one-hot dimension explosion |
| **`other2.ipynb`** | *Direct mirror of `other_2.ipynb`* | Convenience alias matching alternate naming schemes | Identical to `other_2.ipynb` |

---

## 🔬 Experiment 1: HistGradientBoosting & PSI Drift Monitoring (`other_1.ipynb`)

### 1. Motivation
Traditional machine learning benchmarks assume static, independently and identically distributed (i.i.d.) test data. In real-world financial fraud:
1. **Labels are delayed**: Fraud outcomes only mature after chargeback investigations (weeks after the transaction occurs).
2. **Distributions drift**: Attackers adapt to existing rules, causing feature and score drift over time.

### 2. Methodological Highlights
- **Histogram-based Gradient Boosting**: Fast binning suited for continuous distributions.
- **Population Stability Index (PSI)**: Quantifies shift between training baseline and production periods:
  - $\text{PSI} < 0.10$: Negligible shift.
  - $0.10 \le \text{PSI} < 0.25$: Moderate drift (investigate).
  - $\text{PSI} \ge 0.25$: Severe drift (trigger alert / retrain).
- **Weekly Matured Label Evaluation**: Tracks precision and recall week-by-week only on settled transactions.
- **Retraining Evaluation**: Trains Candidate v2.0 using newly matured labels through Drift Period 1, showing improved holdout PR-AUC (0.333 → 0.339) and recall (43.92% → 46.75%).

---

## 🔬 Experiment 2: CatBoost & Dual Decisioning (`other_2.ipynb`)

### 1. Motivation
One-hot encoding high-cardinality geographic and merchant categorical variables can cause sparse, deep decision trees. CatBoost computes ordered target statistics natively on categorical inputs during training.

### 2. Methodological Highlights
- **Native Categorical Inputs**: Direct string handling of `channel`, `region`, `customer_segment`, `merchant_category`, `merchant_risk_tier`, `card_present`, and `international`.
- **Three-Tier Operational Policy**:
  - **Decline**: Transactions exceeding the fraud threshold ($\tau_{\text{fraud}} = 0.1059$).
  - **Manual Review**: Transactions in the suspicion band ($\tau_{\text{review}} \le \text{score} < \tau_{\text{fraud}}$, where $\tau_{\text{review}} = 0.60 \times \tau_{\text{fraud}}$).
  - **Approve**: Transactions below $\tau_{\text{review}}$.
- **Class-Weighted Optimization**: Balances minority fraud signal without manual downsampling.
- **Retrained Candidate Performance**: Candidate v2.0 catches **7,905 fraud cases** (49.01% recall) on the challenging out-of-time holdout.

---

## 📈 Cross-Model Holdout Comparison (Drift Period 2: 138,600 Transactions, 11.64% Fraud Rate)

| Metric | Main Model (`main_code.ipynb`) XGBoost (Tuned) | Experiment 1 (`other_1.ipynb`) HistGBM Candidate v2.0 | Experiment 2 (`other_2.ipynb`) CatBoost Candidate v2.0 |
| :--- | :---: | :---: | :---: |
| **Accuracy** | **86.21%** | 82.59% | 82.58% |
| **Precision** | **40.34%** | 35.78% | 33.19% |
| **Recall** | 38.54% | 46.75% | **49.01%** |
| **F1-Score** | 39.42% | 40.54% | 39.58% |
| **PR-AUC** | 0.322 | **0.339** | 0.336 |
| **Brier Score** | — | **0.0922** | 0.1876 |
| **ECE (Calibration)** | — | **0.040** | 0.301 |

### Key Takeaways for Academic Presentation:
1. **Precision vs Recall Trade-off**: The XGBoost main pipeline focuses on higher precision (40.34%), reducing false customer friction. CatBoost prioritizes fraud capture (49.01% recall).
2. **Calibration Matters**: HistGradientBoosting provides the most reliable raw probability estimates (lowest ECE: 0.040), making it ideal for score-based tiered risk routing.
3. **Continuous Retraining**: In both experiments, retraining on newly matured labels significantly boosted recall on the drift holdout set without degrading PR-AUC.
