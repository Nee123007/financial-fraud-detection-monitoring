# Dataset Overview: Indian Financial Fraud Monitoring (770k Transactions)

This repository includes a representative lightweight sample dataset (`sample_fraud_transactions.csv`, 1,000 transactions) for quick testing, exploratory inspection, and testing pipeline execution.

The complete production-scale dataset (770,000 transactions, ~198.9 MB CSV) is hosted publicly on Kaggle.

---

## 📌 Dataset Quick Access
- **Kaggle URL**: [Indian Fraud Transactions for Monitoring](https://www.kaggle.com/datasets/neel1237/indian-fraud-transactions-for-monitoring)
- **Kaggle Handle**: `neel1237/indian-fraud-transactions-for-monitoring`
- **File Name**: `financial_fraud_monitoring_synthetic_770k.csv`

---

## 🚀 Quick Download

### Method 1: Using the provided Python script
```bash
python scripts/download_dataset.py
```

### Method 2: In Python / Colab via `kagglehub`
```python
import kagglehub

path = kagglehub.dataset_download("neel1237/indian-fraud-transactions-for-monitoring")
print("Dataset downloaded to:", path)
```

---

## 📊 Dataset Structure & Schema

The dataset captures realistic financial fraud dynamics in the Indian payments ecosystem (UPI, POS, eCommerce, IMPS/NEFT/RTGS, NetBanking, ATM) across 2025.

### Key Attributes:

| Feature | Type | Description |
| :--- | :--- | :--- |
| `transaction_id` | String | Unique synthetic transaction identifier (e.g., `SYN-0000001`) |
| `event_timestamp_ist` | Datetime (IST) | Time of transaction in Indian Standard Time (UTC+05:30) |
| `transaction_date` | Date | Date of transaction (`YYYY-MM-DD`) |
| `monitoring_window` | Categorical | Temporal window: `baseline`, `transition`, `drift_period_1`, `drift_period_2` |
| `channel` | Categorical | Payment rail: `upi`, `pos`, `ecommerce`, `mobile_banking`, `atm`, `imps_neft_rtgs` |
| `region` | Categorical | Indian State/UT (e.g., Maharashtra, Tamil Nadu, Uttar Pradesh, Gujarat, Delhi, Karnataka) |
| `customer_segment` | Categorical | `mass_market`, `salaried`, `small_business`, `affluent`, `student` |
| `merchant_category` | Categorical | `grocery`, `food_delivery`, `retail`, `jewelry`, `travel`, `bill_payment`, `mobile_recharge`, etc. |
| `merchant_risk_tier` | Categorical | Inherent category risk level: `low`, `medium`, `high` |
| `transaction_amount_inr`| Numeric | Amount of the transaction in Indian Rupees (₹) |
| `device_trust_score` | Numeric | Normalized device reputation index (0 - 100) |
| `account_age_days` | Numeric | Age of the customer bank account in days |
| `avg_transaction_amount_30d`| Numeric | 30-day historical mean transaction amount for the customer |
| `amount_to_avg_ratio` | Numeric | Ratio of current amount to 30-day mean (anomaly trigger) |
| `days_since_last_txn` | Numeric | Inactivity indicator |
| `transaction_count_24h`| Numeric | Frequency velocity in the last 24 hours |
| `failed_auth_24h` | Numeric | Count of failed OTP/PIN attempts in the last 24 hours |
| `distance_from_home_km`| Numeric | Geolocation distance from registered customer address |
| `is_new_merchant` | Binary (0/1) | Whether this is the customer's first purchase with this merchant |
| `is_high_risk_country` | Binary (0/1) | Cross-border risk flag |
| `label_available_at_ist`| Datetime (IST) | Timestamp when ground truth fraud confirmation arrived (simulates chargeback delay: 7–28 days) |
| `actual_is_fraud` | Binary (0/1) | Target label: 1 if fraudulent, 0 if legitimate |
| `model_version` | String | Baseline deployed model identifier |
| `data_drift_regime` | Categorical | Indicator of underlying feature distribution shift |
| `concept_drift_regime`| Categorical | Indicator of underlying fraud pattern evolution |

---

## ⏱️ Temporal Splits & Label Delay Simulation

Real-world financial fraud models cannot immediately observe fraud outcomes. Chargebacks, customer disputes, and bank investigations typically take 7 to 28 days to resolve.

- **Baseline Training**: Early 80% of `baseline` window transactions whose labels arrived **before** the validation period begins.
- **Baseline Validation**: Remaining 20% of `baseline` window transactions used to select optimal decision thresholds.
- **Transition & Drift Period 1**: Mid-year transactions exhibiting gradual feature and concept drift.
- **Drift Period 2 (Untouched Holdout)**: Late-year transactions (138,600 rows) with an elevated 11.64% fraud rate, serving as the true out-of-time benchmark.
