# Phishing Website Detection

Machine learning project for detecting phishing websites vs. legitimate websites based on URL, domain, and page-level features, modeled after [Phishing Website Detection](https://github.com/Mishrankit18/NetShield).

## Project Structure

```text
phishing website/
├── Dataset/
│   └── phishing.csv          # 11,054 samples, 30 features + target class
├── Images/
│   ├── model_comparison.png  # Accuracy bar chart across 8 classifiers
│   └── cm_*.png              # Confusion matrices for each classifier
├── models/
│   ├── best_model.joblib     # Best performing trained model (Random Forest)
│   └── metrics.json          # Evaluation metrics for all 8 models
├── src/
│   ├── features.py           # Feature metadata & URL heuristic extractor
│   └── train.py              # Training, evaluation, and plotting script
├── app.py                    # Interactive Streamlit web application
└── requirements.txt
```

## Model Comparison

Trained on an 80/20 stratified split (`8,843` train / `2,211` test samples):

| Model | Accuracy | F1 Score |
| :--- | :---: | :---: |
| **Random Forest** | **97.38%** | **97.67%** |
| XGBoost | 97.06% | 97.38% |
| Decision Tree | 96.56% | 96.92% |
| Support Vector Machine | 96.02% | 96.47% |
| Gradient Boosting | 95.39% | 95.90% |
| K-Nearest Neighbors | 94.12% | 94.74% |
| AdaBoost | 94.03% | 94.69% |
| Logistic Regression | 93.89% | 94.55% |

## Quick Start

1. **Install dependencies** (if not already installed):
   ```bash
   pip install -r requirements.txt
   ```

2. **Train & evaluate all models** (already pre-run, re-run anytime):
   ```bash
   python src/train.py
   ```

3. **Launch the Streamlit web app**:
   ```bash
   streamlit run app.py
   ```

