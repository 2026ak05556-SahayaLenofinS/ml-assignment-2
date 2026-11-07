# SpamScope – E-mail Spam Classification (ML Assignment 2)

## a. Problem statement
<!-- EDIT in your own words -->
E-mail spam wastes time and carries phishing risk. The task is to decide, from numeric
features extracted from an e-mail's text, whether the message is **spam (1)** or
**not spam / ham (0)**. This is a binary classification problem. Five classical machine-learning
classifiers (Logistic Regression, Decision Tree, KNN, Gaussian Naive Bayes, SVM) are trained on
the same data, compared with six metrics, and exposed through a Streamlit app.

## b. Dataset description
- **Source:** UCI Machine Learning Repository – Spambase (https://archive.ics.uci.edu/dataset/94/spambase)
- **Instances:** 4,601 e-mails  |  **Features:** 57 numeric  |  **Target:** `spam` (1 = spam, 0 = ham)
- **Class balance:** ~39.4% spam / ~60.6% ham  <!-- confirm from your run -->
- **Feature groups:** 48 `word_freq_*` (percentage of words matching a given word),
  6 `char_freq_*` (percentage of characters matching `; ( [ ! $ #`),
  3 capital-letter run-length features (average, longest, total).
- **Missing values:** none. **Split:** 80% train / 20% test, stratified, `random_state=42`.
- **Preprocessing:** `StandardScaler` inside the pipelines of Logistic Regression, KNN and SVM
  (Decision Tree and Naive Bayes use raw values). Scaling is saved inside each `.joblib` model.

## c. GitHub repository link
- Repository: **<PASTE YOUR GITHUB REPO URL>**
- Live Streamlit app: **<PASTE YOUR STREAMLIT URL>**

### Repository structure
```
├── app.py               Streamlit app
├── train_models.py      trains + saves the 5 models, writes results and test_data.csv
├── evaluation.py        shared metric function
├── download_data.py     fetches the Spambase dataset
├── requirements.txt
├── test_data.csv        held-out 20% test set (features + target)
├── results.csv          metrics table
└── model/               5 saved .joblib models
```

## d. Models used

### Comparison table (test set)
<!-- PASTE the table printed by train_models.py (also in results.md) -->
| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | | | | | | |
| Decision Tree | | | | | | |
| KNN | | | | | | |
| Naive Bayes | | | | | | |
| Support Vector Machine | | | | | | |

### Observations
<!-- WRITE from YOUR numbers. Use the train-vs-test table in results.md for overfitting remarks. -->
| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | |
| Decision Tree | |
| KNN | |
| Naive Bayes | |
| Support Vector Machine | |
| **Overall winner** | |

## How to run
```bash
pip install -r requirements.txt
python download_data.py
python train_models.py
streamlit run app.py
```
