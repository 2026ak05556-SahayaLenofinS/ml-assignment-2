import os
import warnings
import sys
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from evaluation import score_model

warnings.filterwarnings("ignore", category=FutureWarning)
DATA_PATH = sys.argv[1] if len(sys.argv) > 1 else os.path.join("data", "spambase.csv")
LABEL = "spam"
SEED = 42
STUDENT_NAME = "SAHAYA LENOFIN S"      # <- edit: shown in the lab screenshot output
BITS_ID = "2026AK05556"        # <- edit

print("=" * 60)
print("BITS Pilani - M.Tech (AIML) - Machine Learning Assignment 2")
print("Student Name :", STUDENT_NAME)
print("BITS ID      :", BITS_ID)
print("=" * 60)

frame = pd.read_csv(DATA_PATH)
print("Dataset shape:", frame.shape)
print("Class counts:", frame[LABEL].value_counts().to_dict())

feats, labels = frame.drop(columns=LABEL), frame[LABEL]
feats_train, feats_test, labels_train, labels_test = train_test_split(
    feats, labels, test_size=0.2, stratify=labels, random_state=SEED)

# Scaling lives inside the pipelines, so each saved .joblib carries its own preprocessing.
candidates = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)),
    "Decision Tree": DecisionTreeClassifier(max_depth=8, min_samples_leaf=5, random_state=SEED),
    "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=9)),
    "Naive Bayes": GaussianNB(),
    "SVM": make_pipeline(StandardScaler(), SVC(kernel="rbf", C=2.0, probability=True, random_state=SEED)),
}

os.makedirs("model", exist_ok=True)
test_rows, train_acc = {}, {}
for label, clf in candidates.items():
    clf.fit(feats_train, labels_train)
    joblib.dump(clf, os.path.join("model", label.lower().replace(" ", "_") + ".joblib"))
    test_rows[label] = score_model(clf, feats_test, labels_test)
    train_acc[label] = score_model(clf, feats_train, labels_train)["Accuracy"]

table = pd.DataFrame(test_rows).T.round(4)
table.index.name = "ML Model Name"
table.to_csv("results.csv")

gap = pd.DataFrame({"Train accuracy": pd.Series(train_acc),
                    "Test accuracy": table["Accuracy"]}).round(4)
gap["Gap"] = (gap["Train accuracy"] - gap["Test accuracy"]).round(4)

with open("results.md", "w") as fh:
    fh.write("## Test-set metrics (paste into README)\n\n" + table.to_markdown() + "\n\n")
    fh.write("## Train vs test accuracy (use for overfitting observations)\n\n" + gap.to_markdown() + "\n\n")
    fh.write(f"Best by MCC: {table['MCC'].idxmax()} | Best by AUC: {table['AUC'].idxmax()} "
             f"| Best by F1: {table['F1'].idxmax()}\n")

pd.concat([feats_test, labels_test], axis=1).to_csv("test_data.csv", index=False)

print("\n" + table.to_markdown())
print("\n" + gap.to_markdown())
print("\nBest by MCC:", table["MCC"].idxmax())
print("Saved: model/*.joblib, results.csv, results.md, test_data.csv")
