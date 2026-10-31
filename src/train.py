"""Train and compare the six classifiers required by the project brief."""
from pathlib import Path
import json, joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             classification_report, confusion_matrix)
from preprocessing import clean_text

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "abstracts_labeled.csv"
MODEL_DIR, REPORT_DIR = ROOT/"models", ROOT/"reports"
MODEL_DIR.mkdir(exist_ok=True); REPORT_DIR.mkdir(exist_ok=True)

def main():
    df = pd.read_csv(DATA)
    df = df.dropna(subset=["Abstract", "label"])
    df["Abstract"] = df["Abstract"].astype(str).str.strip()
    df = df[df.Abstract.ne("")].drop_duplicates(subset=["Abstract"])
    X, y = df.Abstract, df.label
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y)
    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(max_depth=80, min_samples_leaf=2, class_weight="balanced", random_state=42),
        "Naive Bayes": MultinomialNB(alpha=0.5),
        "kNN": KNeighborsClassifier(n_neighbors=5, weights="distance", metric="cosine", algorithm="brute"),
        "SVM": LinearSVC(class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=150, max_features="sqrt", min_samples_leaf=2, class_weight="balanced_subsample", n_jobs=-1, random_state=42),
    }
    rows, fitted = [], {}
    for name, estimator in models.items():
        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(preprocessor=clean_text, lowercase=False,
                 ngram_range=(1,2), min_df=2, max_df=0.98, max_features=60000,
                 sublinear_tf=True, dtype=np.float32)),
            ("classifier", estimator)
        ])
        print(f"Training {name}...")
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        pr, rc, f1, _ = precision_recall_fscore_support(y_test, pred, average="weighted", zero_division=0)
        rows.append({"model":name, "accuracy":accuracy_score(y_test,pred),
                     "precision_weighted":pr, "recall_weighted":rc, "f1_weighted":f1})
        fitted[name] = pipe
        report = classification_report(y_test,pred,output_dict=True,zero_division=0)
        pd.DataFrame(report).T.to_csv(REPORT_DIR/f"classification_report_{name.lower().replace(' ','_')}.csv")
        cm=confusion_matrix(y_test,pred,labels=sorted(y.unique()))
        plt.figure(figsize=(6,5)); sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",
            xticklabels=sorted(y.unique()),yticklabels=sorted(y.unique()))
        plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title(f"{name} — confusion matrix")
        plt.tight_layout(); plt.savefig(REPORT_DIR/f"confusion_matrix_{name.lower().replace(' ','_')}.png",dpi=150); plt.close()
    results=pd.DataFrame(rows).sort_values("f1_weighted",ascending=False)
    results.to_csv(REPORT_DIR/"model_comparison.csv",index=False)
    best_name=results.iloc[0]["model"]
    joblib.dump(fitted[best_name], MODEL_DIR/"best_model.joblib")
    (MODEL_DIR/"model_info.json").write_text(json.dumps({
        "best_model":best_name, "classes":sorted(y.unique()),
        "train_rows":len(X_train), "test_rows":len(X_test),
        "random_state":42, "text_column":"Abstract", "target_column":"label"
    },indent=2))
    pd.Series(y_test.value_counts()).to_csv(REPORT_DIR/"test_class_counts.csv",header=["count"])
    print(results.to_string(index=False))
    print("Best model:",best_name)

if __name__ == "__main__":
    main()
