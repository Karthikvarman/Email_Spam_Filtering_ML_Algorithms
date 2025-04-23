import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# Suppress warnings
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

# === Load Dataset ===
input_file_path = "spam_features_cleaned.csv"

print("Loading dataset...")
data = pd.read_csv(input_file_path)

# Auto-label messages if no 'label' column exists
spam_keywords = ["win", "prize", "free", "cash", "urgent", "lottery", "offer", "exclusive", "congratulations"]
if "label" not in data.columns:
    print("Labeling messages based on keywords...")
    data['label'] = data['Message'].apply(
        lambda x: "spam" if any(word in x.lower() for word in spam_keywords) else "ham"
    )

data['label_encoded'] = data['label'].apply(lambda x: 1 if x == "spam" else 0)

# === Data Splitting ===
print("Splitting dataset into training and testing sets...")
messages = data['Message']
labels = data['label_encoded']
X_train, X_test, y_train, y_test = train_test_split(messages, labels, test_size=0.2, random_state=42)

# === Text Vectorization ===
print("Applying TF-IDF vectorization...")
vectorizer = TfidfVectorizer(max_features=3000, stop_words='english', ngram_range=(1, 2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
X_train_dense = X_train_tfidf.toarray()
X_test_dense = X_test_tfidf.toarray()

# === Model Training ===
print("Training models...")
gnb = GaussianNB()
mnb = MultinomialNB()

gnb.fit(X_train_dense, y_train)
mnb.fit(X_train_tfidf, y_train)

# === TF-IDF Model ===
tfidf_predictions = (X_test_tfidf.sum(axis=1) > 0).astype(int).A1

# === Predictions ===
print("Generating predictions...")
gnb_preds_prob = gnb.predict_proba(X_test_dense)[:, 1]
mnb_preds_prob = mnb.predict_proba(X_test_tfidf)[:, 1]
hybrid_preds_prob = 0.5 * gnb_preds_prob + 0.5 * mnb_preds_prob
hybrid_predictions = (hybrid_preds_prob > 0.5).astype(int)

# === Evaluation ===
def evaluate_model(predictions, y_test, model_name):
    cm = confusion_matrix(y_test, predictions)
    report = classification_report(y_test, predictions, target_names=["Ham", "Spam"], output_dict=True)
    return cm, report

gnb_cm, gnb_report = evaluate_model(gnb.predict(X_test_dense), y_test, "Gaussian Naive Bayes")
mnb_cm, mnb_report = evaluate_model(mnb.predict(X_test_tfidf), y_test, "Multinomial Naive Bayes")
hybrid_cm, hybrid_report = evaluate_model(hybrid_predictions, y_test, "Hybrid Model")
tfidf_cm, tfidf_report = evaluate_model(tfidf_predictions, y_test, "TF-IDF Model")

# === Plot All Confusion Matrices ===
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
conf_matrices = [(gnb_cm, "Gaussian Naive Bayes"), (mnb_cm, "Multinomial Naive Bayes"), 
                 (hybrid_cm, "Hybrid Model"), (tfidf_cm, "TF-IDF Model")]

for ax, (cm, title) in zip(axes.flatten(), conf_matrices):
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Ham", "Spam"], 
                yticklabels=["Ham", "Spam"], ax=ax)
    ax.set_title(f"{title} Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

plt.tight_layout()
plt.show(block=False)

# === Plot All Classification Reports ===
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
reports = [(gnb_report, "Gaussian Naive Bayes"), (mnb_report, "Multinomial Naive Bayes"), 
           (hybrid_report, "Hybrid Model"), (tfidf_report, "TF-IDF Model")]

for ax, (report, title) in zip(axes.flatten(), reports):
    report_df = pd.DataFrame(report).transpose()
    sns.heatmap(report_df.iloc[:-1, :].T, annot=True, fmt=".2f", cmap="coolwarm", cbar=False, ax=ax)
    ax.set_title(f"{title} Classification Report")

plt.tight_layout()
plt.show(block=False)

# === Plot Metrics Bar Graph ===
metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
hybrid_values = [
    accuracy_score(y_test, hybrid_predictions),
    precision_score(y_test, hybrid_predictions),
    recall_score(y_test, hybrid_predictions),
    f1_score(y_test, hybrid_predictions)
]

metrics_df = pd.DataFrame({"Metric": metrics, "Score": hybrid_values})

plt.figure(figsize=(6, 4))
sns.barplot(data=metrics_df, x="Metric", y="Score", palette="viridis")
plt.ylim(0, 1)
plt.title("Hybrid Model Metrics")
plt.ylabel("Score")
plt.xlabel("Metrics")
plt.show(block=False)

print("Displaying all plots...")
plt.show()
