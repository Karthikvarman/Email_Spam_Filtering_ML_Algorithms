import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
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

# === Model Training ===
print("Training Logistic Regression model...")
log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train_tfidf, y_train)

# === Predictions ===
print("Generating predictions for the entire dataset...")
data_tfidf = vectorizer.transform(data['Message'])
data['Predicted'] = log_reg.predict(data_tfidf)
data['Predicted_Label'] = data['Predicted'].apply(lambda x: "spam" if x == 1 else "ham")

# === Save Spam and Ham Messages ===
spam_data = data[data['Predicted_Label'] == "spam"]
ham_data = data[data['Predicted_Label'] == "ham"]

spam_file_path = "spam.csv"
ham_file_path = "ham.csv"

print(f"Saving spam messages to {spam_file_path}...")
spam_data.to_csv(spam_file_path, index=False)

print(f"Saving ham messages to {ham_file_path}...")
ham_data.to_csv(ham_file_path, index=False)

# === Evaluation ===
def evaluate_model(predictions, y_test, model_name):
    cm = confusion_matrix(y_test, predictions)
    report = classification_report(y_test, predictions, target_names=["Ham", "Spam"], output_dict=True)
    print(f"\n{model_name} Confusion Matrix:")
    print(cm)
    print(f"\n{model_name} Classification Report:")
    print(classification_report(y_test, predictions, target_names=["Ham", "Spam"]))
    return cm, report

log_reg_cm, log_reg_report = evaluate_model(log_reg.predict(X_test_tfidf), y_test, "Logistic Regression")

# === Plot Figures ===

# Figure 1: Confusion Matrix
plt.figure(1)
sns.heatmap(log_reg_cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Ham", "Spam"], yticklabels=["Ham", "Spam"])
plt.title("Logistic Regression Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

# Figure 2: Classification Report
plt.figure(2)
report_df = pd.DataFrame(log_reg_report).transpose()
sns.heatmap(report_df.iloc[:-1, :].T, annot=True, fmt=".2f", cmap="coolwarm", cbar=False)
plt.title("Logistic Regression Classification Report")

# Figure 3: Metrics Bar Graph
plt.figure(3)
metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
log_reg_values = [
    accuracy_score(y_test, log_reg.predict(X_test_tfidf)),
    precision_score(y_test, log_reg.predict(X_test_tfidf)),
    recall_score(y_test, log_reg.predict(X_test_tfidf)),
    f1_score(y_test, log_reg.predict(X_test_tfidf))
]

metrics_df = pd.DataFrame({"Metric": metrics, "Score": log_reg_values})
sns.barplot(data=metrics_df, x="Metric", y="Score", palette="viridis")
plt.ylim(0, 1)
plt.title("Logistic Regression Metrics")
plt.ylabel("Score")
plt.xlabel("Metrics")

# Display all figures
print("Displaying all figures...")
plt.show()
