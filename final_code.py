import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# Load Dataset
input_file_path = "spam_features_cleaned.csv"
data = pd.read_csv(input_file_path)

# Auto-label messages if no 'label' column exists
spam_keywords = ["win", "prize", "free", "cash", "urgent", "lottery", "offer", "exclusive", "congratulations", 
                 "win", "winner", "prize", "jackpot", "lottery", "free", "money", "cash", "million",
    "dollars", "earn", "income", "profit", "investment", "bonus", "reward", "refund", "offer",
    "urgent", "hurry", "act now", "limited time", "last chance", "don't miss", "final notice", 
    "response needed", "immediate", "risk-free",
    "sale", "exclusive", "discount", "buy now", "order now", "clearance", "deals", "giveaway",
    "trial", "cheap", "promo", "save big", "100%free", "subscribe", "subscribe now",
    "congratulations", "dear friend", "dear user", "click here", "visit link", "verify",
    "password", "account", "login", "billing", "update info", "confidential", "security alert",
    "call now", "toll-free", "contact us", "unsubscribe", "opt-out", "this is not spam", 
    "remove me", "click below",
    "miracle", "cure", "weight loss", "no prescription", "FDA approved", "anti-aging",
    "loan", "credit card", "mortgage", "bad credit", "get out of debt", "consolidate",
    "financial freedom",
    "xxx", "adult", "sex", "hot", "nude", "erotic", "dating", "singles", "meet"]
if "label" not in data.columns:
    data['label'] = data['Message'].apply(
        lambda x: "spam" if any(word in x.lower() for word in spam_keywords) else "ham"
    )

# Save to separate CSV files for spam and ham
spam_data = data[data['label'] == 'spam']
ham_data = data[data['label'] == 'ham']
spam_data.to_csv("spam.csv", index=False)
ham_data.to_csv("ham.csv", index=False)

# Encode labels for ML
data['label_encoded'] = data['label'].apply(lambda x: 1 if x == "spam" else 0)

# Split dataset
messages = data['Message']
labels = data['label_encoded']
X_train, X_test, y_train, y_test = train_test_split(messages, labels, test_size=0.2, random_state=42)

# Text Vectorization
vectorizer = TfidfVectorizer(max_features=3000, stop_words='english', ngram_range=(1, 2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Initialize models
models = {
    "SVM": SVC(kernel='linear', probability=True, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Naive Bayes": MultinomialNB()
}

# Train models and evaluate
results = {}
for model_name, model in models.items():
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)
    
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["Ham", "Spam"])
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    results[model_name] = {
        "Confusion Matrix": cm,
        "Classification Report": report,
        "Metrics": {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1
        }
    }
    print(f"\n=== {model_name} ===")
    print(f"Classification Report:\n{report}")
    print(f"Metrics: {results[model_name]['Metrics']}")
    print(f"Confusion Matrix:\n{cm}")

# Identify Best Algorithm
best_algorithm = max(
    results.items(),
    key=lambda x: (x[1]["Metrics"]["Accuracy"], x[1]["Metrics"]["F1 Score"])
)
best_model_name = best_algorithm[0]
best_model = models[best_model_name]
print("\n=== BEST ALGORITHM BASED ON METRICS ===")
print(f"The best algorithm is {best_model_name} based on Accuracy and F1 Score.")

# Plotting
plt.ion()  # Enable interactive mode

# Confusion Matrices
fig1, axes = plt.subplots(3, 2, figsize=(18, 18))
for idx, (model_name, result) in enumerate(results.items()):
    cm = result["Confusion Matrix"]
    ax = axes[idx, 0]
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Ham", "Spam"], yticklabels=["Ham", "Spam"], ax=ax
    )
    ax.set_title(f"Confusion Matrix - {model_name}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    
    tn, fp, fn, tp = cm.ravel()
    explanation = (
        f"True Negatives (Ham correctly identified): {tn}\n"
        f"False Positives (Ham misclassified as Spam): {fp}\n"
        f"False Negatives (Spam misclassified as Ham): {fn}\n"
        f"True Positives (Spam correctly identified): {tp}"
    )
    ax_info = axes[idx, 1]
    ax_info.axis("off")
    ax_info.text(
        0.5, 0.5, explanation, fontsize=12, color="black",
        ha="center", va="center", wrap=True
    )

# Classification Reports
fig2, ax2 = plt.subplots(figsize=(12, 6))
classification_text = "".join(
    [f"{model}\n{results[model]['Classification Report']}\n\n" for model in results]
)
ax2.axis("off")
ax2.text(0.5, 0.5, classification_text, fontsize=10, ha='center', va='center', family='monospace')
ax2.set_title("Classification Reports")

# Metrics Comparison Plot
fig3, ax3 = plt.subplots(figsize=(10, 6))
metrics_df = pd.DataFrame({model: result["Metrics"] for model, result in results.items()})
metrics_df = metrics_df.transpose()
metrics_df.plot(kind="bar", ax=ax3, colormap="viridis")
ax3.set_title("Comparison of Metrics Across Models")
ax3.set_ylabel("Scores")
ax3.set_xlabel("Models")
ax3.set_xticklabels(metrics_df.index, rotation=45)
ax3.legend(title="Metrics")

plt.show(block=False)  # Show images simultaneously
plt.pause(0.1)  # Allow time for rendering
input("Press Enter to continue...")  # Wait for user before proceeding

# Email Classification Prompt
while True:
    email_message = input("Enter an email message to classify: ")
    email_tfidf = vectorizer.transform([email_message])
    prediction = best_model.predict(email_tfidf)[0]
    label = "Spam" if prediction == 1 else "Ham"
    print(f"\n=== Classification Result ===")
    print(f"{best_model_name}: {label}")
    
    cont = input("Do you want to continue? (yes/no): ").strip().lower()
    if cont != "yes":
        print("Thank you!")
        break