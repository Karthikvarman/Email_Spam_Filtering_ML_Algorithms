import pandas as pd
import re
import string

# Load dataset from the local system
file_path = r"spam_dataset.csv"  # Update with your correct file path
df = pd.read_csv(file_path)

# Print column names to check the actual text and label column names
print("Columns in dataset:", df.columns)

# Identify the correct columns
text_column = "Message"  # Column containing email text
label_column = "Category"  # Column containing spam/ham labels

# Ensure both required columns exist
if text_column not in df.columns or label_column not in df.columns:
    raise ValueError("Required columns not found in dataset!")

# Feature extraction function
def extract_features(text):
    words = text.split()
    return {
        "num_words": len(words),
        "num_uppercase": sum(1 for c in text if c.isupper()),
        "num_lowercase_words": sum(1 for word in words if word.islower()),
        "num_chars": len(text),
        "num_special_chars": sum(1 for c in text if c in string.punctuation),
        "num_links": text.lower().count("http") + text.lower().count("click here"),
        "num_spam_words": sum(1 for word in ["win", "prize", "free", "lottery", "urgent"] if word in text.lower()),
        "num_digits": sum(1 for c in text if c.isdigit())
    }

# Apply feature extraction
df_features = df[text_column].astype(str).apply(extract_features).apply(pd.Series)

# Combine text and extracted features (excluding label column)
df_final = pd.concat([df[[text_column]], df_features], axis=1)

# Define output file path (inside your project folder)
output_path = r"spam_features_extracted-primary.csv"
df_final.to_csv(output_path, index=False)

print(f"✅ Processed dataset saved at: {output_path}")
