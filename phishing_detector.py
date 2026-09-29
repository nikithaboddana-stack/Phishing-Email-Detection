import pandas as pd
import re
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay


# ==========================================
# 1. LOAD DATASET
# ==========================================

data = pd.read_csv("dataset.csv")

print("=" * 60)
print("        PHISHING EMAIL DETECTION MODEL")
print("=" * 60)

print("\nDataset loaded successfully!")
print("Total emails:", len(data))


# ==========================================
# 2. EXTRACT URL FEATURES
# ==========================================

def count_urls(text):
    urls = re.findall(r'https?://\S+|www\.\S+', str(text))
    return len(urls)


def count_suspicious_words(text):
    suspicious_words = [
        "urgent",
        "verify",
        "account suspended",
        "click here",
        "password",
        "login",
        "confirm",
        "bank",
        "winner",
        "claim",
        "limited time",
        "security alert"
    ]

    text = str(text).lower()

    return sum(word in text for word in suspicious_words)


data["url_count"] = data["text"].apply(count_urls)
data["suspicious_word_count"] = data["text"].apply(count_suspicious_words)


# ==========================================
# 3. CONVERT LABELS
# ==========================================

data["label"] = data["label"].map({
    "phishing": 1,
    "safe": 0
})


# Remove invalid rows
data = data.dropna(subset=["text", "label"])


# ==========================================
# 4. SPLIT DATA
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    data["text"],
    data["label"],
    test_size=0.2,
    random_state=42,
    stratify=data["label"]
)


# ==========================================
# 5. TF-IDF FEATURE EXTRACTION
# ==========================================

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    max_features=5000
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


# ==========================================
# 6. TRAIN MACHINE LEARNING MODEL
# ==========================================

model = LogisticRegression(max_iter=1000)

model.fit(X_train_tfidf, y_train)


# ==========================================
# 7. MAKE PREDICTIONS
# ==========================================

predictions = model.predict(X_test_tfidf)


# ==========================================
# 8. ACCURACY
# ==========================================

accuracy = accuracy_score(y_test, predictions)

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")


# ==========================================
# 9. CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(y_test, predictions)

print("\nConfusion Matrix:")
print(cm)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Safe", "Phishing"]
)

display.plot()

plt.title("Phishing Email Detection - Confusion Matrix")
plt.tight_layout()

plt.savefig("confusion_matrix.png")

plt.show()


# ==========================================
# 10. TEST NEW EMAIL
# ==========================================

print("\n" + "=" * 60)
print("             TEST AN EMAIL")
print("=" * 60)

email = input("\nEnter email text: ")

email_vector = vectorizer.transform([email])

prediction = model.predict(email_vector)[0]

probability = model.predict_proba(email_vector)[0].max() * 100


if prediction == 1:
    result = "PHISHING"
else:
    result = "SAFE"


print("\nPrediction :", result)
print("Confidence :", round(probability, 2), "%")

print("\nAnalysis completed!")