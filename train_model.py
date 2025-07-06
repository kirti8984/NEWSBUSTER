import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import numpy as np

# ------------------ Load and Clean Data ------------------
try:
    df_fake = pd.read_csv("Fake.csv", quoting=3, engine='python', error_bad_lines=False)
except:
    df_fake = pd.read_csv("Fake.csv", on_bad_lines='skip', engine='python')

try:
    df_true = pd.read_csv("True.csv", quoting=3, engine='python', error_bad_lines=False)
except:
    df_true = pd.read_csv("True.csv", on_bad_lines='skip', engine='python')

# Add labels
df_fake["label"] = 0  # Fake
df_true["label"] = 1  # Real

# Combine and clean
df = pd.concat([df_fake, df_true])
df = df[["title", "label"]]
df.dropna(inplace=True)

# ------------------ Features and Labels ------------------
X = df["title"]
y = df["label"]

# ------------------ Vectorization ------------------
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_vectorized = vectorizer.fit_transform(X)

# ------------------ Train-test Split ------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, y, test_size=0.2, random_state=42
)

# ------------------ Model Training ------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ------------------ Evaluation ------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("✅ Accuracy:", accuracy)
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

# ------------------ Save Model & Vectorizer ------------------
joblib.dump(model, "fake_news_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

