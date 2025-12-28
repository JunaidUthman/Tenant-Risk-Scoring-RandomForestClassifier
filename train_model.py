import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Configuration
INPUT_FILE = "tenant_data.csv"
MODEL_DIR = "model_artifacts"
MODEL_PATH = os.path.join(MODEL_DIR, "tenant_risk_model.pkl")

def train():
    # 1. Check if data exists
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Error: '{INPUT_FILE}' not found. Please run generate_data.py first.")
        return

    print("Step 1: Loading Data...")
    df = pd.read_csv(INPUT_FILE)
    
    # 2. Separate Features (X) and Target (y)
    X = df[['missedPeriods', 'totalDisputes']]
    y = df["label"]

    # 3. Split into Training and Testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Step 2: Training RandomForestClassifier...")
    # Using 100 trees and setting max_depth to prevent overfitting
    clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    clf.fit(X_train, y_train)

    # 4. Evaluate the Model
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Model Training Complete. Accuracy: {acc:.2f}")
    print("\nDetailed Report:\n", classification_report(y_test, y_pred))

    # 5. Save the Model
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    print(f"💾 Model saved successfully to '{MODEL_PATH}'")

if __name__ == "__main__":
    train()