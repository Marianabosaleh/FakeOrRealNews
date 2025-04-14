import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

# Step 1: Load Preprocessed Data
with open('data/processed_data.pkl', 'rb') as f:
    data = pickle.load(f)
    X_train_tfidf = data['X_train_tfidf']
    X_test_tfidf = data['X_test_tfidf']
    y_train = data['y_train']
    y_test = data['y_test']

print("Data loaded successfully!")

# Step 2: Train/Validation/Test Split
X_train, X_val, y_train, y_val = train_test_split(
    X_train_tfidf, y_train, test_size=0.2, random_state=42, stratify=y_train
)
print("Train/Validation/Test split complete!")

# Step 3: Train Logistic Regression with Regularization
logistic_model = LogisticRegression(C=0.1, max_iter=1000, random_state=42)  # Regularization with C=0.1
logistic_model.fit(X_train, y_train)

# Step 4: Evaluate on Validation Set
y_val_pred = logistic_model.predict(X_val)
val_accuracy = accuracy_score(y_val, y_val_pred)
print(f"Validation Accuracy: {val_accuracy:.4f}")

# Step 5: Evaluate on Test Set
y_test_pred = logistic_model.predict(X_test_tfidf)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"Test Accuracy: {test_accuracy:.4f}")

# Display Classification Report for Test Set
print("\nClassification Report (Test Set):")
print(classification_report(y_test, y_test_pred, target_names=['FAKE', 'REAL'], zero_division=1))

# Step 6: Visualize Confusion Matrix
ConfusionMatrixDisplay.from_estimator(
    logistic_model, X_test_tfidf, y_test, display_labels=['FAKE', 'REAL'], cmap='Blues'
)
plt.title('Confusion Matrix - Logistic Regression')
plt.show()

# Step 7: Learning Curve Analysis
train_sizes = np.linspace(0.1, 1.0, 10)
train_scores = []
val_scores = []

for frac in train_sizes:
    if frac == 1.0:
        X_frac, y_frac = X_train, y_train  # Use the entire training data
    else:
        X_frac, _, y_frac, _ = train_test_split(X_train, y_train, train_size=frac, random_state=42)
    logistic_model.fit(X_frac, y_frac)
    train_scores.append(logistic_model.score(X_frac, y_frac))
    val_scores.append(logistic_model.score(X_val, y_val))

plt.plot(train_sizes * 100, train_scores, label='Training Accuracy')
plt.plot(train_sizes * 100, val_scores, label='Validation Accuracy')
plt.xlabel('Training Set Size (%)')
plt.ylabel('Accuracy')
plt.title('Learning Curve - Logistic Regression')
plt.legend()
plt.show()
