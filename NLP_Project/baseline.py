import pickle
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import json

# Load preprocessed data
with open('data/processed_data.pkl', 'rb') as f:
    data = pickle.load(f)

# Extract data
X_train_tfidf = data['X_train_tfidf']
X_test_tfidf = data['X_test_tfidf']
y_train = data['y_train']
y_test = data['y_test']

# Display class distribution in the training set
print("Class Distribution in Training Data:")
print(y_train.value_counts(normalize=True))

# Visualize class distribution
y_train.value_counts(normalize=True).plot(kind='bar')
plt.title('Class Distribution in Training Set')
plt.xticks(ticks=[0, 1], labels=['FAKE', 'REAL'], rotation=0)
plt.ylabel('Proportion')
plt.show()

# Create a dummy classifier that always predicts the majority class
dummy_clf = DummyClassifier(strategy='most_frequent')
dummy_clf.fit(X_train_tfidf, y_train)

# Predict using the dummy classifier
y_pred_dummy = dummy_clf.predict(X_test_tfidf)

# Evaluate the baseline model
baseline_accuracy = accuracy_score(y_test, y_pred_dummy)
print("\nBaseline Metrics:")
print(f"Accuracy: {baseline_accuracy:.4f}")
print(classification_report(y_test, y_pred_dummy, zero_division=1, target_names=['FAKE', 'REAL']))

# Save baseline metrics
baseline_metrics = classification_report(y_test, y_pred_dummy, zero_division=1, target_names=['FAKE', 'REAL'], output_dict=True)
with open('baseline_metrics.json', 'w') as f:
    json.dump(baseline_metrics, f, indent=4)
print("\nBaseline metrics saved to 'baseline_metrics.json'")

# Output some predictions to see the baseline in action
print("\nSample Predictions:")
print(y_pred_dummy[:10])
