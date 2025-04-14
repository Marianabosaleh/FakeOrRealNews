import pickle
import torch

# Load preprocessed data
with open('data/processed_data.pkl', 'rb') as f:
    data = pickle.load(f)

# Extract the data
X_train_tfidf = data['X_train_tfidf']
X_val_tfidf = data['X_val_tfidf']
X_test_tfidf = data['X_test_tfidf']
y_train = data['y_train']
y_val = data['y_val']
y_test = data['y_test']

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train_tfidf.toarray(), dtype=torch.float32)
X_val_tensor = torch.tensor(X_val_tfidf.toarray(), dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_tfidf.toarray(), dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.map({'FAKE': 0, 'REAL': 1}).values, dtype=torch.long)
y_val_tensor = torch.tensor(y_val.map({'FAKE': 0, 'REAL': 1}).values, dtype=torch.long)
y_test_tensor = torch.tensor(y_test.map({'FAKE': 0, 'REAL': 1}).values, dtype=torch.long)

print("Processed data loaded and converted to tensors successfully.")
