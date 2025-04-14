import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score
import matplotlib.pyplot as plt
import numpy as np
import pickle

# Check for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Step 1: Load Preprocessed Data
with open('data/processed_data.pkl', 'rb') as f:
    data = pickle.load(f)

X_train_tfidf = data['X_train_tfidf']
X_val_tfidf = data['X_val_tfidf']
X_test_tfidf = data['X_test_tfidf']
y_train = data['y_train'].map({'FAKE': 0, 'REAL': 1}).values
y_val = data['y_val'].map({'FAKE': 0, 'REAL': 1}).values
y_test = data['y_test'].map({'FAKE': 0, 'REAL': 1}).values

# Convert TF-IDF matrices to PyTorch tensors
X_train_tensor = torch.tensor(X_train_tfidf.toarray(), dtype=torch.float32).to(device)
X_val_tensor = torch.tensor(X_val_tfidf.toarray(), dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test_tfidf.toarray(), dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
y_val_tensor = torch.tensor(y_val, dtype=torch.long).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

# Create DataLoader for batching
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# Handle class imbalance
class_counts = np.bincount(y_train)
class_weights = torch.tensor([1.0 / count for count in class_counts], dtype=torch.float32).to(device)

# Step 2: Define Attention Layer with Multi-Head Attention
class Attention(nn.Module):
    def __init__(self, hidden_dim, num_heads=8):
        super(Attention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)

    def forward(self, lstm_output):
        attn_output, attn_weights = self.multihead_attn(lstm_output, lstm_output, lstm_output)
        return attn_output.mean(dim=1), attn_weights.mean(dim=1)  # Average over heads

# Step 3: Define Enhanced BiLSTM with Attention Model
class EnhancedBiLSTMWithAttention(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim, num_layers, dropout):
        super(EnhancedBiLSTMWithAttention, self).__init__()
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(embedding_dim // 2)
        )
        self.lstm = nn.LSTM(
            embedding_dim // 2, hidden_dim, num_layers=num_layers, bidirectional=True, batch_first=True, dropout=dropout
        )
        self.attention = Attention(hidden_dim * 2, num_heads=8)  # Increased attention heads
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc1_bn = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.embedding(x)  # Learnable embedding -> (batch_size, embedding_dim // 2)
        x = x.unsqueeze(1)  # Add sequence dimension -> (batch_size, seq_len=1, embedding_dim // 2)
        lstm_out, _ = self.lstm(x)  # LSTM output -> (batch_size, seq_len, hidden_dim * 2)

        # Match dimensions between x and lstm_out for residual connection
        x_resized = torch.cat([x, x], dim=-1)  # Duplicate x along the last dimension
        x_resized = x_resized.expand_as(lstm_out)  # Match size with lstm_out

        x = x_resized + lstm_out  # Residual connection
        context, attn_weights = self.attention(x)  # Apply attention
        x = self.dropout(context)
        x = torch.relu(self.fc1(x))
        x = self.fc1_bn(x)
        x = self.dropout(x)
        return self.fc2(x), attn_weights


# Initialize model
input_dim = X_train_tfidf.shape[1]
embedding_dim = 256
hidden_dim = 128
output_dim = 2
num_layers = 3  # Increased depth for better feature extraction
dropout = 0.4  # Higher dropout to reduce overfitting

model = EnhancedBiLSTMWithAttention(input_dim, embedding_dim, hidden_dim, output_dim, num_layers, dropout).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.AdamW(model.parameters(), lr=0.0003, weight_decay=5e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

# Training Loop with Early Stopping and Gradient Accumulation
epochs = 50
patience = 5
accumulation_steps = 4
best_val_loss = float('inf')
best_f1 = 0
patience_counter = 0

train_losses = []
val_losses = []
val_accuracies = []

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    optimizer.zero_grad()
    for i, (X_batch, y_batch) in enumerate(train_loader):
        outputs, _ = model(X_batch)
        loss = criterion(outputs, y_batch) / accumulation_steps
        loss.backward()
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        running_loss += loss.item()
    train_losses.append(running_loss / len(train_loader))

    # Validation Phas
    model.eval()
    val_loss = 0.0
    correct_predictions = 0  # To calculate accuracy
    total_predictions = 0
    y_val_pred = []
    y_val_true = []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            outputs, _ = model(X_batch)
            loss = criterion(outputs, y_batch)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            y_val_pred.extend(predicted.cpu().numpy())
            y_val_true.extend(y_batch.cpu().numpy())

            # Accuracy Calculation
            correct_predictions += (predicted == y_batch).sum().item()
            total_predictions += y_batch.size(0)

    val_loss /= len(val_loader)
    val_losses.append(val_loss)
    val_accuracy = correct_predictions / total_predictions
    scheduler.step(val_loss)
    val_f1 = f1_score(y_val_true, y_val_pred, average='weighted')
    val_accuracies.append(val_f1)

    print(
        f"Epoch {epoch + 1}/{epochs}, Training Loss: {train_losses[-1]:.4f}, Validation Loss: {val_loss:.4f}, F1 Score: {val_f1:.4f}, Accuracy: {val_accuracy:.4f}")


    # Save the best model
    if val_f1 > best_f1:
        best_f1 = val_f1
        patience_counter = 0
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered!")
            break

# Load the best model
model.load_state_dict(torch.load('best_model.pth', weights_only=True))

# Step 5: Evaluate the Model
# Step 5: Evaluate the Model
model.eval()
y_pred = []
y_true = []
attention_weights = []
correct_predictions = 0  # Initialize correct predictions counter
total_predictions = 0    # Initialize total predictions counter

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        outputs, attn_weights = model(X_batch)
        _, predicted = torch.max(outputs, 1)
        y_pred.extend(predicted.cpu().numpy())
        y_true.extend(y_batch.cpu().numpy())
        attention_weights.append(attn_weights.cpu().numpy())

        # Accuracy Calculation
        correct_predictions += (predicted == y_batch).sum().item()
        total_predictions += y_batch.size(0)

# Calculate Test Accuracy
test_accuracy = correct_predictions / total_predictions

# Classification Report
report = classification_report(y_true, y_pred, target_names=['FAKE', 'REAL'], zero_division=1)
print("\nBiLSTM Metrics:")
print(report)

# Display Test Accuracy
print(f"Test Accuracy: {test_accuracy:.4f}")

# Confusion Matrix
conf_matrix = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=['FAKE', 'REAL'])
disp.plot(cmap='Blues')
plt.title('Confusion Matrix - Enhanced BiLSTM')
plt.savefig('confusion_matrix_enhanced.png', dpi=300)
plt.show()

# Loss Plot (Optional, if used for additional insights)
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training vs. Validation Loss')
plt.legend()
plt.savefig('loss_plot_enhanced.png', dpi=300)
plt.show()

