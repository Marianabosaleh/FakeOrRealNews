import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Step 1: Load the dataset
file_path = 'data/fake_or_real_news.csv'  # Replace with your file path if needed
data = pd.read_csv(file_path)

# Drop unnecessary columns
data = data.drop(columns=['Unnamed: 0'])  # 'Unnamed: 0' is just an index column

# Step 2: Check class distribution
class_distribution = data['label'].value_counts()
print("Class Distribution:\n", class_distribution)

# Step 3: Splitting the dataset
X = data['text']  # Text content for features
y = data['label']  # Labels ('REAL' or 'FAKE')

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print(f"Training set size: {len(X_train)}")
print(f"Validation set size: {len(X_val)}")
print(f"Test set size: {len(X_test)}")

# Step 4: Vectorize text using TF-IDF
tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_tfidf = tfidf.fit_transform(X_train)
X_val_tfidf = tfidf.transform(X_val)
X_test_tfidf = tfidf.transform(X_test)

print("TF-IDF vectorization complete.")

# Step 5: Save the processed data
with open('data/processed_data.pkl', 'wb') as f:
    pickle.dump({
        'X_train_tfidf': X_train_tfidf,
        'X_val_tfidf': X_val_tfidf,
        'X_test_tfidf': X_test_tfidf,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'tfidf_vectorizer': tfidf
    }, f)

print("Processed data saved to 'processed_data.pkl'.")
print(f"Number of features in TF-IDF: {X_train_tfidf.shape[1]}")

