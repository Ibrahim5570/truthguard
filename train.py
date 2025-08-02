import os
import sys
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import joblib
from datetime import datetime
import re

# Determine environment and set appropriate model directory
IS_STREAMLIT_CLOUD = "STREAMLIT_SERVER_PORT" in os.environ or os.getenv("HOSTNAME", "").startswith("service-")
MODEL_DIR = os.environ.get("MODEL_DIR", "/tmp/models" if IS_STREAMLIT_CLOUD else "models")

print(f"DEBUG: IS_STREAMLIT_CLOUD = {IS_STREAMLIT_CLOUD}")
print(f"DEBUG: MODEL_DIR = {MODEL_DIR}")

# Create directory if needed
os.makedirs(MODEL_DIR, exist_ok=True)
print(f"DEBUG: Created model directory at {MODEL_DIR}")

# Text cleaning function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s!?]+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# -------------------------------
# 1. Load the main dataset
# -------------------------------
print("🔄 Loading main dataset...")
try:
    df = pd.read_csv('data/fake_news_data.csv')
    print(f"✅ Loaded {len(df)} examples from main dataset")
except Exception as e:
    print(f"❌ Error loading main dataset: {str(e)}")
    sys.exit(1)

# -------------------------------
# 2. Preprocess the data
# -------------------------------
print("🧹 Preprocessing data...")
df['title'] = df['title'].apply(clean_text)
df = df[df['title'].str.strip() != '']
df = df.dropna(subset=['title'])
print(f"✅ Preprocessed data. Remaining examples: {len(df)}")

# -------------------------------
# 3. Load User Feedback (if available)
# -------------------------------
feedback_data = []
if os.path.exists('data/feedback.jsonl'):
    print("🔄 Loading user feedback for retraining...")
    with open('data/feedback.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                feedback_data.append({
                    'title': entry['headline'],
                    'real': int(entry['user_correction'])
                })

    if feedback_data:
        df_feedback = pd.DataFrame(feedback_data)
        df = pd.concat([df, df_feedback], ignore_index=True)
        print(f"✅ Added {len(df_feedback)} user feedback examples")

# -------------------------------
# 4. Load Correct Predictions (Positive Reinforcement)
# -------------------------------
correct_data = []
if os.path.exists('data/correct_predictions.jsonl'):
    print("✅ Loading correct predictions for positive reinforcement...")
    with open('data/correct_predictions.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                if entry['confidence'] > 75:  # Only high-confidence corrects
                    correct_data.append({
                        'title': entry['headline'],
                        'real': 1 if entry['model_prediction'] == 'Real News' else 0
                    })

    if correct_data:
        df_correct = pd.DataFrame(correct_data)
        df = pd.concat([df, df_correct], ignore_index=True)
        print(f"✅ Added {len(df_correct)} high-confidence correct predictions")

# -------------------------------
# 5. Split the data
# -------------------------------
print("SplitOptions data into train and test sets...")
X = df['title'].values
y = df['real'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"✅ Split data: {len(X_train)} training examples, {len(X_test)} test examples")

# -------------------------------
# 6. Vectorize the text
# -------------------------------
print("Vectorizerizing text with TF-IDF...")
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train).toarray()
X_test_vec = vectorizer.transform(X_test).toarray()
print(f"✅ Vectorized text. Feature count: {X_train_vec.shape[1]}")

# -------------------------------
# 7. Save the vectorizer
# -------------------------------
vectorizer_path = os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl')
joblib.dump(vectorizer, vectorizer_path)
print(f"✅ Saved vectorizer to {vectorizer_path}")

# -------------------------------
# 8. Define the neural network
# -------------------------------
class NewsClassifier(nn.Module):
    def __init__(self, input_size):
        super(NewsClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.network(x)

# -------------------------------
# 9. Train the model
# -------------------------------
print("🏋️ Training neural network...")
input_size = X_train_vec.shape[1]
model = NewsClassifier(input_size)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train_vec)
y_train_tensor = torch.FloatTensor(y_train).view(-1, 1)
X_test_tensor = torch.FloatTensor(X_test_vec)
y_test_tensor = torch.FloatTensor(y_test).view(-1, 1)

# Training loop
num_epochs = 50
batch_size = 64
best_val_loss = float('inf')
patience = 5
patience_counter = 0

for epoch in range(num_epochs):
    # Training
    model.train()
    epoch_loss = 0
    for i in range(0, len(X_train_tensor), batch_size):
        batch_X = X_train_tensor[i:i+batch_size]
        batch_y = y_train_tensor[i:i+batch_size]
        
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    # Validation
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_test_tensor)
        val_loss = criterion(val_outputs, y_test_tensor)
    
    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        # Save the best model
        model_path = os.path.join(MODEL_DIR, 'fake_news_model.pth')
        torch.save(model.state_dict(), model_path)
        print(f"✅ Saved best model to {model_path}")
    else:
        patience_counter += 1
    
    if patience_counter >= patience:
        print(f"🛑 Early stopping at epoch {epoch+1}")
        break
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(X_train_tensor):.4f}, Val Loss: {val_loss:.4f}')

# -------------------------------
# 10. Evaluate the model
# -------------------------------
print("📊 Evaluating model performance...")
model.eval()
with torch.no_grad():
    y_pred = model(X_test_tensor)
    y_pred = (y_pred > 0.5).float().numpy()

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"✅ Model evaluation:")
print(f"   Accuracy: {accuracy:.4f}")
print(f"   Precision: {precision:.4f}")
print(f"   Recall: {recall:.4f}")
print(f"   F1 Score: {f1:.4f}")

# -------------------------------
# 11. Save model metrics
# -------------------------------
metrics = {
    'date': datetime.now().strftime("%Y-%m-%d"),
    'accuracy': round(accuracy * 100, 2),
    'precision': round(precision * 100, 2),
    'recall': round(recall * 100, 2),
    'f1': round(f1 * 100, 2)
}

# Save to metrics file
metrics_file = 'data/model_metrics.jsonl'
os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
with open(metrics_file, 'a') as f:
    f.write(json.dumps(metrics) + '\n')

print(f"✅ Saved model metrics to {metrics_file}")

print("🎉 Training complete!")
