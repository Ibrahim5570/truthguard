# train.py
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os
import json

# -------------------------------
# 1. Load & Prepare Data
# -------------------------------
print("📂 Loading dataset: FakeNewsNet.csv...")
df = pd.read_csv('FakeNewsNet.csv', names=['title', 'url', 'domain', 'tweet_num', 'real'], header=0)
df.dropna(subset=['title'], inplace=True)
df['title'] = df['title'].astype(str)
df['real'] = df['real'].astype(int)

# -------------------------------
# 2. Data Augmentation: Add Known Misinformation
# -------------------------------
augmented_examples = [
    {"title": "CDC confirms masks cause oxygen loss in children", "real": 0},
    {"title": "Scientists say climate change is just a natural cycle", "real": 0},
    {"title": "Apple unveils iPhone 16 with holographic display", "real": 0},
    {"title": "Queen Elizabeth II celebrates 100th birthday", "real": 0},
    {"title": "5G networks spread coronavirus", "real": 0},
    {"title": "Vaccines cause autism, says new study", "real": 0},
    {"title": "Bill Gates implants microchips via vaccines", "real": 0},
    {"title": "NASA faked the moon landing again", "real": 0},
    {"title": "Trump wins Nobel Peace Prize for ending Ukraine war", "real": 0},
    {"title": "Eating bleach cures COVID-19", "real": 0},
    {"title": "Aliens invade Poland and say Hitler was right", "real": 0},
    {"title": "Beyonce and Jay-Z expecting fourth child via surrogate", "real": 0}
]

df_aug = pd.DataFrame(augmented_examples)
df = pd.concat([df, df_aug], ignore_index=True)
print(f"✅ Dataset after augmentation: {len(df)} samples")

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
# 5. Balance Dataset
# -------------------------------
from sklearn.utils import resample
df_fake = df[df['real'] == 0]
df_real = df[df['real'] == 1]
n_samples = min(len(df_fake), len(df_real))
df_fake_balanced = resample(df_fake, n_samples=n_samples, random_state=42)
df_real_balanced = resample(df_real, n_samples=n_samples, random_state=42)
df_balanced = pd.concat([df_fake_balanced, df_real_balanced])

print(f"✅ Balanced dataset: {len(df_balanced)} samples (50/50)")

# -------------------------------
# 6. Enhanced Text Cleaning
# -------------------------------
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s!?]+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df_balanced['cleaned_text'] = df_balanced['title'].apply(clean_text)
X = df_balanced['cleaned_text']
y = df_balanced['real']

# -------------------------------
# 7. TF-IDF Vectorization
# -------------------------------
print("🔄 Vectorizing text (TF-IDF)...")
vectorizer = TfidfVectorizer(
    max_features=10000,
    stop_words='english',
    ngram_range=(1, 3),
    min_df=1,
    max_df=0.8
)
X_vec = vectorizer.fit_transform(X).toarray()
y_vec = y.values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y_vec, test_size=0.2, random_state=42, stratify=y_vec
)

# Convert to tensors
X_train_t = torch.FloatTensor(X_train)
X_test_t = torch.FloatTensor(X_test)
y_train_t = torch.FloatTensor(y_train).unsqueeze(1)
y_test_t = torch.FloatTensor(y_test).unsqueeze(1)

train_data = torch.utils.data.TensorDataset(X_train_t, y_train_t)
test_data = torch.utils.data.TensorDataset(X_test_t, y_test_t)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=False)

# -------------------------------
# 8. Model with Dropout
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

# Initialize
input_size = X_train.shape[1]
model = NewsClassifier(input_size)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# -------------------------------
# 9. Training Loop
# -------------------------------
epochs = 2
train_losses, test_losses = [], []
train_accs, test_accs = [], []

print("🚀 Starting training...\n")
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        predicted = (outputs > 0.5).float()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_acc = correct / total

    # Evaluation
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_loss = test_loss / len(test_loader)
    test_acc = correct / total

    train_losses.append(train_loss)
    test_losses.append(test_loss)
    train_accs.append(train_acc)
    test_accs.append(test_acc)

    print(f"Epoch {epoch+1}/{epochs} - "
          f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
          f"Test Loss: {test_loss:.4f}, Acc: {test_acc:.4f}")

# -------------------------------
# 10. Evaluate & Plot Results
# -------------------------------
# Accuracy & Loss Plots
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_accs, label='Train Accuracy')
plt.plot(test_accs, label='Test Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Confusion Matrix
model.eval()
with torch.no_grad():
    y_pred = model(X_test_t)
    y_pred_labels = (y_pred > 0.5).float().numpy().flatten()

cm = confusion_matrix(y_test, y_pred_labels)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Fake', 'Real'],
            yticklabels=['Fake', 'Real'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Classification Report
print("📋 Classification Report:")
print(classification_report(y_test, y_pred_labels, target_names=['Fake', 'Real']))

# -------------------------------
# 11. Save Model
# -------------------------------
os.makedirs('models', exist_ok=True)
joblib.dump(vectorizer, 'models/tfidf_vectorizer.pkl')
torch.save(model.state_dict(), 'models/fake_news_model.pth')
print("✅ Final model saved to 'models/'")
