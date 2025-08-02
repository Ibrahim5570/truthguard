# predict.py
import torch
import joblib
import numpy as np
import re
import os
import json
from datetime import datetime

# -------------------------------
# 1. Load Model & Vectorizer
# -------------------------------
print("📰 Fake News Detector - Final Version\n")

if not os.path.exists('models/tfidf_vectorizer.pkl'):
    raise FileNotFoundError("Vectorizer not found. Run 'train.py' first.")
if not os.path.exists('models/fake_news_model.pth'):
    raise FileNotFoundError("Model weights not found. Run 'train.py' first.")

vectorizer = joblib.load('models/tfidf_vectorizer.pkl')


class NewsClassifier(torch.nn.Module):
    def __init__(self, input_size):
        super(NewsClassifier, self).__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_size, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(32, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)


input_size = len(vectorizer.get_feature_names_out())
model = NewsClassifier(input_size)
model.load_state_dict(torch.load('models/fake_news_model.pth', map_location='cpu'))
model.eval()

print("✅ Model loaded successfully!\n")


# -------------------------------
# 2. Text Cleaning
# -------------------------------
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s!?]+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# -------------------------------
# 3. Rule-Based Filters
# -------------------------------
ABSURD_KEYWORDS = [
    'dances naked', 'secretly flees', 'hitler was right', 'aliens invade',
    'moon is cheese', 'earth is flat', 'birds are drones', 'trump dances naked',
    'queen elizabeth 100th birthday', 'queen elizabeth returns from the dead',
    'masks cause oxygen loss', 'vaccines cause autism', '5g causes coronavirus',
    'covid is a hoax', 'cdc cover-up', 'fda hiding cure', 'climate change is natural cycle',
    'global warming hoax', 'scientists deny climate change', 'eating bleach cures covid',
    'bill gates microchip vaccine', 'iphone 16 holographic display', 'time machine'
]

SENSITIVE_TOPICS = [
    'masks', 'vaccine', 'covid', 'cdc', 'fda', 'climate', 'global warming',
    'autism', '5g', 'oxygen loss', 'scientists say', 'cure', 'microchip', 'gates'
]


def is_absurd(headline):
    return any(keyword in headline.lower() for keyword in ABSURD_KEYWORDS)


def is_sensitive(headline):
    return any(topic in headline.lower() for topic in SENSITIVE_TOPICS)


# -------------------------------
# 4. Prediction Function
# -------------------------------
def predict_news(headline):
    if is_absurd(headline):
        print(f"🔥 Triggered rule: '{headline}' contains absurd keywords.")
        return "Fake News", 95.0

    if is_sensitive(headline):
        print("⚠️ Warning: This involves a sensitive health or science topic. Verify with trusted sources.")

    cleaned = clean_text(headline)
    try:
        vec = vectorizer.transform([cleaned]).toarray()
    except Exception as e:
        print(f"Vectorizer error: {e}")
        return "Error", 0.0

    x = torch.FloatTensor(vec)
    with torch.no_grad():
        prob = model(x).item()

    prediction = "Real News" if prob > 0.5 else "Fake News"
    confidence = (prob if prob > 0.5 else 1 - prob) * 100
    return prediction, confidence


# -------------------------------
# 5. Feedback Collection
# -------------------------------
def log_feedback(headline, pred, conf, is_correct, correction=None, reason=""):
    entry = {
        "headline": headline,
        "model_prediction": pred,
        "confidence": round(conf, 2),
        "timestamp": datetime.now().isoformat(),
        "is_correct": is_correct,
        "reason": reason or "None"
    }
    if not is_correct:
        entry["user_correction"] = "1" if correction == "real" else "0"

    filename = 'data/correct_predictions.jsonl' if is_correct else 'data/feedback.jsonl'
    os.makedirs('data', exist_ok=True)
    with open(filename, 'a', encoding='utf-8') as f:
        f.write(json.dumps(entry) + '\n')


def collect_feedback(headline, pred, conf):
    print(f"\n🔍 Our model classified this as **{pred}** with {conf:.1f}% confidence.")
    feedback = input("Do you agree? (y/n): ").strip().lower()

    if feedback in ['y', 'yes']:
        log_feedback(headline, pred, conf, is_correct=True)
        print("✅ Thank you! Your confirmation helps reinforce correct predictions.")
    elif feedback in ['n', 'no']:
        correct_label = input("What should the correct label be? (real/fake): ").strip().lower()
        if correct_label not in ['real', 'fake']:
            print("Invalid input. Feedback not saved.")
            return
        reason = input("Optional: Why do you think so? (e.g., source, fact-check): ").strip()
        log_feedback(headline, pred, conf, is_correct=False, correction=correct_label, reason=reason)
        print("✅ Thank you! Your feedback helps improve the model.")
    else:
        print("❌ No valid response. Skipping feedback.")


# -------------------------------
# 6. Interactive Loop
# -------------------------------
print("🔍 Fake News Detector (Final Version)")
print("💡 Combines ML + rules to catch misinformation")
print("⌨️  Type 'quit', 'exit', or 'q' to exit.\n")
print("-" * 70)

while True:
    headline = input("\n📰 Headline: ").strip()
    if headline.lower() in ['quit', 'exit', 'q']:
        print("👋 Goodbye! Stay informed and skeptical!")
        break
    if not headline:
        print("⚠️ Please enter a headline.")
        continue

    try:
        pred, conf = predict_news(headline)
        print(f"\n🎯 Prediction: **{pred}**")
        print(f"📊 Confidence: **{conf:.1f}%**")
        collect_feedback(headline, pred, conf)
    except Exception as e:
        print(f"❌ Prediction failed: {e}")

# -------------------------------
# 7. Retrain After Exit
# -------------------------------
print("\n🔄 Updating model with your feedback...")
os.system("python train.py")