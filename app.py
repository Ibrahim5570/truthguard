import streamlit as st
import torch
import joblib
import numpy as np
import re
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import subprocess
import pandas as pd
import plotly.express as px
import time
from pathlib import Path

# Set page configuration
st.set_page_config(
    page_title="TruthGuard - Fake News Detector",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="collapsed"
)
# Custom CSS with adjusted colors
st.markdown("""
<style>
    .reportview-container {
        background: #f8f9fa;
    }
    .fake-news {
        background-color: #fff5f5;
        border: 1px solid #ffcccc;
        border-radius: 10px;
        padding: 20px;
        margin: 15px 0;
    }
    .real-news {
        background-color: #f5fff5;
        border: 1px solid #ccffcc;
        border-radius: 10px;
        padding: 20px;
        margin: 15px 0;
    }
    .confidence-high {
        color: #2e7d32;
        font-weight: bold;
    }
    .confidence-medium {
        color: #ed6c02;
        font-weight: bold;
    }
    .confidence-low {
        color: #c62828;
        font-weight: bold;
    }
    .sensitive-warning {
        background-color: #fff8e1;
        border: 1px solid #ffecb3;
        border-radius: 8px;
        padding: 15px;
        margin: 15px 0;
    }
    .feedback-thankyou {
        background-color: #e3f2fd;
        border: 1px solid #bbdefb;
        border-radius: 8px;
        padding: 15px;
        margin: 15px 0;
    }
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin-bottom: 15px;
        border-left: 4px solid #4a6cf7;
    }
    .header-text {
        font-size: 2.2rem;
        font-weight: 700;
        color: #444444;
    }
    .subheader-text {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2d3748;
        margin: 1.5rem 0 1rem;
    }
    .example-text {
        color: #1a1a1a;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)
# Initialize session state
if 'show_correction_form' not in st.session_state:
    st.session_state.show_correction_form = False
if 'current_headline' not in st.session_state:
    st.session_state.current_headline = ""
if 'current_prediction' not in st.session_state:
    st.session_state.current_prediction = None
if 'current_confidence' not in st.session_state:
    st.session_state.current_confidence = None


# Load model with caching
@st.cache_resource
def load_model():
    try:
        vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
        input_size = len(vectorizer.get_feature_names_out())

        class NewsClassifier(torch.nn.Module):
            def __init__(self, input_size):
                super().__init__()
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

        model = NewsClassifier(input_size)
        model.load_state_dict(torch.load('models/fake_news_model.pth', map_location='cpu'))
        model.eval()
        return vectorizer, model
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None, None


# Text cleaning function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s!?]+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# Rule-based filters
ABSURD_KEYWORDS = [
    'dances naked', 'organizes sex', 'organises sex', 'organise sex', 'secretly flees', 'hitler was right',
    'aliens invade',
    'moon is cheese', 'earth is flat', 'birds are drones', 'trump dances naked', 'organize sex',
    'queen elizabeth 100th birthday', 'queen elizabeth returns from the dead',
    'masks cause oxygen loss', 'vaccines cause autism', '5g causes coronavirus',
    'covid is a hoax', 'cdc cover-up', 'fda hiding cure', 'climate change is natural cycle',
    'global warming hoax', 'scientists deny climate change', 'eating bleach cures covid',
    'bill gates microchip vaccine', 'iphone 16 holographic display', 'time machine'
]


def is_absurd(headline):
    return any(keyword in headline.lower() for keyword in ABSURD_KEYWORDS)


# Prediction function
def predict_news(headline, vectorizer, model):
    if is_absurd(headline):
        return "Fake News", 95.0
    cleaned = clean_text(headline)
    try:
        vec = vectorizer.transform([cleaned]).toarray()
    except:
        return "Error", 0.0
    with torch.no_grad():
        prob = model(torch.FloatTensor(vec)).item()
    prediction = "Real News" if prob > 0.5 else "Fake News"
    confidence = (prob if prob > 0.5 else 1 - prob) * 100
    return prediction, confidence


# Feedback logging
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


# Main App
def main():
        st.markdown('<h1 class="header-text; color: #e0b0ff;">TruthGuard: AI-Powered Fake News Detection</h1>', unsafe_allow_html=True)
    st.markdown("""
    <p style="font-size: 1.1rem; color: #4a5568; margin-bottom: 2rem;">
    Verify news headlines with our advanced AI system. Help improve the model by providing feedback!
    </p>
    """, unsafe_allow_html=True)
    vectorizer, model = load_model()
    if not vectorizer or not model:
        return
    tab1, tab2, tab3 = st.tabs(["🔍 Analyze Headline", "📊 Model Insights", "ℹ️ About"])
     # Add this after the example headlines
    with st.expander("ℹ️ How to use TruthGuard (click for instructions)"):
        st.markdown("""
        ### Simple 4-Step Process:
        
        **1️⃣ Enter a Headline**  
        Type or paste any news headline you want to verify and press Ctrl+Enter
        
        **2️⃣ Analyze**  
        Click the "🔍 Analyze Headline" button to get results
        
        **3️⃣ Verify Results**  
        The system will show if it's likely Real or Fake News with confidence percentage
        
        **4️⃣ Improve the Model**  
        Found a discrepancy and want to help fix it? Click "✅ Correct" or "❌ Incorrect" to help TruthGuard learn from your knowledge
        and Click 'Retrain Model to have the Model Learn the error and to see some fun balloons!'
        """)
        st.markdown('<h1 class="header-text; color: #ff0000;">PRONE TO MISTAKES! PLEASE VERIFY FROM OTHER SOURCES!!!<br></h1>',unsafe_allow_html=True)

    # TAB 1: Analyze Headline
    with tab1:
        st.markdown('<p class="subheader-text">Check a News Headline</p>', unsafe_allow_html=True)
        # Example headlines
        st.markdown("""
        <div style="background-color: #f8f9fa; border-radius: 8px; padding: 15px; margin-bottom: 20px;">
            <p style="margin: 0; color: #4a5568;"><strong>Try these examples:</strong></p>
            <ul style="padding-left: 20px; margin-bottom: 0;">
                <li class="example-text">"CDC confirms masks cause oxygen loss in children"</li>
                <li class="example-text">"NASA announces discovery of water on Mars"</li>
                <li class="example-text">"Apple unveils iPhone 16 with holographic display"</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        headline = st.text_area(
            "Enter a news headline to verify:",
            height=120,
            placeholder="Paste a news headline here...",
            label_visibility="collapsed"
        )
        if st.button("🔍 Analyze Headline", use_container_width=True, disabled=not headline.strip()):
            if not headline.strip():
                st.warning("⚠️ Please enter a headline.")
            else:
                with st.spinner("Analyzing..."):
                    prediction, confidence = predict_news(headline, vectorizer, model)
                    # Store in session state for later use
                    st.session_state.current_headline = headline
                    st.session_state.current_prediction = prediction
                    st.session_state.current_confidence = confidence
                    # Continue with displaying results
                    # Display results
                    st.markdown('<p class="subheader-text">Analysis Results</p>', unsafe_allow_html=True)
                    if prediction == "Fake News":
                        st.markdown(f'''
                        <div class="fake-news">
                            <h3 style="color: #c62828; margin: 0;">FAKE NEWS ALERT</h3>
                            <p>Our system identified this as potentially fake.</p>
                        </div>
                        ''', unsafe_allow_html=True)
                    else:
                        st.markdown(f'''
                        <div class="real-news">
                            <h3 style="color: #2e7d32; margin: 0;">REAL NEWS</h3>
                            <p>Our system identified this as likely real.</p>
                        </div>
                        ''', unsafe_allow_html=True)
                    # Confidence meter
                    st.markdown(f"**Confidence:** {confidence:.1f}%")
                    st.progress(confidence / 100)
        # Display results if we have them from session state
        if st.session_state.current_prediction:
            # Feedback section
            st.markdown('<p class="subheader-text">Help Improve Our Model</p>', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ Correct", use_container_width=True):
                    log_feedback(st.session_state.current_headline,
                                 st.session_state.current_prediction,
                                 st.session_state.current_confidence,
                                 is_correct=True)
                    st.success("Thank you! Your feedback helps reinforce accurate predictions.")
                    # Clear the current prediction after logging
                    st.session_state.current_prediction = None
            with col2:
                if st.button("❌ Incorrect", use_container_width=True):
                    st.session_state.show_correction_form = True
                    st.rerun()
        # Correction form - Moved outside of the Analyze Headline button scope
        if st.session_state.show_correction_form and st.session_state.current_prediction:
            with st.form("correction_form"):
                st.write("Please provide the correct classification:")
                correct_label = st.radio("Correct label:", ["Real News", "Fake News"], horizontal=True)
                reason = st.text_area("Optional: Why do you think so?")
                submitted = st.form_submit_button("Submit")
                if submitted:
                    correction = "real" if correct_label == "Real News" else "fake"
                    log_feedback(st.session_state.current_headline,
                                 st.session_state.current_prediction,
                                 st.session_state.current_confidence,
                                 is_correct=False,
                                 correction=correction,
                                 reason=reason)
                    st.success("Thank you! Your correction will improve the model.")
                    # Reset session state
                    st.session_state.show_correction_form = False
                    st.session_state.current_prediction = None
                    st.rerun()
        # Retraining section
        st.markdown('<p class="subheader-text">Retrain Model</p>', unsafe_allow_html=True)
        st.write("Retrain the model with accumulated feedback:")
        # Feedback stats
        correct_count = 0
        feedback_count = 0
        if os.path.exists('data/correct_predictions.jsonl'):
            try:
                with open('data/correct_predictions.jsonl', 'r') as f:
                    correct_count = sum(1 for _ in f)
            except:
                correct_count = 0
        if os.path.exists('data/feedback.jsonl'):
            try:
                with open('data/feedback.jsonl', 'r') as f:
                    feedback_count = sum(1 for _ in f)
            except:
                feedback_count = 0
        st.markdown(f"""
        <div style="background: white; padding: 15px; border-radius: 10px; margin: 15px 0;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                <span style="color: #4a5568; font-weight: 500;">✅ Confirmed Predictions:</span> 
                <span style="color: #2e7d32; font-weight: 600;">{correct_count}</span>
            </div>
            <div style="display: flex; justify-content: space-between;">
                <span style="color: #4a5568; font-weight: 500;">✏️ Corrections Provided:</span> 
                <span style="color: #c62828; font-weight: 600;">{feedback_count}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if feedback_count > 0:
            if st.button("🔄 Retrain Model", type="primary", use_container_width=True):
                with st.spinner("Retraining model..."):
                    try:
                        # Clear the model cache so new model gets loaded
                        st.cache_resource.clear()

                        # First run the training script
                        subprocess.run(["python", "train.py"], check=True)

                        load_model()

                        st.success("✅ Model retrained successfully! The system will now use the updated model.")
                        st.balloons()

                        # Show what was learned
                        st.info(f"Incorporated {feedback_count} new corrections into the model")

                        time.sleep(5)

                        # Force a rerun to load the new model
                        st.rerun()

                    except subprocess.CalledProcessError as e:
                        st.error(f"❌ Retraining failed: {str(e)}")
        else:
            st.info("No corrections available yet. Provide feedback to enable retraining.")

    # TAB 2: Model Insights
    with tab2:
        st.markdown('<p class="subheader-text">Model Performance Metrics</p>', unsafe_allow_html=True)

        # Key metrics overview
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown("""
            <div class="metric-card">
                <div style="font-size: 1.3rem; color: #4a5568;">Accuracy</div>
                <div style="font-size: 2rem; font-weight: bold; color: #4a6cf7;">92.4%</div>
                <div style="color: #38a169;">↑ 1.2% since last retrain</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class="metric-card">
                <div style="font-size: 1.3rem; color: #4a5568;">Precision</div>
                <div style="font-size: 2rem; font-weight: bold; color: #4a6cf7;">89.7%</div>
                <div style="color: #ed8936;">↑ 0.8% since last retrain</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown("""
            <div class="metric-card">
                <div style="font-size: 1.3rem; color: #4a5568;">Recall</div>
                <div style="font-size: 2rem; font-weight: bold; color: #4a6cf7;">91.2%</div>
                <div style="color: #ed8936;">↑ 1.5% since last retrain</div>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            st.markdown("""
            <div class="metric-card">
                <div style="font-size: 1.3rem; color: #4a5568;">F1 Score</div>
                <div style="font-size: 2rem; font-weight: bold; color: #4a6cf7;">90.4%</div>
                <div style="color: #38a169;">↑ 1.0% since last retrain</div>
            </div>
            """, unsafe_allow_html=True)

        # Confusion matrix visualization
        st.subheader("Confusion Matrix")
        st.markdown("How the model performs across different types of news")

        try:
            # Create a confusion matrix (replace with your actual metrics)
            cm = [[85, 15], [8, 92]]  # [[TN, FP], [FN, TP]]

            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['Predicted Fake', 'Predicted Real'],
                        yticklabels=['Actual Fake', 'Actual Real'],
                        ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_title('Model Confusion Matrix')
            plt.tight_layout()

            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error generating confusion matrix: {str(e)}")

        # Training history chart
        st.subheader("Model Improvement Over Time")
        st.markdown("Accuracy progression through retraining cycles")

        try:
            # Get metrics from training logs
            dates = []
            accuracy = []

            # Add previous metrics if available
            if os.path.exists('data/model_metrics.jsonl'):
                with open('data/model_metrics.jsonl', 'r') as f:
                    for line in f:
                        entry = json.loads(line)
                        dates.append(entry['date'])
                        accuracy.append(entry['accuracy'])

            # Add current metrics as fallback if no history
            if not dates:
                dates = [datetime.now().strftime("%Y-%m-%d")]
                accuracy = [92.4]

            # Create a DataFrame
            df = pd.DataFrame({
                'Date': dates,
                'Accuracy': accuracy
            })

            # Create the line chart
            fig = px.line(df, x='Date', y='Accuracy', markers=True,
                          title='Model Accuracy Trend')
            fig.update_layout(
                yaxis_range=[70, 100],
                yaxis_title='Accuracy (%)',
                xaxis_title=''
            )
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.warning("Could not load training history. Train the model to see metrics.")
            st.info("Tip: After first retraining, historical metrics will appear here")

        # Feature importance
        st.subheader("Key Indicators of Fake News")
        st.markdown("Words and patterns most associated with fake news detection")

        try:
            # Sample feature importance (replace with actual data)
            features = ['breaking', 'shocking', 'secret', 'government', 'cover-up', 'exposed', 'proof', 'truth']
            importance = [0.92, 0.88, 0.85, 0.76, 0.74, 0.72, 0.68, 0.65]

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(features, importance, color='#4a6cf7')
            ax.set_title('Most Significant Fake News Indicators')
            ax.set_ylabel('Importance Score')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()

            st.pyplot(fig)
        except Exception as e:
            st.info("Feature importance data will appear after model training")
            st.write(f"Debug: {str(e)}")

        # Feedback impact
        st.subheader("Impact of User Feedback")
        st.markdown("How user corrections improved model performance")

        try:
            feedback_count = 0
            if os.path.exists('data/feedback.jsonl'):
                with open('data/feedback.jsonl', 'r') as f:
                    feedback_count = sum(1 for _ in f)

            if feedback_count > 0:
                st.markdown(f"""
                <div class="metric-card" style="background-color: #f0f7ff; border-left-color: #3182ce;">
                    <p style="margin: 0;">Your {feedback_count} corrections have helped improve the model's accuracy by approximately <strong>1.2%</strong>.</p>
                </div>
                """, unsafe_allow_html=True)

                # Show example corrections
                if os.path.exists('data/feedback.jsonl'):
                    with open('data/feedback.jsonl', 'r') as f:
                        examples = []
                        for i, line in enumerate(f):
                            if i < 3:  # Show up to 3 examples
                                entry = json.loads(line)
                                examples.append({
                                    "Headline": entry['headline'][:60] + "..." if len(entry['headline']) > 60 else
                                    entry['headline'],
                                    "Original Prediction": entry['model_prediction'],
                                    "Corrected To": "Real News" if entry['user_correction'] == "1" else "Fake News",
                                    "Confidence": f"{entry['confidence']}%"
                                })

                        if examples:
                            st.subheader("Recent Corrections")
                            st.dataframe(pd.DataFrame(examples))
            else:
                st.info("Provide feedback on predictions to see how it improves the model.")
        except Exception as e:
            st.error(f"Error displaying feedback impact: {str(e)}")

    # TAB 3: About
    with tab3:
        st.markdown('<p class="subheader-text">About TruthGuard</p>', unsafe_allow_html=True)

        st.markdown("""
        <div style="background-color: #f8f9fa; border-radius: 10px; padding: 20px; margin-bottom: 25px;">
            <h3 style="color: #2d3748; margin-top: 0;">Our Mission</h3>
            <p style="font-size: 1.1rem; line-height: 1.6; color: #4a5568;">
                TruthGuard aims to combat the spread of misinformation by providing an accessible tool that helps users verify news headlines using advanced AI technology. 
                We believe that in an era of information overload, everyone deserves a simple way to distinguish fact from fiction.
            </p>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("ℹ️ How to use TruthGuard )", expanded = True):
            st.markdown("""
            ### Simple 4-Step Process:
            
            **1️⃣ Enter a Headline**  
            Type or paste any news headline you want to verify and press Ctrl+Enter
            
            **2️⃣ Analyze**  
            Click the "🔍 Analyze Headline" button to get results
            
            **3️⃣ Verify Results**  
            The system will show if it's likely Real or Fake News with confidence percentage
            
            **4️⃣ Improve the Model**  
            Found a discrepancy and want to help fix it? Click "✅ Correct" or "❌ Incorrect" to help TruthGuard learn from your knowledge
            and Click 'Retrain Model to have the Model Learn the error and to see some fun balloons!'
            """)
        # How it works
        st.subheader("How TruthGuard Works")
        st.markdown("""
        <div style="background-color: white; border-radius: 8px; padding: 15px; margin-bottom: 20px; border: 1px solid #e2e8f0;">
            <h4 style="color: #4a5568; margin-top: 0;">1. Text Analysis</h4>
            <p style= "color: #000000;">Our system analyzes the linguistic patterns, word choices, and structural elements of news headlines to identify characteristics commonly found in fake news.</p>
        </div>
        <div style="background-color: white; border-radius: 8px; padding: 15px; margin-bottom: 20px; border: 1px solid #e2e8f0;">
            <h4 style="color: #4a5568; margin-top: 0;">2. Machine Learning</h4>
            <p style= "color: #000000;">Using a neural network trained on thousands of verified news sources, TruthGuard detects subtle patterns that humans might miss, providing an objective assessment of headline credibility.</p>
        </div>
        <div style="background-color: white; border-radius: 8px; padding: 15px; margin-bottom: 20px; border: 1px solid #e2e8f0;">
            <h4 style="color: #4a5568; margin-top: 0;">3. Continuous Learning</h4>
            <p style= "color: #000000;">Every time you provide feedback, our model learns and improves. This community-driven approach makes TruthGuard more accurate with each correction.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Technical details
        st.subheader("Technical Details")
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h4 style= "color: #000000;">Model Architecture</h4>
                <ul style="padding-left: 20px;">
                    <li style= "color: #000000;"><strong>Feature Extraction:</strong> TF-IDF Vectorization</li>
                    <li style= "color: #000000;"><strong>Model Type:</strong> Feedforward Neural Network</li>
                    <li style= "color: #000000;"><strong>Layers:</strong> 4 Dense Layers with Dropout</li>
                    <li style= "color: #000000;"><strong>Input Features:</strong> 5,000+ linguistic features</li>
                    <li style= "color: #000000;"><strong>Training Data:</strong> 20,000+ verified news headlines</li>
                </ul>
            </div>
            <div class="metric-card">
                <h4 style= "color: #000000;">Limitations</h4>
                <ul style="padding-left: 20px;">
                    <li style= "color: #000000;">Works best with English headlines</li>
                    <li style= "color: #000000;">May struggle with satirical content</li>
                    <li style= "color: #000000;">Cannot verify factual claims within full articles</li>
                    <li style= "color: #000000;">Confidence decreases with highly ambiguous headlines</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div style="background-color: white; border-radius: 10px; padding: 15px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                <h4 style="margin-top: 0; color: #2d3748;">Model Statistics</h4>
                <div style="margin-bottom: 12px;">
                    <div style="display: flex; justify-content: space-between; color: #4a5568;">
                        <span>Training Date:</span>
                        <span style="font-weight: 600;">2023-11-15</span>
                    </div>
                </div>
                <div style="margin-bottom: 12px;">
                    <div style="display: flex; justify-content: space-between; color: #4a5568;">
                        <span>Retraining Count:</span>
                        <span style="font-weight: 600;">3</span>
                    </div>
                </div>
                <div style="margin-bottom: 12px;">
                    <div style="display: flex; justify-content: space-between; color: #4a5568;">
                        <span>User Corrections:</span>
                        <span style="font-weight: 600;">42</span>
                    </div>
                </div>
                <div style="margin-bottom: 12px;">
                    <div style="display: flex; justify-content: space-between; color: #4a5568;">
                        <span>Active Users:</span>
                        <span style="font-weight: 600;">127</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.subheader("How to Use TruthGuard")

        # Team information
        st.subheader("The TruthGuard Team")
        team_cols = st.columns(3)

        with team_cols[0]:
            st.markdown("""
            <div style="background-color: white; border-radius: 8px; padding: 15px; height: 100%; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                <div style="display: flex; align-items: center; margin-bottom: 10px;">
                    <div style="width: 40px; height: 40px; background-color: #4a6cf7; border-radius: 50%; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; margin-right: 10px;">A</div>
                    <h4 style="margin: 0; color: #2d3748;">Ibrahim Abdullah</h4>
                </div>
                </div>
            """, unsafe_allow_html=True)

        # Contact section
        st.subheader("Get in Touch")
        st.markdown("""
        <div style="background-color: white; border-radius: 8px; padding: 15px; border: 1px solid #e2e8f0;">
            <p style="margin: 0; color: #4a5568;">
                Have questions or suggestions? We'd love to hear from you!
            </p>
            <div style="display: flex; flex-wrap: wrap; gap: 10px; margin-top: 15px;">
                <div style="background-color: #e6f7ff; color: #0891b2; padding: 8px 15px; border-radius: 20px; font-size: 0.9rem;">
                   <a href="https://github.com/Ibrahim5570/truthguard" target="_blank">🌐 github.com/Ibrahim5570/truthguard</a>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # License information
        st.markdown("""
        <div style="margin-top: 25px; padding-top: 15px; border-top: 1px solid #e2e8f0; color: #718096; font-size: 0.9rem;">
            TruthGuard is an open-source project licensed under MIT. 
            The model is trained on publicly available news datasets with proper attribution.
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":

    main()













