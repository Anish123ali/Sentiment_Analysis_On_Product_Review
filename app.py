import streamlit as st
import pickle
import numpy as np
import io
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import re

# --- Streamlit Page Configuration ---
# Set the page configuration to use the wide layout
st.set_page_config(
    page_title="Text Sentiment Analyzer",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items=None
)

# Define the CSS directly inside the script using st.markdown
st.markdown("""
<style>
    /* Main container styling */
    .stApp {
        background-color: #f7f9fc;
        color: #1a1a1a;
        font-family: 'Inter', sans-serif;
    }
    /* Title styling */
    h1 {
        color: #4a90e2; /* Blue color for title */
        text-align: center;
        margin-bottom: 0.5rem;
    }
    /* Subheader/instruction styling */
    h3 {
        color: #555555;
        text-align: center;
        margin-bottom: 2rem;
    }
    /* Text area styling */
    .stTextArea label {
        font-size: 1.1rem;
        font-weight: 600;
        color: #333;
    }
    /* Result card styling */
    .result-card {
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        text-align: center;
        margin-top: 30px;
        margin-bottom: 20px;
        transition: transform 0.3s ease-in-out;
    }
    .positive {
        background-color: #e6ffed; /* Light green */
        border: 2px solid #38c172; /* Darker green border */
    }
    .negative {
        background-color: #ffe6e6; /* Light red */
        border: 2px solid #e3342f; /* Darker red border */
    }
    .neutral {
        background-color: #fff8e1; /* Light yellow */
        border: 2px solid #ffab00; /* Darker yellow border */
    }
    .result-text {
        font-size: 1.8rem; /* Larger font */
        font-weight: 700;
        margin-bottom: 10px;
        color: #1a1a1a;
    }
    .probability-text {
        font-size: 1.1rem;
        color: #666;
    }
    /* Custom button style */
    .stButton>button {
        border-radius: 8px;
        font-weight: bold;
        padding: 0.6rem 1rem;
    }
    /* Custom metric styling */
    [data-testid="stMetric"] {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }
</style>
""", unsafe_allow_html=True)

# --- Utility Functions for Model Loading ---

@st.cache_resource
def load_assets(vectorizer_path, model_path):
    """Loads the CountVectorizer and the Sentiment Model from the uploaded files."""
    try:
        # Load CountVectorizer (transformer)
        with open(vectorizer_path, 'rb') as f:
            vectorizer_bytes = f.read()
            vectorizer_data = io.BytesIO(vectorizer_bytes)
            vectorizer = pickle.load(vectorizer_data)

        # Load Sentiment Model (predictor)
        with open(model_path, 'rb') as f:
            model_bytes = f.read()
            model_data = io.BytesIO(model_bytes)
            model = pickle.load(model_data)

        return vectorizer, model
    except FileNotFoundError:
        st.error("Error: The model or vectorizer files were not found. Please ensure 'countvectorizer.pkl' and 'sentiment_model.pkl' are uploaded.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred during model loading: {e}")
        st.stop()


# --- Main Prediction Function ---

def predict_sentiment(text, vectorizer, model):
    """Vectorizes the text, predicts the sentiment, and returns full probability data."""
    
    # 1. Vectorize the input text
    # The [text] is wrapped in a list because transform expects an iterable of documents.
    text_vectorized = vectorizer.transform([text])

    # 2. Get prediction probabilities
    # Predict_proba returns probabilities for all classes (e.g., [prob_neg, prob_pos])
    probabilities = model.predict_proba(text_vectorized)[0]
    
    # Determine the class labels based on the model's classes_ attribute
    if hasattr(model, 'classes_') and len(model.classes_) == len(probabilities):
        classes = [str(c) for c in model.classes_]
    else:
        # Fallback assuming binary classification: 0 (Negative), 1 (Positive)
        classes = ['Negative', 'Positive'] 
        
    # Map the class index to a human-readable label and create a dictionary
    # Assuming standard order: class 0 is generally negative/first, class 1 is positive/second
    prob_dict = {
        'Negative': probabilities[0], 
        'Positive': probabilities[1]
    }
    
    # Determine the predicted label and its confidence
    pred_index = np.argmax(probabilities)
    confidence = probabilities[pred_index]

    if len(classes) == 2:
        # Binary Classification (0, 1) -> Negative, Positive
        label = "Positive" if pred_index == 1 else "Negative"
    elif len(classes) == 3:
        # Ternary Classification (0, 1, 2) -> Negative, Neutral, Positive (common order)
        # Note: This heavily depends on the model's training order.
        if pred_index == 2:
             label = "Positive"
        elif pred_index == 1:
             label = "Neutral"
        else:
             label = "Negative"
    else:
        # Generic case
        label = list(prob_dict.keys())[pred_index]


    return label, confidence, prob_dict


# --- Chart Generation Functions ---

def create_confidence_gauge(confidence, sentiment):
    """Creates a Plotly Gauge Chart for the prediction confidence."""
    
    # Define colors based on sentiment for a more dynamic look
    if "Positive" in sentiment:
        color = "#38c172" # Green
    elif "Negative" in sentiment:
        color = "#e3342f" # Red
    else:
        color = "#ffab00" # Yellow
        
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = confidence * 100,
        title = {'text': "Confidence Score (%)"},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': 'lightgray'},
                {'range': [50, 80], 'color': 'lightblue'},
                {'range': [80, 100], 'color': 'lightgreen'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': confidence * 100
            }}
    ))
    
    fig.update_layout(
        height=300, 
        margin=dict(l=10, r=10, t=50, b=10),
        template="plotly_white"
    )
    return fig

def create_probability_bar_chart(prob_dict):
    """Creates a Plotly Bar Chart for the probability distribution."""
    
    # Convert dictionary to DataFrame for Plotly Express
    df = pd.DataFrame(list(prob_dict.items()), columns=['Sentiment', 'Probability'])
    
    # Define colors
    sentiment_colors = {
        'Positive': '#38c172', 
        'Negative': '#e3342f', 
        'Neutral': '#ffab00'
    }
    
    # Map colors to sentiments present in the data
    color_map = {s: sentiment_colors.get(s, '#4a90e2') for s in df['Sentiment']}
    
    fig = px.bar(
        df, 
        x='Sentiment', 
        y='Probability', 
        color='Sentiment', 
        color_discrete_map=color_map,
        text=df['Probability'].apply(lambda x: f'{x:.2f}'), # Display probability on bars
        title='Probability Distribution Across Classes'
    )
    
    # Customize layout
    fig.update_layout(
        xaxis_title="", 
        yaxis_title="Probability (0 to 1)", 
        yaxis=dict(range=[0, 1.0], tickformat=".1f"),
        uniformtext_minsize=8, 
        uniformtext_mode='hide',
        height=350,
        template="plotly_white"
    )
    fig.update_traces(marker_line_width=0)
    
    return fig


# --- Streamlit UI Components ---

def main():
    """Main function to run the Streamlit app."""

    st.title("💡 AI-Powered Sentiment Analyzer")
    st.markdown("### Classify text as Positive or Negative using a trained ML model and explore the results visually.")

    # Load the assets (vectorizer and model)
    # The models must be present in the execution environment
    vectorizer, model = load_assets("countvectorizer.pkl", "sentiment_model.pkl")

    # Input area
    st.markdown("---")
    input_text = st.text_area(
        "Enter Text for Sentiment Analysis:",
        "I am so happy with the performance of this Streamlit app! It loads everything quickly and the results are easy to understand.",
        height=150
    )

    # Prediction Button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        analyze_button = st.button("Analyze Sentiment", use_container_width=True, type="primary")

    st.markdown("---")
    
    # --- Result Display ---
    result_container = st.container()

    if analyze_button and input_text:
        # Add a spinner while processing
        with st.spinner('Analyzing...'):
            try:
                # Perform prediction
                sentiment, confidence, prob_dict = predict_sentiment(input_text, vectorizer, model)

                # Determine card styling based on sentiment
                card_class = ""
                if "Positive" in sentiment:
                    card_class = "positive"
                    icon = "😊"
                elif "Negative" in sentiment:
                    card_class = "negative"
                    icon = "😞"
                else:
                    card_class = "neutral"
                    icon = "😐"

                with result_container:
                    # 1. Main Result Card
                    st.markdown("## Analysis Summary")
                    st.markdown(
                        f"""
                        <div class='result-card {card_class}'>
                            <div class='result-text'>{icon} Predicted Sentiment: {sentiment}</div>
                            <div class='probability-text'>Confidence: **{confidence:.2f}**</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    
                    # 2. Charts and Metrics
                    st.markdown("### Visual Insights")
                    
                    chart_col, gauge_col = st.columns([2, 1])
                    
                    with chart_col:
                        # Bar Chart: Probability Distribution
                        bar_fig = create_probability_bar_chart(prob_dict)
                        st.plotly_chart(bar_fig, use_container_width=True)
                        
                    with gauge_col:
                        # Gauge Chart: Confidence Score
                        gauge_fig = create_confidence_gauge(confidence, sentiment)
                        st.plotly_chart(gauge_fig, use_container_width=True)


                    # 3. Text Metrics
                    st.markdown("### Text Metrics")
                    
                    # Calculate metrics
                    word_count = len(input_text.split())
                    char_count = len(input_text)
                    sentence_count = len(re.split(r'[.!?]+', input_text)) - 1 # Simple sentence count

                    metric_col1, metric_col2, metric_col3 = st.columns(3)
                    
                    metric_col1.metric("Total Words", f"{word_count}")
                    metric_col2.metric("Total Characters", f"{char_count}")
                    metric_col3.metric("Estimated Sentences", f"{sentence_count}")
                    
                    st.markdown("<br>", unsafe_allow_html=True) # Spacer

            except Exception as e:
                st.error(f"Prediction and visualization failed. Ensure the input text and the model are compatible: {e}")
    elif analyze_button and not input_text:
        st.warning("Please enter some text to analyze.")
    elif not analyze_button and not input_text:
        # Placeholder on initial load
        st.info("Enter your text above and click 'Analyze Sentiment' to see the detailed analysis and charts.")


# Run the app
if __name__ == '__main__':
    main()