import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
from utils.question_bank import QUESTIONS
from utils.archetype_mapping import get_archetype
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="Iris Personality Archetype",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Load Assets ---
@st.cache_data
def load_assets():
    """Loads all necessary models and data. Caches for performance."""
    try:
        ocean_model = joblib.load('model/ocean_model.pkl')
        imputation_model = joblib.load('model/imputation_model.pkl')
        selected_features = joblib.load('model/feature_selector.pkl')
        return ocean_model, imputation_model, selected_features
    except FileNotFoundError:
        st.error("Model files not found. Please run `train_model.py` first.")
        return None, None, None

def load_css(file_name):
    """Injects custom CSS into the Streamlit app."""
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# --- UI Components ---
def glassmorphism_card(content, title=""):
    """Creates a styled card using HTML for a glassmorphism effect."""
    st.markdown(f"""
    <div class="glass-card">
        {f'<h2>{title}</h2>' if title else ''}
        {content}
    </div>
    """, unsafe_allow_html=True)

def create_radar_chart(scores):
    """Generates a Plotly radar chart for the OCEAN scores."""
    traits = ['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism']
    # Reorder scores to match traits order
    ordered_scores = [scores['OPN'], scores['CSN'], scores['EXT'], scores['AGR'], scores['EST']]
    
    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=ordered_scores,
        theta=traits,
        fill='toself',
        name='Your Score',
        line=dict(color='#C77DFF')
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[1, 5],
                color='white',
                gridcolor='rgba(255, 255, 255, 0.4)'
            ),
            angularaxis=dict(
                color='white'
            )
        ),
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color="white")
    )
    return fig

# --- Main Application Logic ---
def main():
    """Controls the flow of the Streamlit application."""
    load_css('style.css')
    ocean_model, imputation_model, selected_features = load_assets()

    if not all([ocean_model, imputation_model, selected_features]):
        return

    # Initialize session state
    if 'page' not in st.session_state:
        st.session_state.page = 'home'

    # Page Routing
    if st.session_state.page == 'home':
        render_home_page()
    elif st.session_state.page == 'quiz':
        render_quiz_page(selected_features, ocean_model)
    elif st.session_state.page == 'results':
        render_results_page()

def render_home_page():
    """Displays the landing page of the app."""
    st.title("🔮 Discover Your Personality Archetype")
    st.markdown("---")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        <p>Welcome to the Iris Personality Archetype predictor. This isn't just another personality test. 
        We use a machine learning model to analyze your answers and reveal a deeper truth about who you are.</p>
        <p>By answering just 10 carefully selected questions, our AI can infer your complete personality profile across the Big Five traits (OCEAN) and map you to a unique archetype.</p>
        """, unsafe_allow_html=True)
        if st.button("Begin the Journey"):
            st.session_state.page = 'quiz'
            st.experimental_rerun()
    
    with col2:
        st.markdown("") # Spacer
        
    st.markdown("---")
    st.subheader("How It Works")
    
    tech_explanation = """
    <div class="glass-card">
        <ul>
            <li><b>Dataset:</b> We trained our models on the IPIP-50 dataset from Kaggle, which contains 50 questions and responses from a large pool of participants.</li>
            <li><b>Feature Selection:</b> Instead of making you answer all 50 questions, we used an ExtraTreesClassifier to identify the 10 most impactful questions for predicting all five personality traits.</li>
            <li><b>Trait Prediction:</b> A Random Forest Regressor model predicts your OCEAN scores based on your answers to these 10 questions.</li>
            <li><b>Archetype Mapping:</b> Your unique combination of high and low scores across the five traits maps you to one of 12 distinct personality archetypes.</li>
            <li><b>Limitations:</b> This is a simplified model for educational and entertainment purposes. Personality is complex and cannot be fully captured by a short quiz.</li>
        </ul>
    </div>
    """
    st.markdown(tech_explanation, unsafe_allow_html=True)


def render_quiz_page(selected_features, ocean_model):
    """Displays the 10-question quiz."""
    st.header("Answer the Questions Honestly")
    st.markdown("Rate each statement on a scale of 1 (Disagree) to 5 (Agree).")

    answers = {}
    with st.form(key='quiz_form'):
        for i, feature in enumerate(selected_features):
            question_text = QUESTIONS.get(feature, f"Question {i+1}")
            answers[feature] = st.radio(
                question_text,
                options=[1, 2, 3, 4, 5],
                index=2,  # Default to neutral
                horizontal=True,
                key=f"q_{i}"
            )
        
        submit_button = st.form_submit_button(label='Reveal My Archetype')

    if submit_button:
        with st.spinner('Analyzing your responses...'):
            time.sleep(1) # Simulate processing
            
            # Prepare input for the model
            input_df = pd.DataFrame([answers])
            
            # Predict OCEAN scores
            ocean_scores_array = ocean_model.predict(input_df)
            ocean_scores = {
                'EXT': ocean_scores_array[0][0],
                'EST': ocean_scores_array[0][1],
                'AGR': ocean_scores_array[0][2],
                'CSN': ocean_scores_array[0][3],
                'OPN': ocean_scores_array[0][4]
            }
            
            # Get archetype
            archetype_name, archetype_desc = get_archetype(ocean_scores)

            # Store results in session state
            st.session_state.results = {
                'scores': ocean_scores,
                'archetype_name': archetype_name,
                'archetype_desc': archetype_desc
            }
            st.session_state.page = 'results'
            st.experimental_rerun()

def render_results_page():
    """Displays the final results: archetype, scores, and radar chart."""
    results = st.session_state.get('results')
    if not results:
        st.warning("No results found. Please take the quiz first.")
        st.session_state.page = 'home'
        st.experimental_rerun()
        return

    st.title("Your Personality Result")
    
    # Archetype Card
    archetype_content = f"<h3>{results['archetype_name']}</h3><p>{results['archetype_desc']}</p>"
    glassmorphism_card(archetype_content)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Your Personality Fingerprint")
        radar_chart = create_radar_chart(results['scores'])
        st.plotly_chart(radar_chart, use_container_width=True)

    with col2:
        st.subheader("Your Trait Scores")
        traits = {
            'Openness (OPN)': results['scores']['OPN'],
            'Conscientiousness (CSN)': results['scores']['CSN'],
            'Extraversion (EXT)': results['scores']['EXT'],
            'Agreeableness (AGR)': results['scores']['AGR'],
            'Neuroticism (EST)': results['scores']['EST']
        }
        for trait, score in traits.items():
            st.markdown(f"**{trait}**")
            st.progress(score / 5.0)

    if st.button("Take the Quiz Again"):
        # Clear previous results
        if 'results' in st.session_state:
            del st.session_state['results']
        st.session_state.page = 'home'
        st.experimental_rerun()

if __name__ == "__main__":
    main()
