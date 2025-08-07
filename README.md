# Personality Prediction Web Application

## Overview
This project is a machine learning-powered web application for predicting personality traits based on questionnaire responses. It leverages the IPIP-50 dataset and advanced feature selection and regression techniques to model the Big Five (OCEAN) personality traits.

## Features
- **Data Cleaning & Preprocessing:** Robust handling of missing values and data types.
- **Feature Selection:** Uses mutual information to select the most informative questions.
- **Model Training:** Trains Random Forest models to predict OCEAN traits and impute missing questionnaire responses.
- **Modular Codebase:** Organized into utility modules for easy maintenance and extension.
- **Web Integration Ready:** Designed for seamless integration with web frontends.

## Project Structure
```
├── app.py                # Main application (Flask or Streamlit)
├── train_model.py        # Model training and feature selection
├── requirements.txt      # Python dependencies
├── style.css             # Custom styles for the web app
├── codebook.txt          # Dataset/codebook reference
├── utils/                # Utility modules (question bank, mappings)
├── model/                # (Ignored) Trained model files
├── data/                 # (Ignored) Dataset files
```

## How to Use
1. **Clone the repository:**
   ```sh
   git clone https://github.com/sharmaram25/Personality-Prediction.git
   cd Personality-Prediction
   ```
2. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
3. **Add your dataset:**
   - Download the IPIP-50 dataset from the official [IPIP website](https://ipip.ori.org/), or use a reputable open-source version such as [Kaggle: IPIP-50 Big Five Factor Markers](https://www.kaggle.com/datasets/volpatto/ipip-50-big-five-factor-markers).
   - Place your `ipip-50-dataset.csv` in the `data/` directory (not included in repo).
4. **Train the models:**
   ```sh
   python train_model.py
   ```
5. **Run the application:**
   ```sh
   python app.py
   ```

## Notes
- **Large files** (models and datasets) are excluded from the repository for performance and licensing reasons. You must provide your own data and train models locally.
- The application is designed for educational and research purposes.

## Credits
**Created and developed solely by Ram Sharma.**

---
*For questions, suggestions, or collaborations, please contact Ram Sharma via GitHub.*
