import joblib
import re
import numpy as np
import os

MODEL = None
VECTORIZER = None

def load_artifacts():
    global MODEL, VECTORIZER
    if MODEL is None or VECTORIZER is None:
        model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'best_model.pkl')
        vec_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'vectorizer.pkl')
        try:
            MODEL = joblib.load(model_path)
            VECTORIZER = joblib.load(vec_path)
        except FileNotFoundError:
            raise Exception(f"Model or vectorizer not found at {model_path}. Please train the model first.")

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

def predict_scenario(text):
    load_artifacts()
    
    cleaned = clean_text(text)
    vec = VECTORIZER.transform([cleaned])
    
    prediction = MODEL.predict(vec)[0]
    
    # Get probabilities
    if hasattr(MODEL, "predict_proba"):
        probs = MODEL.predict_proba(vec)[0]
        confidence = np.max(probs)
    else:
        confidence = 1.0 

    explanation = get_explanation(vec, prediction)
    
    return {
        "prediction": prediction,
        "confidence": confidence,
        "explanation": explanation
    }

def get_explanation(vec, predicted_class):
    if not hasattr(MODEL, "coef_"):
        return "Explanation not available for this model type."
        
    feature_names = np.array(VECTORIZER.get_feature_names_out())
    
    class_idx = np.where(MODEL.classes_ == predicted_class)[0][0]
    
    if MODEL.coef_.ndim > 1:
        coefs = MODEL.coef_[class_idx]
    else:
        coefs = MODEL.coef_[0]
        if class_idx == 0:
            coefs = -coefs 
            
    contributions = vec.toarray()[0] * coefs
    
    top_indices = np.argsort(contributions)[-3:][::-1]
    top_words = []
    for idx in top_indices:
        if contributions[idx] > 0:
            top_words.append((feature_names[idx], contributions[idx]))
            
    if not top_words:
        return "No strongly indicative words found in the input."
        
    explanation = f"The prediction '{predicted_class}' was heavily influenced by the presence of words/phrases like:\n"
    for word, score in top_words:
        explanation += f"- '{word}' (contribution score: {score:.2f})\n"
        
    return explanation
