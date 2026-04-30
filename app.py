import streamlit as st
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from predict import predict_scenario, load_artifacts

st.set_page_config(page_title="Moral Compass Classifier", layout="centered")

st.title("🧭 Moral Compass Classifier")
st.markdown("""
This app predicts the moral alignment of a given scenario or decision.
The categories are:
- **Utilitarian**: Maximizing overall good.
- **Ethical**: Morally principled and fair.
- **Selfish**: Self-centered or harmful to others.
""")

model_path = os.path.join(os.path.dirname(__file__), 'models', 'best_model.pkl')
if not os.path.exists(model_path):
    st.error("⚠️ Model not found! Please run the training script first (`python src/train.py`).")
else:
    try:
        load_artifacts()
    except Exception as e:
        st.error(f"Error loading model: {e}")

    user_input = st.text_area("Describe a scenario or decision:", placeholder="e.g., I found a wallet on the street and kept the cash inside.")

    if st.button("Analyze"):
        if not user_input.strip():
            st.warning("Please enter a scenario.")
        else:
            with st.spinner("Analyzing moral alignment..."):
                try:
                    result = predict_scenario(user_input)
                    
                    st.subheader("Results")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric(label="Predicted Alignment", value=result["prediction"])
                    with col2:
                        st.metric(label="Confidence Score", value=f"{result['confidence']:.1%}")
                        
                    st.divider()
                    st.subheader("Explainability")
                    st.info(result["explanation"])
                    
                except Exception as e:
                    st.error(f"An error occurred during prediction: {e}")

st.divider()
st.caption("Built with Python, Scikit-Learn, and Streamlit.")
