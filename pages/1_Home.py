import streamlit as st

st.title("🏠 Home")

st.markdown("""
This is the Deepfake Detection Forensic Playground.
It is designed to evaluate, test, and explain deepfake detection models using Explainable AI (XAI).

### Pipeline Flow:
1. **Datasets**: Pick and sample a Kaggle dataset (with optional SuperResolution).
2. **Models**: Train a CNN, ViT, or KNN on your sampled dataset and watch a Manim animation of the architecture.
3. **Explainers**: Generate visual heatmaps showing *why* the model made its decision, computing Faithfulness and Sparsity.
4. **Interpretation**: Stream a real-time natural language forensic report from a local Llama 3.1 LLM analyzing your specific results!
""")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🚀 Start Pipeline", use_container_width=True, type="primary"):
        st.switch_page("pages/2_Datasets.py")

with col2:
    if st.button("📚 Documentation", use_container_width=True):
        st.info("API Documentation is available at http://localhost:8000/docs")

with col3:
    if st.button("📋 Summary", use_container_width=True):
        st.info("This project fuses classical DL with XAI to bring transparency to deepfake detection.")
