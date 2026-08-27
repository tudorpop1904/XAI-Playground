import streamlit as st
import requests
from pages.config import API_BASE_URL

st.title("📁 Step 1: Datasets")

st.markdown("""
Select a dataset to use for training the detector. 
The system will use the entire dataset for training.
""")

# Fetch datasets from API
try:
    datasets = requests.get(f"{API_BASE_URL}/api/v1/datasets").json()
except Exception as e:
    st.error(f"Cannot connect to backend: {e}")
    datasets = {}

if datasets:
    dataset_name = st.selectbox("Choose a Kaggle Dataset", list(datasets.keys()))
    dataset_slug = datasets[dataset_name]
    
    st.write(f"**Slug:** `{dataset_slug}`")
    
    enhance = st.toggle("Apply SuperResolution (Warning: Adds processing time)")
    
    if st.button("Download & Prepare Dataset", type="primary"):
        with st.spinner("Downloading and preparing dataset via kagglehub..."):
            try:
                res = requests.post(f"{API_BASE_URL}/api/v1/datasets/prepare?slug={dataset_slug}&enhance={enhance}").json()
                if res.get("status") == "success":
                    st.success(f"Prepared {res['reals']} real and {res['fakes']} fake images at `{res['path']}`!")
                    st.session_state.dataset_slug = dataset_slug
                    st.session_state.dataset_path = res["path"]
                else:
                    st.error("Failed to prepare dataset.")
            except Exception as e:
                st.error(f"Error: {e}")

if st.session_state.get("dataset_path"):
    st.success("✅ Dataset is ready for training.")
    if st.button("Next: Train Model ➡️"):
        st.switch_page("pages/3_Models.py")
