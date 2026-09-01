import streamlit as st

st.set_page_config(
    page_title="Deepfake Detection Playground",
    page_icon="🕵️",
    layout="wide"
)

st.title("🕵️ Deepfake Detection Forensic Playground")
st.markdown("Welcome to the Zero-Touch Deepfake Analysis Wizard!")

# Since we are using Streamlit's native `pages/` directory (multipages),
# this file simply serves as the landing container.
# If you want to use the new `st.navigation` (Streamlit 1.36+), we could configure it here,
# but the standard `pages/` directory approach is universally supported and works out-of-the-box.

st.sidebar.success("Select a page above to start the pipeline.")

# Initialize session state variables if they don't exist
if "dataset_path" not in st.session_state:
    st.session_state.dataset_path = None
if "dataset_slug" not in st.session_state:
    st.session_state.dataset_slug = None
if "model_name" not in st.session_state:
    st.session_state.model_name = "CNN_1"
if "model_trained" not in st.session_state:
    st.session_state.model_trained = False
if "test_image_path" not in st.session_state:
    st.session_state.test_image_path = None
if "xai_results" not in st.session_state:
    st.session_state.xai_results = []
if "detection_result" not in st.session_state:
    st.session_state.detection_result = None

st.switch_page("pages/1_Home.py")