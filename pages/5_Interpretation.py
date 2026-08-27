import streamlit as st
import requests
import json
from pages.config import API_BASE_URL

st.title("🤖 Step 4: LLM Interpretation")

if not st.session_state.get("xai_results") or not st.session_state.get("detection_result"):
    st.warning("Please generate XAI explanations first.")
    st.stop()

st.markdown("""
Using **Ollama**, we will now translate the mathematical XAI metrics into a human-readable forensic report.
""")

st.sidebar.markdown("### LLM Configuration")
available_models = ["llama3.1:8b-instruct-q4_K_M"]
try:
    res = requests.get(f"{API_BASE_URL}/api/v1/interpret/status")
    if res.status_code == 200:
        data = res.json()
        if data.get("status") and "models" in data:
            available_models = data["models"]
except Exception:
    pass

selected_llm = st.sidebar.selectbox("Select Ollama Model", available_models)

# Build the payload from session state. We'll pick the first explainer for simplicity,
# or we can pass all of them if the schema allows. Our schema only allows one explainer currently.
# So we'll iterate through them or just pick the best one. Let's just do the first one for the demo.

st.info("Generating report based on the first selected XAI method...")
first_result = st.session_state.xai_results[0]
payload = {
    "detector_name": st.session_state.model_name,
    "explainer_name": first_result["explainer"],
    "ai_deepfake": st.session_state.detection_result["label"] == "Deepfake",
    "confidence": st.session_state.detection_result["confidence"],
    "metrics": first_result["metrics"],
    "llm_model": selected_llm
}

if st.button("Generate Forensic Report", type="primary"):
    with st.spinner("Connecting to local Llama 3.1..."):
        try:
            # We use requests.post with stream=True to handle the StreamingResponse from FastAPI
            response = requests.post(
                f"{API_BASE_URL}/api/v1/interpret", 
                json=payload,
                stream=True
            )
            
            if response.status_code == 200:
                st.markdown("### Forensic Analysis Report")
                
                # We need a generator for st.write_stream
                def generate():
                    for chunk in response.iter_content(chunk_size=1024):
                        if chunk:
                            yield chunk.decode('utf-8')
                            
                st.write_stream(generate())
            else:
                st.error(f"Error: {response.text}")
                
        except Exception as e:
            st.error(f"Failed to connect to LLM endpoint: {e}")
            st.info("Make sure 'ollama serve' is running in the background!")

if st.button("🏠 Return Home"):
    st.switch_page("pages/1_Home.py")
