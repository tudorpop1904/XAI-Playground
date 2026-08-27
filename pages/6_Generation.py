import streamlit as st
import requests

st.title("🎨 Step 5: Synthetic Image Generation")

st.markdown("""
Since standard Convolutional Neural Networks are highly vulnerable to modern Diffusion Models, 
let's put your detector to the test! Use this page to generate a **synthetic deepfake** using 
Hugging Face's `diffusers` library.
""")

prompt = st.text_input("Enter a prompt for the diffusion model:", value="A hyper-realistic portrait of an astronaut on Mars")

st.markdown("### Generation Settings")
col1, col2 = st.columns(2)

mode = col1.radio("Processing Mode", ["Cloud (Hugging Face API)", "Local GPU (diffusers)"])
model_id = col2.selectbox("Model", ["runwayml/stable-diffusion-v1-5", "stabilityai/stable-diffusion-2-1", "stabilityai/stable-diffusion-xl-base-1.0"])

hf_token = ""
if mode == "Cloud (Hugging Face API)":
    st.info("Cloud mode requires a Hugging Face API Token (it will not be saved).")
    hf_token = st.text_input("Hugging Face Token", type="password")

if st.button("Generate Image", type="primary"):
    if mode == "Cloud (Hugging Face API)" and not hf_token:
        st.error("Please provide a Hugging Face token for cloud generation.")
    else:
        req_mode = "cloud" if "Cloud" in mode else "local"
        
        with st.spinner(f"Generating image via {req_mode} mode... (This may take a while if downloading models locally)"):
            try:
                res = requests.post(
                    "http://localhost:8000/api/v1/generate",
                    json={
                        "prompt": prompt,
                        "mode": req_mode,
                        "hf_token": hf_token,
                        "model_id": model_id
                    }
                )
                
                if res.status_code == 200:
                    data = res.json()
                    st.success(f"Image generated in {data['time_taken']:.2f}s!")
                    st.session_state.generated_image_path = data["image_path"]
                else:
                    st.error(f"Generation failed: {res.text}")
            except Exception as e:
                st.error(f"Error connecting to backend: {e}")

if st.session_state.get("generated_image_path"):
    st.image(st.session_state.generated_image_path, caption=prompt, width=400)
    
    if st.button("Send to Detector ➡️"):
        st.session_state.test_image_path = st.session_state.generated_image_path
        st.switch_page("pages/4_Explainers.py")
