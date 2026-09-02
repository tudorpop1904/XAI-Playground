import streamlit as st
import requests
import time
from pages.config import API_BASE_URL

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

SUGGESTED_MODELS = [
    "black-forest-labs/FLUX.1-schnell",
    "black-forest-labs/FLUX.1-dev",
    "stabilityai/stable-diffusion-xl-base-1.0",
    "✏️ Enter custom model ID...",
]
model_choice = col2.selectbox("Model", SUGGESTED_MODELS)
if model_choice == "✏️ Enter custom model ID...":
    model_id = col2.text_input("Custom HF model ID", placeholder="org/model-name")
else:
    model_id = model_choice

hf_token = ""
if mode == "Cloud (Hugging Face API)":
    st.info("Cloud mode requires a Hugging Face API Token (it will not be saved).")
    hf_token = st.text_input("Hugging Face Token", type="password")

if st.button("Generate Image", type="primary"):
    if mode == "Cloud (Hugging Face API)" and not hf_token:
        st.error("Please provide a Hugging Face token for cloud generation.")
    else:
        req_mode = "cloud" if "Cloud" in mode else "local"
        
        try:
            res = requests.post(
                f"{API_BASE_URL}/api/v1/generate",
                json={
                    "prompt": prompt,
                    "mode": req_mode,
                    "hf_token": hf_token,
                    "model_id": model_id
                }
            ).json()

            if req_mode == "local":
                # Local mode is synchronous — result is immediate
                if res.get("status") == "success":
                    st.success(f"Image generated in {res['time_taken']:.2f}s!")
                    st.session_state.generated_image_path = res["image_path"]
                else:
                    st.error(f"Generation failed: {res}")

            elif "job_id" in res:
                # Cloud mode is async — poll for result
                job_id = res["job_id"]
                st.info(f"📬 Cloud generation job queued! ID: `{job_id}`")

                # ─── Polling Loop ─────────────────────────────────────
                progress_bar = st.progress(0, text="Connecting to Hugging Face servers...")
                MAX_WAIT_SECONDS = 300
                poll_interval = 2
                elapsed = 0
                fake_progress = 5

                while elapsed < MAX_WAIT_SECONDS:
                    time.sleep(poll_interval)
                    elapsed += poll_interval

                    try:
                        job = requests.get(f"{API_BASE_URL}/api/v1/jobs/{job_id}").json()
                    except Exception as poll_err:
                        st.warning(f"Polling error: {poll_err}")
                        continue

                    job_status = job.get("status", "unknown")

                    if job_status in ("queued", "running"):
                        fake_progress = min(fake_progress + 4, 90)
                        label = "⏳ Queued..." if job_status == "queued" else "🎨 Generating image..."
                        progress_bar.progress(fake_progress, text=label)
                    elif job_status == "done":
                        progress_bar.progress(100, text="✅ Generation complete!")
                        result = job.get("result", {})
                        st.success(f"Image generated in {result.get('time_taken', 0):.2f}s!")
                        st.session_state.generated_image_path = result.get("image_path")
                        break
                    elif job_status == "error":
                        progress_bar.progress(100, text="❌ Failed")
                        st.error(f"Generation failed: {job.get('error', 'Unknown error')}")
                        break
                else:
                    st.error(f"⏱️ Timeout — generation took more than {MAX_WAIT_SECONDS}s.")
            else:
                st.error(f"Unexpected response: {res}")

        except Exception as e:
            st.error(f"Error connecting to backend: {e}")

if st.session_state.get("generated_image_path"):
    st.image(st.session_state.generated_image_path, caption=prompt, width=400)
    
    if st.button("Send to Detector ➡️"):
        st.session_state.test_image_path = st.session_state.generated_image_path
        st.switch_page("pages/4_Explainers.py")
