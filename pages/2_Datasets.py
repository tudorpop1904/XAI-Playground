import streamlit as st
import requests
import time
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
        try:
            res = requests.post(
                f"{API_BASE_URL}/api/v1/datasets/prepare?slug={dataset_slug}&enhance={enhance}"
            ).json()

            if "job_id" not in res:
                st.error(f"Unexpected response: {res}")
            else:
                job_id = res["job_id"]
                st.info(f"📬 Job queued! ID: `{job_id}` — Downloading dataset in background...")

                # ─── Polling Loop ─────────────────────────────────────────────
                progress_bar = st.progress(0, text="Waiting for worker to pick up job...")
                status_placeholder = st.empty()
                MAX_WAIT_SECONDS = 600
                poll_interval = 2
                elapsed = 0
                fake_progress = 0

                while elapsed < MAX_WAIT_SECONDS:
                    time.sleep(poll_interval)
                    elapsed += poll_interval

                    try:
                        job = requests.get(f"{API_BASE_URL}/api/v1/jobs/{job_id}").json()
                    except Exception as poll_err:
                        status_placeholder.warning(f"Polling error: {poll_err}")
                        continue

                    job_status = job.get("status", "unknown")

                    if job_status == "queued":
                        fake_progress = min(fake_progress + 2, 15)
                        progress_bar.progress(fake_progress, text="⏳ Queued — waiting for worker...")
                    elif job_status == "running":
                        fake_progress = min(fake_progress + 5, 85)
                        progress_bar.progress(fake_progress, text="⚙️ Worker is downloading & indexing dataset...")
                    elif job_status == "done":
                        progress_bar.progress(100, text="✅ Done!")
                        result = job.get("result", {})
                        st.success(
                            f"Prepared **{result.get('reals', '?')} real** and "
                            f"**{result.get('fakes', '?')} fake** images at `{result.get('path', '?')}`!"
                        )
                        st.session_state.dataset_slug = dataset_slug
                        st.session_state.dataset_path = result.get("path")
                        break
                    elif job_status == "error":
                        progress_bar.progress(100, text="❌ Failed")
                        st.error(f"Dataset preparation failed: {job.get('error', 'Unknown error')}")
                        break
                else:
                    st.error(f"⏱️ Timeout — dataset preparation took more than {MAX_WAIT_SECONDS}s. Check VM logs.")

        except Exception as e:
            st.error(f"Error: {e}")

if st.session_state.get("dataset_path"):
    st.success("✅ Dataset is ready for training.")
    if st.button("Next: Train Model ➡️"):
        st.switch_page("pages/3_Models.py")
