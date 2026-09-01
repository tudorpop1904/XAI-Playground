import streamlit as st
import requests
import random
from pathlib import Path
import os
import subprocess
from pages.config import API_BASE_URL

st.title("🧠 Step 2: Models")

if not st.session_state.get("dataset_path"):
    st.warning("Please select and prepare a dataset first.")
    st.stop()
    
st.markdown("Select an architecture and train it on your prepared dataset.")

model_type = st.selectbox("Model Architecture", ["CNN", "ViT", "KNN", "KMC"])

# Fetch available models
available_models = []
try:
    res = requests.get(f"{API_BASE_URL}/api/v1/models")
    if res.status_code == 200:
        available_models = res.json()
except Exception:
    pass

# Filter models by selected architecture
arch_models = [m for m in available_models if m["name"].startswith(model_type)]

action = st.radio("Action", ["Train New Model", "Use Existing Model"] if arch_models else ["Train New Model"])

if action == "Use Existing Model" and arch_models:
    model_options = {m["name"]: m for m in arch_models}
    selected_model_name = st.selectbox("Select Cached Model", list(model_options.keys()))
    selected_model = model_options[selected_model_name]
    
    st.session_state.model_name = selected_model_name
    st.session_state.model_trained = True
    
    hist = selected_model.get("history", {})
    if hist:
        specs = hist.get("specs", {})
        col1, col2, col3 = st.columns(3)
        if "train_acc" in hist and hist["train_acc"]:
            col1.metric("Train Accuracy", f"{hist['train_acc'][-1]:.2%}")
        if "train_loss" in hist and hist["train_loss"]:
            col2.metric("Train Loss", f"{hist['train_loss'][-1]:.4f}")
        if specs:
            channels = specs.get("input_channels", 3)
            col3.metric("Input Channels", f"{channels} (RGB{' + FFT' if specs.get('add_fft') else ''}{' + LBP' if specs.get('add_lbp') else ''}{' + Sobel' if specs.get('add_sobel') else ''})")
        
elif action == "Train New Model":
    # Generate next index for the model name
    next_index = len(arch_models) + 1
    model_name = f"{model_type}_Model_{next_index}"
    st.info(f"Will train a new model named: **{model_name}**")
    
    # Forensic Channels Selection (Specific to CNN Early Fusion)
    add_fft = False
    add_lbp = False
    add_sobel = False
    
    if model_type == "CNN":
        st.markdown("#### 🔬 Forensic Feature Channels (Faster-Than-Lies)")
        st.caption("Choose which additional mathematical filters to prepend to the 3 RGB channels:")
        col_fft, col_lbp, col_sobel = st.columns(3)
        add_fft = col_fft.checkbox("FFT 2D Magnitude", value=False, help="Fast Fourier Transform channel to detect high-frequency artifacts and upsampling checkerboards.")
        add_lbp = col_lbp.checkbox("LBP Texture", value=False, help="Local Binary Pattern channel to detect unnatural skin and surface micro-texture smoothness.")
        add_sobel = col_sobel.checkbox("Sobel Edge Filter", value=False, help="Sobel gradient magnitude channel to detect border bleed and structural discontinuities.")
        
        num_channels = 3 + int(add_fft) + int(add_lbp) + int(add_sobel)
        ch_labels = ["RGB (3)"]
        if add_fft: ch_labels.append("FFT (1)")
        if add_lbp: ch_labels.append("LBP (1)")
        if add_sobel: ch_labels.append("Sobel (1)")
        st.caption(f"Input Tensor Shape: `[{num_channels}, 128, 128]` ({' + '.join(ch_labels)})")

    st.markdown("#### Hyperparameters")
    col1, col2, col3 = st.columns(3)
    epochs = col1.slider("Epochs", min_value=1, max_value=50, value=1)
    batch_size = col2.slider("Batch Size", min_value=4, max_value=128, value=16, step=4)
    learning_rate = col3.number_input("Learning Rate", min_value=0.00001, max_value=0.1, value=0.001, step=0.0001, format="%.5f")
    
    if st.button("Train Model", type="primary"):
        with st.spinner(f"Training {model_name} on {st.session_state.dataset_slug}..."):
            try:
                payload = {
                    "model_name": model_name,
                    "dataset_slug": st.session_state.dataset_slug,
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "model_type": model_type,
                    "add_fft": add_fft,
                    "add_lbp": add_lbp,
                    "add_sobel": add_sobel,
                }
                res = requests.post(
                    f"{API_BASE_URL}/api/v1/models/train",
                    json=payload
                )
                
                if res.status_code == 200:
                    data = res.json()
                    st.success(f"Model trained successfully!")
                    st.session_state.model_name = model_name
                    st.session_state.model_trained = True
                    
                    # Show metrics
                    hist = data.get("history", {})
                    if hist:
                        col1, col2 = st.columns(2)
                        if "train_acc" in hist and hist["train_acc"]:
                            col1.metric("Final Train Accuracy", f"{hist['train_acc'][-1]:.2%}")
                        if "train_loss" in hist and hist["train_loss"]:
                            col2.metric("Final Train Loss", f"{hist['train_loss'][-1]:.4f}")
                else:
                    st.error(f"Training failed: {res.text}")
                    
            except Exception as e:
                st.error(f"Error: {e}")

if st.session_state.get("model_trained"):
    st.success("✅ Model is trained and cached.")
    
    st.markdown("### Manim Architecture Animation")
    st.info("Generating a dynamic Manim animation for this model...")
    
    # We pick a random image from the dataset for the Manim scene
    # We'll just grab the first real image
    dataset_path = Path(st.session_state.dataset_path)
    if dataset_path.suffix == ".json":
        import json
        with open(dataset_path, "r", encoding="utf-8") as f:
            idx_data = json.load(f)
        if idx_data.get("real"):
            st.session_state.test_image_path = idx_data["real"][0]
    else:
        real_imgs = list((dataset_path / "real").glob("*.jpg"))
        if real_imgs:
            st.session_state.test_image_path = str(real_imgs[0].resolve())
            
    st.markdown("### Manim Architecture Animation")
    enable_manim = st.checkbox("Render Architecture Animation (Manim)", value=False)
    
    if enable_manim:
        st.info("Generating a dynamic Manim animation for this model...")
        
        # We pick a random image from the dataset for the Manim scene
        # We'll just grab the first real image
        
        # In a fully deployed setup, we would run Manim as an API endpoint, but since we have Streamlit
        # running on the same server, we can invoke it via subprocess directly for demonstration!
        scene_name = f"{model_type}ProcessingScene"
        
        # Check if the animation exists (previous agent might have generated it)
        manim_output = Path(f"{scene_name}.mp4")
        
        if not manim_output.exists():
            with st.spinner(f"Rendering Manim Animation for {model_type}... This may take 10-20 seconds."):
                try:
                    subprocess.run(
                        ["manim", "manim/model_animations.py", scene_name, "-qm", "-o", f"../{manim_output.name}"],
                        check=True,
                        capture_output=True
                    )
                except subprocess.CalledProcessError as e:
                    st.error(f"Failed to generate animation: {e.stderr.decode()}")
                    
        if manim_output.exists():
            st.video(str(manim_output))
    
    if st.button("Next: Explain with XAI ➡️"):
        st.switch_page("pages/4_Explainers.py")
