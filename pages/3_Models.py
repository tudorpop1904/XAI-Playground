import streamlit as st
import requests
import random
from pathlib import Path
import os
import subprocess

st.title("🧠 Step 2: Models")

if not st.session_state.get("dataset_path"):
    st.warning("Please select and prepare a dataset first.")
    st.stop()
    
st.markdown("Select an architecture and train it on your prepared dataset.")

model_type = st.selectbox("Model Architecture", ["CNN", "ViT", "KNN"])

# Fetch available models
available_models = []
try:
    res = requests.get("http://localhost:8000/api/v1/models")
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
        col1, col2 = st.columns(2)
        col1.metric("Train Accuracy", f"{hist['train_acc'][-1]:.2%}")
        col2.metric("Train Loss", f"{hist['train_loss'][-1]:.4f}")
        
elif action == "Train New Model":
    # Generate next index for the model name
    next_index = len(arch_models) + 1
    model_name = f"{model_type}_Model_{next_index}"
    st.info(f"Will train a new model named: **{model_name}**")
    
    st.markdown("#### Hyperparameters")
    col1, col2, col3 = st.columns(3)
    epochs = col1.slider("Epochs", min_value=1, max_value=50, value=1)
    batch_size = col2.slider("Batch Size", min_value=4, max_value=128, value=16, step=4)
    learning_rate = col3.number_input("Learning Rate", min_value=0.00001, max_value=0.1, value=0.001, step=0.0001, format="%.5f")
    
    if st.button("Train Model", type="primary"):
        with st.spinner(f"Training {model_name} on {st.session_state.dataset_slug}..."):
            try:
                res = requests.post(
                    "http://localhost:8000/api/v1/models/train",
                    json={
                        "model_name": model_name,
                        "dataset_slug": st.session_state.dataset_slug,
                        "epochs": epochs,
                        "batch_size": batch_size,
                        "learning_rate": learning_rate
                    }
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
                        col1.metric("Final Train Accuracy", f"{hist['train_acc'][-1]:.2%}")
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
