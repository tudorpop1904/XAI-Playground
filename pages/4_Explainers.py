import streamlit as st
import requests
import io
import os
import subprocess
from pathlib import Path
from PIL import Image
from pages.config import API_BASE_URL

st.title("🔦 Step 3: Explainers (XAI)")

if not st.session_state.get("model_trained") or not st.session_state.get("test_image_path"):
    st.warning("Please train a model and select a test image first.")
    st.stop()

st.markdown("Select one or more Explainable AI methods to interpret the model's decision on a test image.")

# --- Custom Image Upload ---
uploaded_file = st.file_uploader("Upload a custom image to analyze (Optional)", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    # Save the custom image to the storage directory
    custom_image_dir = Path("storage/images")
    custom_image_dir.mkdir(parents=True, exist_ok=True)
    custom_image_path = custom_image_dir / "custom_upload.jpg"
    
    with open(custom_image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    st.session_state.test_image_path = str(custom_image_path)
    st.success("Custom image loaded successfully!")

st.image(st.session_state.test_image_path, caption="Test Image", width=200)
# ---------------------------

explainer_options = ["grad_cam", "vanilla_saliency", "occlusion", "pmi", "sobol"]
selected_explainers = st.multiselect("XAI Methods", explainer_options, default=["grad_cam"])

st.markdown("#### Hyperparameters")
col1, col2, col3 = st.columns(3)
grid_rows = col1.slider("Grid Rows (Perturbation)", 2, 16, 4)
grid_cols = col2.slider("Grid Cols (Perturbation)", 2, 16, 4)
n_samples = col3.slider("N Samples (Sobol)", 16, 256, 64)

enhance_inference = st.toggle("Apply SuperResolution to test image before inference?")
enable_manim = st.checkbox("Render Manim Animations for Explainers", value=False)

if st.button("Generate Explanations", type="primary"):
    if not selected_explainers:
        st.error("Please select at least one explainer.")
    else:
        st.session_state.xai_results = []
        
        with st.spinner("Analyzing image..."):
            # Load the test image to upload
            with open(st.session_state.test_image_path, "rb") as f:
                img_bytes = f.read()

            filename = Path(st.session_state.test_image_path).name
            for expl in selected_explainers:
                files = {"file": (filename, img_bytes, "image/jpeg")}
                data = {
                    "detector": st.session_state.model_name,
                    "explainer": expl,
                    "enhance": enhance_inference,
                    "grid_rows": grid_rows,
                    "grid_cols": grid_cols,
                    "n_samples": n_samples
                }
                try:
                    res = requests.post(f"{API_BASE_URL}/api/v1/analyze", files=files, data=data)
                    if res.status_code == 200:
                        analysis = res.json()
                        st.session_state.detection_result = {
                            "label": "Deepfake" if analysis["ai_deepfake"] else "Real",
                            "confidence": analysis["confidence"],
                        }
                        
                        st.session_state.xai_results.append({
                            "explainer": expl,
                            "metrics": analysis["metrics"],
                            "heatmap_path": analysis["returned_obj"] # This is a path to the .pt file!
                        })
                    else:
                        st.error(f"Error for {expl}: {res.text}")
                except Exception as e:
                    st.error(f"Failed to connect: {e}")
                    
if st.session_state.get("xai_results"):
    st.success("✅ Explanations generated!")
    st.metric("Prediction", st.session_state.detection_result["label"], f"{st.session_state.detection_result['confidence']:.1%}")
    
    st.markdown("### XAI Heatmaps & Metrics")
    import torch
    import matplotlib.pyplot as plt
    import numpy as np
    
    # We display each explainer's results
    for result in st.session_state.xai_results:
        expl_name = result["explainer"].replace("_", " ").title()
        st.subheader(expl_name)
        col1, col2 = st.columns([1, 1])
        
        with col1:
            try:
                from PIL import Image
                # Load the heatmap tensor and render via matplotlib
                heatmap_tensor = torch.load(result["heatmap_path"], weights_only=False).cpu().detach().numpy()
                if heatmap_tensor.ndim == 3 and heatmap_tensor.shape[0] == 1:
                    heatmap_tensor = heatmap_tensor.squeeze(0) # [1, H, W] -> [H, W]
                    
                # Load original image and resize to match heatmap shape
                orig_img = Image.open(st.session_state.test_image_path).convert("RGB")
                orig_img = orig_img.resize((heatmap_tensor.shape[1], heatmap_tensor.shape[0]))
                    
                fig, ax = plt.subplots()
                ax.imshow(orig_img)
                ax.imshow(heatmap_tensor, cmap='jet', alpha=0.5)
                ax.axis('off')
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Could not render heatmap: {e}")
                
        with col2:
            metrics = result["metrics"]
            st.metric("Sparsity", f"{metrics.get('sparsity', 0):.4f}")
            st.metric("Faithfulness", f"{metrics.get('faithfulness', 0):.4f}")
            st.metric("Stability", f"{metrics.get('stability', 0):.4f}")
            
        # --- Dynamic Manim Generation for Explainers ---
        if enable_manim:
            # Map the selected explainer to the corresponding Manim scene class
            scene_map = {
                "grad_cam": "GradCAMScene",
                "vanilla_saliency": "VanillaSaliencyScene",
                "occlusion": "OcclusionSensitivityScene",
                "pmi": "PMIScene",
                "sobol": "SobolScene"
            }
            
            scene_name = scene_map.get(result["explainer"])
            if scene_name:
                manim_output = Path(f"{scene_name}.mp4")
                
                if not manim_output.exists():
                    with st.spinner(f"Rendering Manim Animation for {expl_name}... This may take 10-20 seconds."):
                        try:
                            # -qm for medium quality, -o to specify output name in current dir
                            subprocess.run(
                                ["manim", "manim/xai_animations.py", scene_name, "-qm", "-o", f"../{manim_output.name}"],
                                check=True,
                                capture_output=True
                            )
                        except subprocess.CalledProcessError as e:
                            st.error(f"Failed to generate animation: {e.stderr.decode()}")
                
                if manim_output.exists():
                    st.video(str(manim_output))
        # -----------------------------------------------
            
    if st.button("Next: Interpret with LLM ➡️"):
        st.switch_page("pages/5_Interpretation.py")
