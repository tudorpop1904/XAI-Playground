from typing import Any, Literal
from pydantic import BaseModel, Field

class TrainRequest(BaseModel):
    model_name: str = Field(..., min_length=1, max_length=100, description="Unique model identifier")
    dataset_slug: str = Field(..., min_length=1, description="Kaggle or local dataset slug")
    epochs: int = Field(1, ge=1, le=50, description="Training epochs (1-50)")
    batch_size: int = Field(16, ge=4, le=128, description="Batch size (4-128)")
    learning_rate: float = Field(1e-3, ge=1e-5, le=0.1, description="Learning rate (0.00001 - 0.1)")
    freeze_backbone: bool = Field(False, description="Freeze pre-trained feature extractor")
    model_type: Literal["CNN", "ViT", "KNN", "cnn", "vit", "knn"] = Field("CNN", description="Model architecture")
    add_fft: bool = Field(False, description="Append 2D FFT forensic magnitude channel")
    add_lbp: bool = Field(False, description="Append LBP texture channel")
    add_sobel: bool = Field(False, description="Append Sobel edge gradient channel")
    knn_k: int = Field(5, ge=1, le=25, description="Number of k-NN neighbors (1-25)")
    knn_metric: Literal["cosine", "euclidean"] = Field("cosine", description="Distance metric")
    knn_backbone: Literal["resnet18", "ftl_cnn", "vit"] = Field("resnet18", description="k-NN feature extractor")

class InterpretationRequest(BaseModel):
    detector_name: str = Field(..., min_length=1, description="Name of the detector model used")
    explainer_name: Literal["grad_cam", "vanilla_saliency", "occlusion", "pmi", "sobol"] = Field(..., description="XAI method")
    ai_deepfake: bool = Field(..., description="Binary classification outcome")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model prediction confidence score (0.0 - 1.0)")
    metrics: dict[str, Any] = Field(default_factory=dict, description="Quantitative XAI metrics dictionary")
    llm_model: str = Field("llama3.1:8b-instruct-q4_K_M", min_length=1, description="Ollama model identifier")

class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=1000, description="Diffusion text prompt")
    mode: Literal["cloud", "local"] = Field("cloud", description="Inference mode ('cloud' or 'local')")
    hf_token: str = Field("", description="Hugging Face API token for cloud inference")
    model_id: Literal[
        "runwayml/stable-diffusion-v1-5",
        "stabilityai/stable-diffusion-2-1",
        "stabilityai/stable-diffusion-xl-base-1.0"
    ] = Field("runwayml/stable-diffusion-v1-5", description="Pretrained diffusion model repo")
