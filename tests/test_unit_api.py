"""
tests/test_unit_api.py
=======================
Comprehensive Unit and Integration Test Suite for FastAPI endpoints.
Can be executed directly via: python -m unittest tests/test_unit_api.py
"""

from __future__ import annotations

import io
import unittest
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from fastapi.testclient import TestClient

from api.main import app
from core.detectors.cnn import CNNDetector
from core.detectors.base import AbstractBaseDetector
from core.utils.paths import MODELS_DIR, BASE_DIR
from api.routers.interpret import build_forensic_prompt
from api.schemas.requests import InterpretationRequest


class TestAPIEndpoints(unittest.TestCase):
    """
    Unit test cases for FastAPI router endpoints.
    """

    @classmethod
    def setUpClass(cls) -> None:
        """Initialize TestClient and register a synthetic model in cache and on disk."""
        cls.client = TestClient(app)
        
        # Ensure directories exist
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        (BASE_DIR / "storage" / "images").mkdir(parents=True, exist_ok=True)

        # Create and save a lightweight mock detector
        cls.test_model_name = "CNN_TestModel_1"
        cls.test_detector = CNNDetector(num_classes=2, add_fft=False, add_lbp=False, add_sobel=False)
        cls.test_detector.name = cls.test_model_name
        cls.test_detector.eval()
        
        cls.model_path = MODELS_DIR / f"{cls.test_model_name}.pth"
        torch.save(cls.test_detector, cls.model_path)
        AbstractBaseDetector._cache[cls.test_model_name] = cls.test_detector

    @classmethod
    def tearDownClass(cls) -> None:
        """Cleanup test model artifact."""
        if cls.model_path.exists():
            try:
                cls.model_path.unlink()
            except Exception:
                pass
        AbstractBaseDetector._cache.pop(cls.test_model_name, None)

    def _create_dummy_image_bytes(self, size: tuple[int, int] = (128, 128)) -> bytes:
        """Helper to create synthetic JPEG image in memory."""
        img = Image.new("RGB", size, color=(120, 180, 240))
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        return buffer.getvalue()

    # ── 1. Interpret & Status Endpoints ──────────────────────────────

    def test_interpret_status_endpoint(self) -> None:
        """Test GET /api/v1/interpret/status returns valid dict structure."""
        response = self.client.get("/api/v1/interpret/status")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("status", data)

    def test_build_forensic_prompt_logic(self) -> None:
        """Test prompt template generation with XAI metrics."""
        req = InterpretationRequest(
            image_name="test_sample.jpg",
            detector_name="CNN_Model_1",
            explainer_name="grad_cam",
            ai_deepfake=True,
            confidence=0.925,
            metrics={"sparsity": 0.75, "faithfulness": 0.82, "stability": 0.91},
            llm_model="llama3.1:8b-instruct-q4_K_M",
        )
        prompt = build_forensic_prompt(req)
        self.assertIn("Deepfake", prompt)
        self.assertIn("CNN_Model_1", prompt)
        self.assertIn("grad_cam", prompt)
        self.assertIn("0.75", prompt)

    # ── 2. Datasets & Training Error Handling ────────────────────────

    def test_datasets_prepare_missing_file(self) -> None:
        """Test POST /api/v1/models/train returns 400 when dataset is not prepared."""
        payload = {
            "model_name": "CNN_UnitTest_1",
            "dataset_slug": "non_existent_dataset_slug",
            "epochs": 1,
            "batch_size": 16,
            "learning_rate": 0.001,
            "model_type": "CNN",
        }
        response = self.client.post("/api/v1/models/train", json=payload)
        self.assertEqual(response.status_code, 400)
        self.assertIn("not found", response.json()["detail"])

    # ── 3. Analysis & XAI Explainer Endpoints ────────────────────────

    def test_analyze_grad_cam(self) -> None:
        """Test POST /api/v1/analyze with Grad-CAM explainer."""
        img_bytes = self._create_dummy_image_bytes()
        files = {"file": ("test_sample.jpg", img_bytes, "image/jpeg")}
        data = {
            "detector": self.test_model_name,
            "explainer": "grad_cam",
            "enhance": "false",
        }
        response = self.client.post("/api/v1/analyze", files=files, data=data)
        self.assertEqual(response.status_code, 200)
        res_json: dict[str, Any] = response.json()
        self.assertIn("confidence", res_json)
        self.assertIn("ai_deepfake", res_json)
        self.assertIn("metrics", res_json)
        self.assertIn("sparsity", res_json["metrics"])
        self.assertIn("faithfulness", res_json["metrics"])

    def test_analyze_vanilla_saliency(self) -> None:
        """Test POST /api/v1/analyze with Vanilla Saliency."""
        img_bytes = self._create_dummy_image_bytes()
        files = {"file": ("test_saliency.jpg", img_bytes, "image/jpeg")}
        data = {
            "detector": self.test_model_name,
            "explainer": "vanilla_saliency",
            "enhance": "false",
        }
        response = self.client.post("/api/v1/analyze", files=files, data=data)
        self.assertEqual(response.status_code, 200)
        res_json: dict[str, Any] = response.json()
        self.assertIn("returned_obj", res_json)

    def test_analyze_occlusion(self) -> None:
        """Test POST /api/v1/analyze with Occlusion Sensitivity."""
        img_bytes = self._create_dummy_image_bytes()
        files = {"file": ("test_occlusion.jpg", img_bytes, "image/jpeg")}
        data = {
            "detector": self.test_model_name,
            "explainer": "occlusion",
            "grid_rows": "2",
            "grid_cols": "2",
            "enhance": "false",
        }
        response = self.client.post("/api/v1/analyze", files=files, data=data)
        self.assertEqual(response.status_code, 200)

    def test_analyze_pmi(self) -> None:
        """Test POST /api/v1/analyze with PMI explainer."""
        img_bytes = self._create_dummy_image_bytes()
        files = {"file": ("test_pmi.jpg", img_bytes, "image/jpeg")}
        data = {
            "detector": self.test_model_name,
            "explainer": "pmi",
            "grid_rows": "2",
            "grid_cols": "2",
            "enhance": "false",
        }
        response = self.client.post("/api/v1/analyze", files=files, data=data)
        self.assertEqual(response.status_code, 200)

    def test_analyze_sobol(self) -> None:
        """Test POST /api/v1/analyze with Sobol Sensitivity Analysis."""
        img_bytes = self._create_dummy_image_bytes()
        files = {"file": ("test_sobol.jpg", img_bytes, "image/jpeg")}
        data = {
            "detector": self.test_model_name,
            "explainer": "sobol",
            "grid_rows": "2",
            "grid_cols": "2",
            "n_samples": "16",
            "enhance": "false",
        }
        response = self.client.post("/api/v1/analyze", files=files, data=data)
        self.assertEqual(response.status_code, 200)

    def test_analyze_with_super_resolution_enhancement(self) -> None:
        """Test POST /api/v1/analyze with enhance=True."""
        img_bytes = self._create_dummy_image_bytes()
        files = {"file": ("test_enhance.jpg", img_bytes, "image/jpeg")}
        data = {
            "detector": self.test_model_name,
            "explainer": "grad_cam",
            "enhance": "true",
        }
        response = self.client.post("/api/v1/analyze", files=files, data=data)
        self.assertEqual(response.status_code, 200)


if __name__ == "__main__":
    unittest.main()
