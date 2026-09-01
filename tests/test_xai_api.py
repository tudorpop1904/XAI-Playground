"""
tests/test_xai_api.py
======================
Integration test script for XAI explainers via FastAPI backend.
"""

from __future__ import annotations

import io
from typing import Any

from PIL import Image
import requests

API_URL = "http://localhost:8000/api/v1/analyze"


def test_explainers() -> None:
    """
    Test all available XAI explainer endpoints with a dummy test image.
    """
    # 1. Create a synthetic test image
    dummy_img = Image.new("RGB", (128, 128), color="red")
    img_byte_arr = io.BytesIO()
    dummy_img.save(img_byte_arr, format="JPEG")
    img_bytes = img_byte_arr.getvalue()

    explainers = ["grad_cam", "vanilla_saliency", "occlusion", "pmi", "sobol"]

    print(f"Testing {len(explainers)} explainers on API {API_URL}...")
    success_count = 0

    for explainer in explainers:
        print(f"\n--- Testing {explainer} ---")
        files = {"file": ("test.jpg", img_bytes, "image/jpeg")}
        data = {
            "detector": "CNN_Model",
            "explainer": explainer,
            "enhance": False,
        }

        try:
            res = requests.post(API_URL, files=files, data=data, timeout=30)
            if res.status_code == 200:
                print(f"[SUCCESS] {explainer}!")
                res_data: dict[str, Any] = res.json()
                print(f"   Confidence: {res_data.get('confidence')}")
                print(f"   Faithfulness: {res_data.get('metrics', {}).get('faithfulness')}")
                success_count += 1
            else:
                print(f"[FAILED] {explainer} with status {res.status_code}")
                print(res.text[:500])
        except Exception as err:
            print(f"[EXCEPTION] {explainer}: {err}")

    print(f"\n=== SUMMARY: {success_count}/{len(explainers)} Passed ===")


if __name__ == "__main__":
    test_explainers()
