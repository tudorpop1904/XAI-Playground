"""
api/workers/consumer.py

RabbitMQ consumer — runs in a background daemon thread started at FastAPI lifespan.
Listens on the 'xai_jobs' queue and dispatches to task handlers based on message type.

Supported job types:
  - "kaggle_prepare"  : Downloads and indexes a Kaggle dataset.
  - "hf_generate"     : Generates an image via HuggingFace InferenceClient.
"""

import json
import os
import time
import threading

import pika

from api.workers import job_store
from core.utils.logger import get_logger

logger = get_logger(__name__)

QUEUE_NAME = "xai_jobs"
RABBITMQ_URL = os.environ.get("RABBITMQ_URL", "amqp://guest:guest@localhost:5672/")

# ─── Task Handlers ────────────────────────────────────────────────────────────

def _handle_kaggle_prepare(job_id: str, payload: dict) -> None:
    """Downloads and indexes a Kaggle dataset."""
    import kagglehub
    from pathlib import Path

    slug = payload["slug"]
    job_store.update_job(job_id, "running")
    logger.info(f"[Worker] kaggle_prepare started: slug={slug}")

    try:
        dpath_str = kagglehub.dataset_download(slug, force_download=False)
        dpath = Path(dpath_str)

        real_images = (
            list(dpath.rglob("*real*/*.jpg")) + list(dpath.rglob("*real*/*.png"))
        )
        fake_images = (
            list(dpath.rglob("*fake*/*.jpg"))
            + list(dpath.rglob("*fake*/*.png"))
            + list(dpath.rglob("*ai*/*.jpg"))
        )

        if not real_images or not fake_images:
            all_imgs = list(dpath.rglob("*.jpg")) + list(dpath.rglob("*.png"))
            half = len(all_imgs) // 2
            real_images, fake_images = all_imgs[:half], all_imgs[half:]

        out_dir = Path("storage/datasets")
        out_dir.mkdir(parents=True, exist_ok=True)
        folder_name = slug.split("/")[-1]
        index_path = out_dir / f"{folder_name}_index.json"

        index_data = {
            "real": [str(p.resolve()) for p in real_images],
            "fake": [str(p.resolve()) for p in fake_images],
        }
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(index_data, f)

        logger.info(f"[Worker] kaggle_prepare done: {len(real_images)} real, {len(fake_images)} fake")
        job_store.update_job(
            job_id,
            "done",
            result={
                "status": "success",
                "path": str(index_path),
                "reals": len(real_images),
                "fakes": len(fake_images),
                "slug": slug,
            },
        )
    except Exception as exc:
        logger.error(f"[Worker] kaggle_prepare failed: {exc}")
        job_store.update_job(job_id, "error", error=str(exc))


def _handle_hf_generate(job_id: str, payload: dict) -> None:
    """Generates an image via HuggingFace InferenceClient and saves it to disk.

    Uses huggingface_hub.InferenceClient which automatically selects the best
    available provider (fal-ai, together, replicate, hf-inference, etc.) for
    the requested model — unlike hardcoding 'hf-inference' which only supports
    a small subset of models.
    """
    from huggingface_hub import InferenceClient
    from pathlib import Path

    prompt = payload["prompt"]
    model_id = payload["model_id"]
    hf_token = payload["hf_token"]

    job_store.update_job(job_id, "running")
    logger.info(f"[Worker] hf_generate started: model={model_id}")

    start_time = time.time()

    try:
        client = InferenceClient(token=hf_token)
        logger.info(f"[Worker] Calling InferenceClient.text_to_image(model={model_id})")

        # Returns a PIL Image directly — handles provider routing automatically
        image = client.text_to_image(prompt, model=model_id)

        from api.main import IMAGES
        IMAGES.mkdir(parents=True, exist_ok=True)
        save_path = IMAGES / f"generated_{int(time.time())}.jpg"
        image.save(save_path)
        elapsed = time.time() - start_time

        logger.info(f"[Worker] hf_generate done in {elapsed:.2f}s: {save_path}")
        job_store.update_job(
            job_id,
            "done",
            result={
                "status": "success",
                "image_path": str(save_path),
                "time_taken": elapsed,
            },
        )
    except Exception as exc:
        logger.error(f"[Worker] hf_generate failed: {exc}")
        job_store.update_job(job_id, "error", error=str(exc))


# ─── Dispatcher ───────────────────────────────────────────────────────────────

_HANDLERS = {
    "kaggle_prepare": _handle_kaggle_prepare,
    "hf_generate": _handle_hf_generate,
}


def _on_message(channel, method, properties, body):
    """Called by pika for each message received from the queue."""
    try:
        msg = json.loads(body)
        job_id = msg["job_id"]
        job_type = msg["type"]
        payload = msg.get("payload", {})

        handler = _HANDLERS.get(job_type)
        if handler is None:
            logger.warning(f"[Worker] Unknown job type: {job_type}. Discarding.")
            channel.basic_ack(delivery_tag=method.delivery_tag)
            return

        # Run handler — errors are caught inside each handler
        handler(job_id, payload)
    except Exception as exc:
        logger.error(f"[Worker] Unexpected error processing message: {exc}")
    finally:
        channel.basic_ack(delivery_tag=method.delivery_tag)


# ─── Consumer Thread ──────────────────────────────────────────────────────────

def _run_consumer() -> None:
    """
    Blocking consumer loop — reconnects on connection errors with exponential backoff.
    Meant to be run inside a daemon thread.
    """
    attempt = 0
    while True:
        try:
            params = pika.URLParameters(RABBITMQ_URL)
            params.socket_timeout = 10
            connection = pika.BlockingConnection(params)
            channel = connection.channel()
            channel.queue_declare(queue=QUEUE_NAME, durable=True)
            channel.basic_qos(prefetch_count=1)
            channel.basic_consume(queue=QUEUE_NAME, on_message_callback=_on_message)
            logger.info(f"[RabbitMQ Consumer] Connected. Waiting for jobs on '{QUEUE_NAME}'...")
            attempt = 0
            channel.start_consuming()
        except Exception as exc:
            wait = min(2 ** attempt, 60)
            logger.warning(
                f"[RabbitMQ Consumer] Connection lost ({exc}). Reconnecting in {wait}s..."
            )
            time.sleep(wait)
            attempt += 1


def start_consumer_thread() -> threading.Thread:
    """Starts the RabbitMQ consumer in a background daemon thread."""
    thread = threading.Thread(target=_run_consumer, name="rabbitmq-consumer", daemon=True)
    thread.start()
    logger.info("[RabbitMQ Consumer] Background thread started.")
    return thread
