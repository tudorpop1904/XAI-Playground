"""
api/workers/producer.py

Publishes job messages to the RabbitMQ 'xai_jobs' queue.
Each message is a JSON object:
  {
    "job_id": "<uuid>",
    "type": "kaggle_prepare" | "hf_generate",
    "payload": { ...task-specific args... }
  }

Opens a fresh connection per publish call (thread-safe, stateless).
"""

import json
import os
import time

import pika

from core.utils.logger import get_logger

logger = get_logger(__name__)

QUEUE_NAME = "xai_jobs"
RABBITMQ_URL = os.environ.get("RABBITMQ_URL", "amqp://guest:guest@localhost:5672/")


def publish_job(job_type: str, payload: dict, job_id: str) -> None:
    """
    Publishes a job to the xai_jobs queue.

    Args:
        job_type: One of "kaggle_prepare" or "hf_generate".
        payload:  Task-specific parameters dict.
        job_id:   Pre-created job ID from job_store.create_job().
    """
    message = json.dumps({"job_id": job_id, "type": job_type, "payload": payload})

    # Retry with exponential backoff in case RabbitMQ is briefly unavailable
    for attempt in range(5):
        try:
            params = pika.URLParameters(RABBITMQ_URL)
            params.socket_timeout = 5
            connection = pika.BlockingConnection(params)
            channel = connection.channel()
            channel.queue_declare(queue=QUEUE_NAME, durable=True)
            channel.basic_publish(
                exchange="",
                routing_key=QUEUE_NAME,
                body=message,
                properties=pika.BasicProperties(
                    delivery_mode=pika.DeliveryMode.Persistent
                ),
            )
            connection.close()
            logger.info(f"[RabbitMQ] Published job {job_id} (type={job_type})")
            return
        except Exception as exc:
            wait = 2 ** attempt
            logger.warning(
                f"[RabbitMQ] Publish attempt {attempt + 1} failed: {exc}. Retrying in {wait}s..."
            )
            time.sleep(wait)

    raise RuntimeError(f"[RabbitMQ] Failed to publish job {job_id} after 5 attempts.")
