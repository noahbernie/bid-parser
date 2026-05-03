import os
import json
import boto3
from botocore.exceptions import BotoCoreError, ClientError


def _client():
    return boto3.client(
        "s3",
        region_name=os.environ.get("S3_REGION", "us-east-1"),
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
    )


def _bucket():
    b = os.environ.get("S3_BUCKET")
    if not b:
        raise RuntimeError("S3_BUCKET env var not set")
    return b


def upload_file(bucket: str, key: str, data: bytes | str, content_type: str = "application/octet-stream") -> None:
    if isinstance(data, str):
        data = data.encode("utf-8")
    _client().put_object(Bucket=bucket, Key=key, Body=data, ContentType=content_type)


def upload_pdf(job_id: str, org_id: str, pdf_bytes: bytes, filename: str) -> tuple[str, str]:
    """Upload original PDF. Returns (bucket, key)."""
    bucket = _bucket()
    key = f"orgs/{org_id}/jobs/{job_id}/original.pdf"
    upload_file(bucket, key, pdf_bytes, content_type="application/pdf")
    print(f"[s3] uploaded PDF → s3://{bucket}/{key} ({len(pdf_bytes)//1024}KB)")
    return bucket, key


def upload_stage_log(job_id: str, stage_order: int, stage_name: str, log_data: dict) -> str:
    """Upload stage log JSON. Returns s3_key."""
    bucket = _bucket()
    safe_name = stage_name.lower().replace(" ", "_").replace("/", "_")
    key = f"parser-logs/{job_id}/{stage_order}_{safe_name}.json"
    upload_file(bucket, key, json.dumps(log_data, indent=2, default=str), content_type="application/json")
    print(f"[s3] uploaded stage log → s3://{bucket}/{key}")
    return key
