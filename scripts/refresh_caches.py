"""
Nightly cache refresher: runs the discovery pipeline and uploads cache artifacts to OCI Object Storage.

Usage (local):
  source .venv/bin/activate
  python scripts/refresh_caches.py --bucket <bucket-name> --namespace <ns> --region <region>

In CI (GitHub Actions) the workflow will set env vars for OCI credentials.

This script imports the backend pipeline to compute HMM/pairs/ranking and then uploads the cache files
that live under `data/cache/` to OCI Object Storage under a prefix `cache-backups/<timestamp>/`.

It requires the `oci` Python SDK (install via `pip install oci`) or the `oci` CLI available.

Secrets required (for GitHub Actions):
- OCI_TENANCY_OCID
- OCI_USER_OCID
- OCI_FINGERPRINT
- OCI_PRIVATE_KEY (base64-encoded or multi-line with newlines escaped)
- OCI_REGION
- OCIR_BUCKET_NAME (or pass as action input)

"""

import argparse
import base64
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Ensure repo src is importable
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import the backend helper to run discovery pipeline
try:
    # Import wrappers from dashboard/backend if available
    from dashboard.backend.app import _run_discovery_pipeline
except Exception:
    # Fallback: try importing _run_discovery_pipeline from installed path
    try:
        from dashboard.backend.app import _run_discovery_pipeline  # type: ignore
    except Exception as exc:
        print("Failed to import backend pipeline. Ensure you're running from repository root and virtualenv is active.")
        raise

# OCI SDK (optional)
try:
    import oci
    import shutil
except Exception:
    oci = None


def upload_directory_to_bucket(namespace, bucket, prefix, local_dir, region=None, oci_config=None):
    """Upload all files under local_dir to OCI Object Storage at prefix/."""
    if oci is None:
        raise RuntimeError("OCI SDK not available. Install 'oci' package or use OCI CLI in your environment.")

    config = oci_config or oci.config.from_file()  # will use env vars if configured
    obj_client = oci.object_storage.ObjectStorageClient(config)

    for p in sorted(Path(local_dir).rglob("*.pkl")):
        rel = p.relative_to(local_dir)
        object_name = f"{prefix}/{rel.as_posix()}"
        print(f"Uploading {p} -> {bucket}/{object_name}")
        with open(p, "rb") as fh:
            data = fh.read()
        obj_client.put_object(namespace_name=namespace, bucket_name=bucket, object_name=object_name, put_object_body=data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", required=True, help="OCI Object Storage bucket name to upload cache artifacts")
    parser.add_argument("--namespace", required=True, help="OCI tenancy namespace")
    parser.add_argument("--region", required=False, help="OCI region (optional, used for config)")
    parser.add_argument("--prefix", required=False, default=None, help="Prefix inside bucket (defaults to timestamp)")
    parser.add_argument("--tickers", required=False, nargs="*", help="Optional list of tickers to run discovery for; if omitted, uses platform defaults")
    parser.add_argument("--force", action="store_true", help="Force recompute even if cache exists locally")
    args = parser.parse_args()

    cfg_payload = {"tickers": args.tickers} if args.tickers else {}

    # Run discovery pipeline (this will write cache files under data/cache/)
    print("Running discovery pipeline...")
    try:
        res = _run_discovery_pipeline(cfg_payload)
    except Exception as e:
        print("Discovery pipeline failed:", e)
        raise

    cache_dir = ROOT / "data" / "cache"
    if not cache_dir.exists():
        print("No cache directory found after discovery. Nothing to upload.")
        return

    prefix = args.prefix or datetime.utcnow().strftime("cache-backups/%Y%m%dT%H%M%SZ")

    # Build OCI config from env if present
    oci_config = None
    if all(os.environ.get(k) for k in ("OCI_TENANCY_OCID", "OCI_USER_OCID", "OCI_FINGERPRINT", "OCI_PRIVATE_KEY")):
        # Create a temp config dict
        oci_config = {
            "tenancy": os.environ["OCI_TENANCY_OCID"],
            "user": os.environ["OCI_USER_OCID"],
            "fingerprint": os.environ["OCI_FINGERPRINT"],
            "key_file": None,
            "region": args.region or os.environ.get("OCI_REGION"),
        }
        # If private key is provided as base64, decode to a temp file
        pk = os.environ.get("OCI_PRIVATE_KEY")
        if pk and "-----BEGIN" not in pk:
            # assume base64
            raw = base64.b64decode(pk).decode()
        else:
            raw = pk
        # write temp key
        key_path = ROOT / ".oci_temp_key.pem"
        with open(key_path, "w") as kf:
            kf.write(raw)
        os.chmod(key_path, 0o600)
        oci_config["key_file"] = str(key_path)

    if oci is None and not shutil.which("oci"):
        print("Neither OCI Python SDK installed nor 'oci' CLI found. Install one to upload artifacts.")
        return

    print(f"Uploading cache files from {cache_dir} to bucket {args.bucket} with prefix {prefix}")
    try:
        upload_directory_to_bucket(args.namespace, args.bucket, prefix, cache_dir, region=args.region, oci_config=oci_config)
    except Exception as e:
        print("Upload failed:", e)
        raise

    print("Upload complete.")


if __name__ == "__main__":
    main()
