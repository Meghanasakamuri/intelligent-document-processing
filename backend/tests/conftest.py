"""Test configuration and fixtures for IDP backend tests."""

import os
import shutil
from pathlib import Path

TEST_DIR = Path(__file__).resolve().parent
TEST_DB = TEST_DIR / "test_documents.db"
TEST_UPLOADS = TEST_DIR / "test_uploads"

# Configure test environment variables
os.environ["IDP_DATABASE_PATH"] = str(TEST_DB)
os.environ["IDP_UPLOAD_DIR"] = str(TEST_UPLOADS)
TEST_UPLOADS.mkdir(parents=True, exist_ok=True)


def cleanup_test_env():
    """Remove test database and uploaded test files."""
    if TEST_DB.exists():
        try:
            TEST_DB.unlink()
        except Exception:
            pass
    if TEST_UPLOADS.exists():
        try:
            shutil.rmtree(TEST_UPLOADS, ignore_errors=True)
        except Exception:
            pass
