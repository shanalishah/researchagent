"""
data_pipeline/scheduler.py
Daily pipeline wrapper: fetch → build index.

Cron usage (Unix/macOS):
    0 2 * * * cd /repo && python data_pipeline/scheduler.py >> logs/pipeline.log 2>&1

Windows Task Scheduler: point to this script with the project root as working directory.
"""
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Ensure project root is on sys.path when run as a script
_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def run() -> None:
    """Run fetch → build → rclone sync sequentially; exit non-zero on any failure."""
    import os
    logger.info("=== Pipeline run: %s ===", datetime.now().isoformat())

    steps = [
        ([sys.executable, "data_pipeline/fetch_corpus.py"], "Fetch"),
        ([sys.executable, "data_pipeline/build_index.py"], "Build index"),
    ]

    for cmd, label in steps:
        logger.info("--- %s ---", label)
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            logger.error("%s failed (exit code %d)", label, result.returncode)
            sys.exit(result.returncode)

    # 3. R2 Push (Optional: if R2_BUCKET is set and rclone is installed)
    bucket = os.getenv("R2_BUCKET")
    remote = os.getenv("RCLONE_REMOTE", "r2")
    
    if bucket:
        logger.info("--- R2 Push (rclone) ---")
        try:
            rclone_cmd = [
                "rclone", "sync", "data_pipeline/", f"{remote}:{bucket}/corpus/",
                "--include", "corpus.db",
                "--include", "index_minilm.faiss",
                "--include", "embeddings_minilm.npy",
                "--include", "id_map.json",
                "--include", "build_meta.json",
                "--include", "bm25_index/**",
                "--transfers", "4",
                "--progress"
            ]
            # Replace --progress with --silent when not in a terminal or if you prefer less noise
            subprocess.run(rclone_cmd, check=True)
            logger.info("R2 Push complete.")
        except FileNotFoundError:
            logger.warning("rclone not found on PATH. Skipping R2 push.")
        except subprocess.CalledProcessError as e:
            logger.error("rclone sync failed with exit code %d", e.returncode)
            # We don't exit non-zero here because the local build succeeded; 
            # push failure is a distribution issue, not a build issue.
    else:
        logger.info("R2_BUCKET not set. Skipping R2 sync.")

    logger.info("=== Done ===")


if __name__ == "__main__":
    run()
