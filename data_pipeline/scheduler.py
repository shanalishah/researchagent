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
from dotenv import load_dotenv
import argparse

# Ensure project root is on sys.path when run as a script
_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

# Load environment variables early (e.g. for R2 sync credentials)
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def run_command(cmd, label):
    """Run a shell command and exit on failure."""
    logger.info("--- %s ---", label)
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        logger.error("%s failed (exit code %d)", label, result.returncode)
        sys.exit(result.returncode)


def push_to_r2():
    """Sync artifacts to Cloudflare R2 using rclone."""
    import os
    bucket = os.getenv("R2_BUCKET")
    remote = os.getenv("RCLONE_REMOTE", "r2")
    
    if not bucket:
        logger.info("R2_BUCKET not set. Skipping R2 sync.")
        return

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
        subprocess.run(rclone_cmd, check=True)
        logger.info("R2 Push complete.")
    except FileNotFoundError:
        logger.warning("rclone not found on PATH. Skipping R2 push.")
    except subprocess.CalledProcessError as e:
        logger.error("rclone sync failed with exit code %d", e.returncode)


def run() -> None:
    """Orchestrate the data pipeline based on selected mode."""
    parser = argparse.ArgumentParser(description="ResearchAgent Pipeline Scheduler")
    parser.add_argument(
        "mode", 
        choices=["arxiv", "s2", "sync", "all"], 
        default="all", 
        nargs="?",
        help="Pipeline mode: 'arxiv' (scout+index), 's2' (enrich+index), 'sync' (r2 only), or 'all' (default)"
    )
    parser.add_argument("--full", action="store_true", help="Pass --full to ingestion and build scripts")
    parser.add_argument("--days", type=int, default=30, help="Days to look back for arXiv scout")
    args = parser.parse_args()

    logger.info("=== Pipeline run mode: %s ===", args.mode)
    
    # 1. arXiv Workflow: Scout + Index
    if args.mode in ["arxiv", "all"]:
        fetch_cmd = [sys.executable, "data_pipeline/fetch_corpus.py", "--arxiv", "--days", str(args.days)]
        if args.full: fetch_cmd.append("--full")
        run_command(fetch_cmd, "Stage 1: arXiv Scout")
        
        build_cmd = [sys.executable, "data_pipeline/build_index.py"]
        if args.full: build_cmd.append("--full")
        run_command(build_cmd, "Incremental Indexing")

    # 2. S2 Workflow: Enrichment + Index
    if args.mode in ["s2", "all"]:
        fetch_cmd = [sys.executable, "data_pipeline/fetch_corpus.py", "--s2"]
        if args.full: fetch_cmd.append("--full")
        run_command(fetch_cmd, "Stage 2: S2 Enrichment")
        
        build_cmd = [sys.executable, "data_pipeline/build_index.py"]
        if args.full: build_cmd.append("--full")
        run_command(build_cmd, "Incremental Indexing")

    # 3. R2 Sync Workflow
    if args.mode in ["sync", "all"]:
        push_to_r2()

    logger.info("=== Pipeline execution finished ===")


if __name__ == "__main__":
    run()
