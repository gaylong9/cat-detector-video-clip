# utils.py
import subprocess
from pathlib import Path
from typing import List, Tuple, Optional
import csv
import logging
import shutil
import tempfile
import os


logger = logging.getLogger(__name__)


def run_cmd(cmd: List[str], capture_output: bool = True) -> Tuple[int, str, str]:
    """
    Run a command via subprocess and return (returncode, stdout, stderr).
    """
    logger.debug("Running command: %s", " ".join(cmd))
    try:
        completed = subprocess.run(cmd, check=False, capture_output=capture_output, text=True)
        return completed.returncode, completed.stdout or "", completed.stderr or ""
    except Exception as e:
        logger.exception("Failed to run command: %s", cmd)
        return 1, "", str(e)


def find_ffmpeg() -> Optional[str]:
    """
    Check ffmpeg existence via shutil.which.
    Returns path or None.
    """
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        logger.debug("ffmpeg found at: %s", ffmpeg_path)
    else:
        logger.error("ffmpeg not found in PATH")
    return ffmpeg_path


def format_seconds(sec: float) -> str:
    """Return string format suitable for ffmpeg (seconds with 2 decimal)"""
    return f"{sec:.2f}"


def write_csv(path: Path, rows: List[List]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["video", "start", "end"])
        writer.writerows(rows)


def append_csv(path: Path, rows: List[List]):
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(["video", "start", "end"])
        writer.writerows(rows)


def read_csv_rows(path: Path) -> List[List[str]]:
    if not path.exists():
        return []
    with path.open("r", newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        rows = [r for r in reader]
    # drop header if present
    if rows and rows[0] and rows[0][0].lower() == "video":
        rows = rows[1:]
    return rows


def remove_tree(path: Path):
    if path.exists():
        if path.is_dir():
            for child in path.iterdir():
                if child.is_dir():
                    remove_tree(child)
                else:
                    child.unlink()
            path.rmdir()
        else:
            path.unlink()


def safe_make_tmp_dir(base: Path, name: str) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    tmp_dir = base / name
    tmp_dir.mkdir(parents=True, exist_ok=True)
    return tmp_dir
