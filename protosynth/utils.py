"""
ProtoSynth Utilities

Utility functions for metadata collection, reproducibility, and system information.
"""

import json
import platform
import subprocess
import sys
from typing import Any, Dict


def collect_run_metadata() -> Dict[str, Any]:
    """
    Collect comprehensive metadata about the current run environment.

    This includes Python version, platform info, git commit, ProtoSynth version,
    and installed package versions for full reproducibility.

    Returns:
        Dict containing all environment metadata
    """
    try:
        git_sha = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        git_sha = "unknown"

    try:
        # Get installed packages
        pip_output = (
            subprocess.check_output([sys.executable, "-m", "pip", "freeze"])
            .decode()
            .splitlines()
        )

        packages = {}
        for line in pip_output:
            if "@" not in line and "==" in line:
                parts = line.split("==")
                if len(parts) == 2:
                    packages[parts[0]] = parts[1]
    except Exception:
        packages = {}

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "git_sha": git_sha,
        "protosynth_version": "0.1.0",
        "packages": packages,
        "timestamp": None,  # Will be set by caller if needed
    }


def format_metadata_summary(metadata: Dict[str, Any]) -> str:
    """
    Format metadata into a human-readable summary.

    Args:
        metadata: Metadata dictionary from collect_run_metadata()

    Returns:
        Formatted string summary
    """
    lines = [
        f"ProtoSynth v{metadata.get('protosynth_version', 'unknown')}",
        f"Python {metadata.get('python', 'unknown')}",
        f"Platform: {metadata.get('platform', 'unknown')}",
        f"Git SHA: {metadata.get('git_sha', 'unknown')[:8]}",
    ]

    if metadata.get("timestamp"):
        lines.append(f"Timestamp: {metadata['timestamp']}")

    key_packages = ["numpy", "pytest", "black", "isort"]
    packages = metadata.get("packages", {})
    for pkg in key_packages:
        if pkg in packages:
            lines.append(f"{pkg}: {packages[pkg]}")

    return "\n".join(lines)
