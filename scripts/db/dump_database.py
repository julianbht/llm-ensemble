#!/usr/bin/env python3
"""Dump the entire database to a compressed SQL file.

Uses pg_dump to create a full backup of the database.
"""

import os
import subprocess
import sys
from pathlib import Path

from llm_ensemble.libs.runtime.env import load_runtime_config

DUMP_OUTPUT_PATH = Path("./artifacts/backups/backup.sql.gz")

# Load runtime configuration (DATABASE_URL, etc.)
load_runtime_config()


def main():
    # Get database URL from environment
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        print("Error: DATABASE_URL not set in environment", file=sys.stderr)
        sys.exit(1)

    output_path = DUMP_OUTPUT_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build pg_dump command
    if output_path.suffix == ".gz":
        # Compressed output
        print(f"Dumping database to {output_path} (compressed)...")
        cmd = f"pg_dump {database_url} | gzip > {output_path}"
        shell = True
    else:
        # Plain SQL output
        print(f"Dumping database to {output_path}...")
        cmd = ["pg_dump", database_url, "-f", str(output_path)]
        shell = False

    try:
        subprocess.run(cmd, check=True, shell=shell)
        print(f"Database dump complete: {output_path}")

        # Print file size
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"File size: {size_mb:.2f} MB")

    except subprocess.CalledProcessError as e:
        print(f"Error dumping database: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
