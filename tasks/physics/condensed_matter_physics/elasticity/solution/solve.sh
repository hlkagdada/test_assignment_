#!/usr/bin/env bash
set -euo pipefail
mkdir -p /app/output
python /solution/elastic_pipeline.py > /app/output/solution_stdout.json
python /solution/run_pipeline_summary.py
