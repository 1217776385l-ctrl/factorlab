#!/bin/bash
# 全流程:主研究管线 → ML → 报告(需要先有数据:fetch_bs2.py 或 update_daily.py)
cd "$(dirname "$0")"
python3 -W ignore run_pipeline.py && \
python3 -W ignore ml_pipeline.py && \
python3 -W ignore gen_report.py && \
echo "=== ALL COMPLETE ==="
