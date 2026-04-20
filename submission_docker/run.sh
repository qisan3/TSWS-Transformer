#!/bin/sh
echo "Starting PHM2025 docker image run..."
python main.py

if [ -f /work/result.csv ]; then
    echo "Results saved at /work/result.csv"
else
    echo "result.csv not found, check logs."
    exit 1
fi