@echo off
cd /d D:\EcoSort-AI
D:\EcoSort-AI\venv\Scripts\python.exe -u train.py --epochs 30 --batch-size 32 --num-workers 0 > training_run.log 2>&1
