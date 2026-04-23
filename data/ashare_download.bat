@echo off
call C:\Users\ttjia\.conda\condabin\conda.bat activate ashare_bt
python C:\Users\ttjia\OneDrive\Work\ashare\data\download_all.py --root C:\Users\ttjia\OneDrive\Work\ashare\market_data >> C:\Users\ttjia\OneDrive\Work\ashare\download_log.txt 2>&1