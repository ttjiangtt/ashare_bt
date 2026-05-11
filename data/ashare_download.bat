@echo off
:: Prevent sleep while downloading
powercfg /change standby-timeout-ac 0
powercfg /change hibernate-timeout-ac 0
call C:\Users\ttjia\.conda\condabin\conda.bat activate ashare_bt
python C:\Users\ttjia\OneDrive\Work\ashare\data\download_all.py --root C:\Users\ttjia\OneDrive\Work\ashare\market_data >> C:\Users\ttjia\OneDrive\Work\ashare\download_log.txt 2>&1 --workers 6 --throttle 0.1