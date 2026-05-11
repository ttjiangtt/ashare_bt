@echo off
:: Prevent sleep while downloading
powercfg /change standby-timeout-ac 0
powercfg /change hibernate-timeout-ac 0
call C:\Users\ttjia\.conda\condabin\conda.bat activate ashare_bt
python C:\Users\ttjia\OneDrive\Work\ashare\signals\live_signals.py  --min_consistency 1 --side BUY >> C:\...\download_log.txt 2>&1