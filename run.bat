@echo off
CHCP 65001

uv run python -m src.main --task=all --data_path=./datasets/data_test.mat

pause
