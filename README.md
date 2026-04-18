# P2 ETF MACRO-CROSSFORMER Engine

**Cross‑Dimension Attention for Macro‑Driven ETF Prediction.**

[![GitHub Actions](https://github.com/P2SAMAPA/P2-ETF-MACRO-CROSSFORMER/actions/workflows/daily_run.yml/badge.svg)](https://github.com/P2SAMAPA/P2-ETF-MACRO-CROSSFORMER/actions/workflows/daily_run.yml)

## Overview

This engine is specifically designed to model cross‑macro interactions. Instead of treating VIX, DXY, HY spreads, etc., as isolated scalars, the MACRO‑CROSSFORMER uses a Crossformer model to capture the joint, co‑evolving dynamics of the entire macro system. It learns how these variables influence each other and ultimately drive ETF returns.

**Key Features:**
- **Cross‑Dimension Attention**: Models dependencies both across time and across macro variables.
- **Router Mechanism**: Efficiently captures cross‑variable relationships without quadratic complexity.
- **Three Universes**: FI/Commodities, Equity Sectors, and Combined.
- **Global & Adaptive Training**: Fixed 80/10/10 split and change‑point‑derived adaptive windows.

## Data

- **Input**: `P2SAMAPA/fi-etf-macro-signal-master-data` (master_data.parquet)
- **Output**: `P2SAMAPA/p2-etf-macro-crossformer-results`

## Usage

```bash
pip install -r requirements.txt
python trainer.py           # Runs training and pushes to HF
streamlit run streamlit_app.py
Configuration
All parameters are in config.py:

MACRO_FEATURES: list of macro indicators used (VIX, DXY, etc.)

LOOKBACK_WINDOW: number of past macro days as input (default 60)

SEGMENT_LEN, D_MODEL, NUM_HEADS: Crossformer architecture settings.
