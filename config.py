"""
Configuration for P2-ETF-MACRO-CROSSFORMER.
"""
import os

# Hugging Face configuration
HF_INPUT_DATASET = "P2SAMAPA/fi-etf-macro-signal-master-data"
HF_INPUT_FILE = "master_data.parquet"
HF_OUTPUT_DATASET = "P2SAMAPA/p2-etf-macro-crossformer-results"
HF_TOKEN = os.environ.get("HF_TOKEN")

# Universes
FI_COMMODITY_TICKERS = ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV"]
EQUITY_TICKERS = ["QQQ", "IWM", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "GDX", "IWM", "IWF", "XSD", "XBI", "XME"]
COMBINED_TICKERS = FI_COMMODITY_TICKERS + EQUITY_TICKERS

BENCHMARK_FI = "AGG"
BENCHMARK_EQ = "SPY"

# Macro indicators to use as features
MACRO_FEATURES = ["VIX", "DXY", "T10Y2Y", "TBILL_3M", "IG_SPREAD", "HY_SPREAD"]

# Training parameters
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
MIN_TRAIN_DAYS = 252 * 2
MIN_TEST_DAYS = 63
TRADING_DAYS_PER_YEAR = 252

# Change Point Detection (for adaptive window)
CP_PENALTY = 3.0
CP_MODEL = "l2"
CP_MIN_DAYS_BETWEEN = 20
CP_CONSENSUS_FRACTION = 0.5
ADAPTIVE_MAX_LOOKBACK = 252

# Crossformer hyperparameters
LOOKBACK_WINDOW = 60          # number of past macro days to use as input
SEGMENT_LEN = 6               # segment length for DSW embedding
D_MODEL = 64                  # model dimension
NUM_HEADS = 4                 # number of attention heads
NUM_ENCODER_LAYERS = 2        # number of Crossformer encoder layers
DROPOUT = 0.1
LEARNING_RATE = 0.001
EPOCHS = 100
PATIENCE = 20
BATCH_SIZE = 32
DEVICE = "cpu"
