"""
Global and Adaptive Window training with Crossformer.
"""
import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime
from sklearn.preprocessing import StandardScaler

import config
from data_manager import load_master_data, prepare_data, get_universe_returns, get_macro_sequence
from crossformer_model import CrossformerETF
from change_point_detector import universe_adaptive_start_date
from push_results import push_daily_result


def evaluate_etf(ticker: str, returns: pd.DataFrame) -> dict:
    col = f"{ticker}_ret"
    if col not in returns.columns:
        return {}
    ret_series = returns[col].dropna()
    if len(ret_series) < 5:
        return {}
    ann_return = ret_series.mean() * config.TRADING_DAYS_PER_YEAR
    ann_vol = ret_series.std() * np.sqrt(config.TRADING_DAYS_PER_YEAR)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0
    cum = (1 + ret_series).cumprod()
    rolling_max = cum.expanding().max()
    drawdown = (cum - rolling_max) / rolling_max
    max_dd = drawdown.min()
    hit_rate = (ret_series > 0).mean()
    cum_return = (1 + ret_series).prod() - 1
    return {
        "ann_return": ann_return, "ann_vol": ann_vol, "sharpe": sharpe,
        "max_dd": max_dd, "hit_rate": hit_rate, "cum_return": cum_return,
        "n_days": len(ret_series)
    }


def create_sequences(macro_df: pd.DataFrame, returns_df: pd.DataFrame, window: int):
    """Create (macro_window, next_day_returns) pairs."""
    macro_data = macro_df.values
    returns_data = returns_df.values
    X, y = [], []
    for i in range(len(macro_data) - window):
        X.append(macro_data[i:i+window])
        y.append(returns_data[i+window])
    return np.array(X), np.array(y)


def train_crossformer(model, train_loader, val_loader, epochs, lr, patience, device):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                pred = model(X_batch)
                loss = criterion(pred, y_batch)
                val_loss += loss.item() * X_batch.size(0)
        val_loss /= len(val_loader.dataset)

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1:3d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"    Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(best_state)
    return model


def train_global(universe: str, returns: pd.DataFrame, macro_df: pd.DataFrame) -> dict:
    print(f"\n--- Global Training: {universe} ---")
    tickers = [col.replace("_ret", "") for col in returns.columns]
    total_days = len(returns)
    train_end = int(total_days * config.TRAIN_RATIO)
    val_end = train_end + int(total_days * config.VAL_RATIO)

    # Ensure validation set is non‑empty
    if val_end <= train_end:
        train_end = int(total_days * 0.7)
        val_end = int(total_days * 0.8)

    train_ret = returns.iloc[:train_end]
    val_ret = returns.iloc[train_end:val_end]
    test_ret = returns.iloc[val_end:]

    train_macro = macro_df.iloc[:train_end]
    val_macro = macro_df.iloc[train_end:val_end]
    test_macro = macro_df.iloc[val_end:]

    if len(val_ret) < 5:
        print("  Validation set too small; merging train and val.")
        train_ret = pd.concat([train_ret, val_ret])
        train_macro = pd.concat([train_macro, val_macro])
        val_ret = test_ret.iloc[:len(test_ret)//2]
        val_macro = test_macro.iloc[:len(test_macro)//2]
        test_ret = test_ret.iloc[len(val_ret):]
        test_macro = test_macro.iloc[len(val_ret):]

    # Scale macro features
    scaler = StandardScaler()
    train_macro_scaled = scaler.fit_transform(train_macro)
    val_macro_scaled = scaler.transform(val_macro) if len(val_macro) > 0 else np.empty((0, len(config.MACRO_FEATURES)))
    test_macro_scaled = scaler.transform(test_macro) if len(test_macro) > 0 else np.empty((0, len(config.MACRO_FEATURES)))

    X_train, y_train = create_sequences(pd.DataFrame(train_macro_scaled), train_ret, config.LOOKBACK_WINDOW)
    X_val, y_val = create_sequences(pd.DataFrame(val_macro_scaled), val_ret, config.LOOKBACK_WINDOW) if len(val_ret) > config.LOOKBACK_WINDOW else (np.empty((0, config.LOOKBACK_WINDOW, len(config.MACRO_FEATURES))), np.empty((0, len(tickers))))
    X_test, _ = create_sequences(pd.DataFrame(test_macro_scaled), test_ret, config.LOOKBACK_WINDOW) if len(test_ret) > config.LOOKBACK_WINDOW else (np.empty((0, config.LOOKBACK_WINDOW, len(config.MACRO_FEATURES))), np.empty((0, len(tickers))))

    if len(X_train) == 0:
        print("  No training sequences available. Skipping.")
        return {"ticker": None, "metrics": {}}

    if len(X_val) == 0:
        # Use a portion of training as validation
        split = max(1, int(len(X_train) * 0.2))
        X_val, y_val = X_train[-split:], y_train[-split:]
        X_train, y_train = X_train[:-split], y_train[:-split]

    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE)

    model = CrossformerETF(
        n_vars=len(config.MACRO_FEATURES),
        seg_len=config.SEGMENT_LEN,
        d_model=config.D_MODEL,
        n_heads=config.NUM_HEADS,
        n_layers=config.NUM_ENCODER_LAYERS,
        n_etfs=len(tickers),
        dropout=config.DROPOUT
    ).to(config.DEVICE)

    model = train_crossformer(model, train_loader, val_loader,
                              config.EPOCHS, config.LEARNING_RATE, config.PATIENCE, config.DEVICE)

    model.eval()
    with torch.no_grad():
        if len(X_test) > 0:
            X_test_t = torch.tensor(X_test[-1:], dtype=torch.float32).to(config.DEVICE)
            pred_returns = model(X_test_t).cpu().numpy().squeeze()
        else:
            X_train_t = torch.tensor(X_train[-1:], dtype=torch.float32).to(config.DEVICE)
            pred_returns = model(X_train_t).cpu().numpy().squeeze()

    best_idx = np.argmax(pred_returns)
    best_ticker = tickers[best_idx]
    pred_return = float(pred_returns[best_idx])
    all_pred_returns = {tickers[i]: float(pred_returns[i]) for i in range(len(tickers))}

    metrics = evaluate_etf(best_ticker, test_ret) if len(test_ret) > 0 else {}
    print(f"  Selected ETF: {best_ticker}, Predicted Return: {pred_return*100:.2f}%")
    return {
        "ticker": best_ticker,
        "pred_return": pred_return,
        "all_pred_returns": all_pred_returns,
        "metrics": metrics,
        "test_start": test_ret.index[0].strftime("%Y-%m-%d") if len(test_ret) else "",
        "test_end": test_ret.index[-1].strftime("%Y-%m-%d") if len(test_ret) else "",
    }


def train_adaptive(universe: str, returns: pd.DataFrame, macro_df: pd.DataFrame) -> dict:
    print(f"\n--- Adaptive Training: {universe} ---")
    tickers = [col.replace("_ret", "") for col in returns.columns]
    cp_date = universe_adaptive_start_date(returns)
    print(f"  Adaptive window starts: {cp_date.date()}")

    end_date = returns.index[-1] - pd.Timedelta(days=config.MIN_TEST_DAYS)
    if end_date <= cp_date:
        end_date = returns.index[-1] - pd.Timedelta(days=10)
    train_mask = (returns.index >= cp_date) & (returns.index <= end_date)
    train_ret = returns.loc[train_mask]
    test_ret = returns.loc[returns.index > end_date]

    train_macro = macro_df.loc[train_mask]
    test_macro = macro_df.loc[returns.index > end_date]

    if len(train_ret) < config.MIN_TRAIN_DAYS:
        print("  Insufficient training days. Falling back to global.")
        return train_global(universe, returns, macro_df)

    scaler = StandardScaler()
    train_macro_scaled = scaler.fit_transform(train_macro)
    test_macro_scaled = scaler.transform(test_macro) if len(test_macro) > 0 else np.empty((0, len(config.MACRO_FEATURES)))

    X_train, y_train = create_sequences(pd.DataFrame(train_macro_scaled), train_ret, config.LOOKBACK_WINDOW)
    X_test, _ = create_sequences(pd.DataFrame(test_macro_scaled), test_ret, config.LOOKBACK_WINDOW) if len(test_ret) > config.LOOKBACK_WINDOW else (np.empty((0, config.LOOKBACK_WINDOW, len(config.MACRO_FEATURES))), np.empty((0, len(tickers))))

    if len(X_train) == 0:
        print("  No training sequences. Falling back to global.")
        return train_global(universe, returns, macro_df)

    val_size = max(10, int(len(X_train) * 0.2))
    X_val, y_val = X_train[-val_size:], y_train[-val_size:]
    X_train, y_train = X_train[:-val_size], y_train[:-val_size]

    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE)

    model = CrossformerETF(
        n_vars=len(config.MACRO_FEATURES),
        seg_len=config.SEGMENT_LEN,
        d_model=config.D_MODEL,
        n_heads=config.NUM_HEADS,
        n_layers=config.NUM_ENCODER_LAYERS,
        n_etfs=len(tickers),
        dropout=config.DROPOUT
    ).to(config.DEVICE)

    model = train_crossformer(model, train_loader, val_loader,
                              config.EPOCHS, config.LEARNING_RATE, config.PATIENCE, config.DEVICE)

    model.eval()
    with torch.no_grad():
        if len(X_test) > 0:
            X_test_t = torch.tensor(X_test[-1:], dtype=torch.float32).to(config.DEVICE)
            pred_returns = model(X_test_t).cpu().numpy().squeeze()
        else:
            X_train_t = torch.tensor(X_train[-1:], dtype=torch.float32).to(config.DEVICE)
            pred_returns = model(X_train_t).cpu().numpy().squeeze()

    best_idx = np.argmax(pred_returns)
    best_ticker = tickers[best_idx]
    pred_return = float(pred_returns[best_idx])
    all_pred_returns = {tickers[i]: float(pred_returns[i]) for i in range(len(tickers))}

    metrics = evaluate_etf(best_ticker, test_ret) if len(test_ret) > 0 else {}
    lookback = (returns.index[-1] - cp_date).days
    print(f"  Selected ETF: {best_ticker}, Predicted Return: {pred_return*100:.2f}%")
    return {
        "ticker": best_ticker,
        "pred_return": pred_return,
        "all_pred_returns": all_pred_returns,
        "adaptive_window": lookback,
        "change_point_date": cp_date.strftime("%Y-%m-%d"),
        "metrics": metrics,
        "test_start": test_ret.index[0].strftime("%Y-%m-%d") if len(test_ret) else "",
        "test_end": test_ret.index[-1].strftime("%Y-%m-%d") if len(test_ret) else "",
    }


def run_training():
    print("Loading data...")
    df_raw = load_master_data()
    df = prepare_data(df_raw)
    macro_df = get_macro_sequence(df)

    all_results = {}
    for universe in ["fi", "equity", "combined"]:
        print(f"\n{'='*50}\nProcessing {universe.upper()}\n{'='*50}")
        returns = get_universe_returns(df, universe)
        if returns.empty:
            continue
        global_res = train_global(universe, returns, macro_df)
        adaptive_res = train_adaptive(universe, returns, macro_df)
        all_results[universe] = {"global": global_res, "adaptive": adaptive_res}
    return all_results


if __name__ == "__main__":
    output = run_training()
    if config.HF_TOKEN:
        push_daily_result(output)
    else:
        print("HF_TOKEN not set.")
