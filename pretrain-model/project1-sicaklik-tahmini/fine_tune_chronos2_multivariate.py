import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import pandas as pd
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

from yagis_multivariate_baslangic import get_train_test_df

PRED_LEN = 1
KNOWN_COVS = ["month_sin", "month_cos"]

def to_tsdf(df: pd.DataFrame, item_id="station_1"):
    out = df.copy()
    out["item_id"] = item_id
    out["timestamp"] = out.index
    out = out.rename(columns={"yagis": "target"})
    cols = ["item_id", "timestamp", "target", "sicaklik", "nem", "buharlasma", "month_sin", "month_cos"]
    return TimeSeriesDataFrame(out[cols])

def _extract_timestamps(tsdf: TimeSeriesDataFrame):
    # 1) MultiIndex -> son level timestamp
    idx = tsdf.index
    if isinstance(idx, pd.MultiIndex):
        return pd.to_datetime(idx.get_level_values(-1))

    # 2) timestamp sütunu varsa onu kullan
    if hasattr(tsdf, "columns") and "timestamp" in tsdf.columns:
        return pd.to_datetime(tsdf["timestamp"].values)

    # 3) index datetime ise direkt kullan
    if isinstance(idx, (pd.DatetimeIndex,)):
        return pd.to_datetime(idx)

    raise RuntimeError("Future dataframe'de timestamp bulunamadı (ne indexte ne sütunda).")

def add_month_covariates(future_tsdf: TimeSeriesDataFrame) -> TimeSeriesDataFrame:
    timestamps = _extract_timestamps(future_tsdf)
    m = pd.Series(timestamps).dt.month.astype(int).values

    future_tsdf = future_tsdf.copy()
    future_tsdf["month_sin"] = np.sin(2 * np.pi * m / 12).astype("float32")
    future_tsdf["month_cos"] = np.cos(2 * np.pi * m / 12).astype("float32")
    return future_tsdf

def _first_forecast_timestamp(fcst: TimeSeriesDataFrame):
    idx = fcst.index
    if isinstance(idx, pd.MultiIndex):
        return idx.get_level_values(-1)[0]
    if "timestamp" in fcst.columns:
        return pd.to_datetime(fcst["timestamp"].iloc[0])
    return idx[0]

def walk_forward_mae_rmse(predictor, train_df, test_df, model_name):
    preds = []
    actuals = test_df["yagis"].values.astype(float)
    history = train_df.copy()

    for ts, true_val in zip(test_df.index, actuals):
        data_tsdf = to_tsdf(history)

        future = predictor.make_future_data_frame(data_tsdf)
        future = add_month_covariates(future)

        fcst = predictor.predict(
            data_tsdf,
            model=model_name,
            known_covariates=future,
        )

        f_ts = _first_forecast_timestamp(fcst)

        # fcst MultiIndex ise satırı çek
        if isinstance(fcst.index, pd.MultiIndex):
            row = fcst.loc[(slice(None), f_ts), :]
        else:
            row = fcst.iloc[[0]]

        if "mean" in row.columns:
            pred = float(row["mean"].iloc[0])
        elif "0.5" in row.columns:
            pred = float(row["0.5"].iloc[0])
        else:
            pred = float(row.select_dtypes(include="number").iloc[0, 0])

        preds.append(pred)
        history.loc[ts] = test_df.loc[ts]  # walk-forward

    preds = np.array(preds, dtype=float)
    mae = float(np.mean(np.abs(preds - actuals)))
    rmse = float(np.sqrt(np.mean((preds - actuals) ** 2)))
    return mae, rmse

def main():
    train_df, test_df = get_train_test_df(test_ratio=0.2)

    predictor = TimeSeriesPredictor(
        prediction_length=PRED_LEN,
        target="target",
        known_covariates_names=KNOWN_COVS,
        freq="MS",
    ).fit(
        train_data=to_tsdf(train_df),
        hyperparameters={"Chronos2": [{"fine_tune": True, "ag_args": {"name_suffix": "MV_FT"}}]},
        time_limit=600,
        enable_ensemble=False,
        verbosity=2,
    )

    mn = predictor.model_names() if callable(getattr(predictor, "model_names", None)) else predictor.model_names
    model_name = mn[0]

    mae, rmse = walk_forward_mae_rmse(predictor, train_df, test_df, model_name)

    print("\nYAĞIŞ - Multivariate Chronos2 Fine-Tuned")
    print("Model:", model_name)
    print("MAE :", mae)
    print("RMSE:", rmse)

if __name__ == "__main__":
    main()
