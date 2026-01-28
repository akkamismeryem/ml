import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import pandas as pd

from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
from chronos_baslangic import get_series

PRED_LEN = 1  # 1 ay sonrası

def to_tsdf(series: pd.Series, item_id="station_1"):
    df = pd.DataFrame({
        "item_id": item_id,
        "timestamp": series.index,
        "target": series.values.astype(float),
    })
    return TimeSeriesDataFrame(df)

def walk_forward_mae_rmse(predictor, train_s, test_s, model_name):
    preds = []
    actuals = test_s.values.astype(float)
    history = train_s.copy()

    for ts, true_val in zip(test_s.index, actuals):
        fcst = predictor.predict(to_tsdf(history), model=model_name)
        row = fcst.loc[(slice(None), ts), :]

        if "mean" in row.columns:
            pred = float(row["mean"].iloc[0])
        elif "0.5" in row.columns:
            pred = float(row["0.5"].iloc[0])
        else:
            pred = float(row.select_dtypes(include="number").iloc[0, 0])

        preds.append(pred)
        history.loc[ts] = true_val

    preds = np.array(preds)
    mae = float(np.mean(np.abs(preds - actuals)))
    rmse = float(np.sqrt(np.mean((preds - actuals) ** 2)))
    return mae, rmse

def get_models(predictor):
    """AutoGluon sürüm farklarına dayanıklı model listesi al."""
    if hasattr(predictor, "model_names"):
        mn = predictor.model_names
        return mn() if callable(mn) else mn
    raise RuntimeError("Bu AutoGluon sürümünde model isimleri alınamadı.")

def train_and_eval(train, test, time_limit):
    predictor = TimeSeriesPredictor(
        prediction_length=PRED_LEN,
        target="target",
    ).fit(
        train_data=to_tsdf(train),
        hyperparameters={
            "Chronos2": [
                {"fine_tune": True, "ag_args": {"name_suffix": f"T{time_limit}"}}
            ]
        },
        time_limit=time_limit,
        enable_ensemble=False,
        verbosity=2,
    )

    models = get_models(predictor)
    # Bu senaryoda tek model eğitiyoruz zaten:
    model_name = models[0]

    mae, rmse = walk_forward_mae_rmse(predictor, train, test, model_name)
    return model_name, mae, rmse

def main():
    train, test = get_series(test_ratio=0.2)

    time_limits = [300, 600]  # CPU'da daha makul; istersen 900 ekleriz
    results = []

    for tl in time_limits:
        print(f"\n=== Fine-tune denemesi: time_limit={tl} ===")
        model_name, mae, rmse = train_and_eval(train, test, tl)
        results.append({
            "time_limit": tl,
            "model": model_name,
            "MAE": mae,
            "RMSE": rmse
        })
        print(f"→ {model_name} | MAE={mae:.4f} | RMSE={rmse:.4f}")

    df = pd.DataFrame(results).sort_values("MAE")
    print("\n=== HİPERPARAMETRE OPTİMİZASYONU SONUÇLARI ===")
    print(df.to_string(index=False))

    best = df.iloc[0]
    print("\nEN İYİ AYAR:")
    print(f"time_limit={int(best['time_limit'])} | model={best['model']} | MAE={best['MAE']:.4f} | RMSE={best['RMSE']:.4f}")

if __name__ == "__main__":
    main()
