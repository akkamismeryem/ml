import numpy as np
import pandas as pd

from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
from chronos_baslangic import get_series  

PRED_LEN = 1  # 1 ay sonrası tahmin

def to_tsdf(series: pd.Series, item_id="station_1"):
    """
    pandas Series -> AutoGluon TimeSeriesDataFrame
    """
    df = pd.DataFrame({
        "item_id": item_id,
        "timestamp": series.index,
        "target": series.values.astype(float),
    })
    return TimeSeriesDataFrame(df)

def walk_forward_mae_rmse(predictor, train_s: pd.Series, test_s: pd.Series, model_name: str):
    """
    Walk-forward değerlendirme (senin önceki Chronos ve LSTM ile birebir aynı mantık)
    """
    preds = []
    actuals = test_s.values.astype(float)

    history = train_s.copy()

    for ts, true_val in zip(test_s.index, actuals):
        hist_tsdf = to_tsdf(history)
        fcst = predictor.predict(hist_tsdf, model=model_name)

        # prediction_length=1 olduğu için ilgili timestamp ts
        row = fcst.loc[(slice(None), ts), :]

        if "mean" in row.columns:
            pred = float(row["mean"].iloc[0])
        elif "0.5" in row.columns:
            pred = float(row["0.5"].iloc[0])
        else:
            pred = float(row.select_dtypes(include="number").iloc[0, 0])

        preds.append(pred)
        history.loc[ts] = true_val  # walk-forward

    preds = np.array(preds, dtype=float)
    mae = float(np.mean(np.abs(preds - actuals)))
    rmse = float(np.sqrt(np.mean((preds - actuals) ** 2)))
    return mae, rmse

def main():
    # Aynı split (zero-shot, LSTM, TCN ile birebir aynı)
    train, test = get_series(test_ratio=0.2)

    # Chronos-2 Fine-tune
    predictor = TimeSeriesPredictor(
        prediction_length=PRED_LEN,
        target="target",
    ).fit(
        train_data=to_tsdf(train),
        hyperparameters={
            "Chronos2": [
                {
                    "fine_tune": True,
                    "ag_args": {"name_suffix": "FineTuned"}
                }
            ]
        },
        time_limit=300,        
        enable_ensemble=False,
        verbosity=2
    )

    mae, rmse = walk_forward_mae_rmse(
        predictor,
        train,
        test,
        model_name="Chronos2FineTuned"
    )

    print("\nChronos-2 Fine-Tuned SONUÇLAR")
    print("MAE :", mae)
    print("RMSE:", rmse)

if __name__ == "__main__":
    main()
