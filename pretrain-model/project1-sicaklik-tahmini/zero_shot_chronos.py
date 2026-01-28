import numpy as np
import torch
from chronos import ChronosPipeline
from chronos_baslangic import get_series

train, test = get_series()

# Pretrained Chronos
pipeline = ChronosPipeline.from_pretrained(
    "amazon/chronos-t5-small",
    device_map="auto",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
)

preds = []
actuals = []
history = train.values.tolist()

for true_val in test.values:
    forecast = pipeline.predict(
        torch.tensor(history, dtype=torch.float32),
        prediction_length=1,
        num_samples=20,
    )
    pred = float(np.median(forecast.numpy()[:, 0]))
    preds.append(pred)
    actuals.append(float(true_val))
    history.append(float(true_val))

preds = np.array(preds)
actuals = np.array(actuals)

mae = np.mean(np.abs(preds - actuals))
rmse = np.sqrt(np.mean((preds - actuals) ** 2))

print("ZERO-SHOT Chronos")
print("MAE :", mae)
print("RMSE:", rmse)

# burada fine tune kullanmadan hedef verim üzerinde direkt chronos modelini uyguladım ve tahmin yaptırdım.


# --- Baseline: Naive (y(t) = y(t-1)) --- 
naive_preds = []
naive_actuals = test.values.astype(float)

prev = float(train.values[-1])
for true_val in naive_actuals:
    naive_preds.append(prev)
    prev = float(true_val)

naive_preds = np.array(naive_preds)

naive_mae = np.mean(np.abs(naive_preds - naive_actuals))
naive_rmse = np.sqrt(np.mean((naive_preds - naive_actuals) ** 2))

print("\nBASELINE Naive")
print("MAE :", naive_mae)
print("RMSE:", naive_rmse)