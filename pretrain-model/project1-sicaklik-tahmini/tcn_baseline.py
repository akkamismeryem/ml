import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler

from chronos_baslangic import get_series 

LOOKBACK = 12  # 12 ay -> 1 ay sonrası

def make_supervised(series_1d: np.ndarray, lookback: int):
    X, y = [], []
    for i in range(lookback, len(series_1d)):
        X.append(series_1d[i - lookback:i])
        y.append(series_1d[i])
    X = np.array(X).reshape(-1, lookback, 1)
    y = np.array(y).reshape(-1, 1)
    return X, y

def walk_forward_predict(model, train_vals_raw, test_vals_raw, lookback, scaler):
    preds = []
    history = train_vals_raw.tolist()

    for true_val in test_vals_raw:
        window = np.array(history[-lookback:], dtype=np.float32).reshape(-1, 1)
        window_scaled = scaler.transform(window).reshape(1, lookback, 1)

        pred_scaled = model.predict(window_scaled, verbose=0)[0, 0]
        pred = scaler.inverse_transform([[pred_scaled]])[0, 0]

        preds.append(float(pred))
        history.append(float(true_val))

    return np.array(preds, dtype=np.float32)

def build_tcn(lookback: int):
    """
    Basit TCN: causal Conv1D + dilation + dropout
    """
    inputs = keras.Input(shape=(lookback, 1))
    x = inputs

    for dilation in [1, 2, 4, 8]:
        x = layers.Conv1D(
            filters=32,
            kernel_size=3,
            padding="causal",
            dilation_rate=dilation,
            activation="relu"
        )(x)
        x = layers.Dropout(0.1)(x)

    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(1)(x)

    model = keras.Model(inputs, outputs)
    return model

def main():
    tf.random.set_seed(42)
    np.random.seed(42)

    train, test = get_series(test_ratio=0.2)
    train_vals = train.values.astype(np.float32)
    test_vals = test.values.astype(np.float32)

    # Ölçekleme: sadece train ile fit
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_vals.reshape(-1, 1)).reshape(-1)

    # Supervised train set
    X_train, y_train = make_supervised(train_scaled, LOOKBACK)

    # Model
    model = build_tcn(LOOKBACK)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse"
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True
        )
    ]

    model.fit(
        X_train, y_train,
        epochs=30,
        batch_size=32,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=0
    )

    preds = walk_forward_predict(model, train_vals, test_vals, LOOKBACK, scaler)

    mae = float(np.mean(np.abs(preds - test_vals)))
    rmse = float(np.sqrt(np.mean((preds - test_vals) ** 2)))

    print("TCN (scratch) BASELINE")
    print("MAE :", mae)
    print("RMSE:", rmse)

if __name__ == "__main__":
    main()
