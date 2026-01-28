# Climate Time Series Forecasting with Pretrained Models

This repository investigates climate time series data (such as temperature and precipitation)
using **pretrained time-series models (Chronos / Chronos-2)** through
transfer learning, zero-shot inference, and fine-tuning approaches.

## Methods

### Baseline Models
- Naive forecasting (y(t) = y(t−1))
- LSTM (trained from scratch)
- TCN (trained from scratch)

### Pretrained Models
- Chronos (zero-shot inference)
- Chronos-2 (fine-tuned on target data)

## Experimental Scenarios

### 1. Temperature Forecasting
- Univariate time series
- Zero-shot Chronos
- Fine-tuned Chronos-2
- Hyperparameter optimization (time_limit)

### 2. Precipitation Forecasting
- Univariate zero-shot Chronos
- Multivariate Chronos-2 fine-tuning
  - Past covariates: temperature, relative humidity, evaporation
  - Known covariates: monthly sinusoidal seasonality (sin/cos)

## Usage

```bash
python chronos_baslangic.py
python zero_shot_chronos.py
python fine_tune_chronos2.py
