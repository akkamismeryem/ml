import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def _load_excel(filename, value_col, new_name):
    path = os.path.join(BASE_DIR, filename)
    df = pd.read_excel(path)
    df = df.rename(columns={"Yıl/Ay": "yil_ay", value_col: new_name})
    df["tarih"] = pd.to_datetime(df["yil_ay"].astype(str), format="%Y/%m")
    df = df.sort_values("tarih").set_index("tarih")
    return df[[new_name]].astype("float32")

def load_merged_dataframe():
    yagis = _load_excel("yil_ay_yagis_miktari_duzenli.xlsx", "Yağış Miktarı", "yagis")
    sic   = _load_excel("yil_ay_ortalama_sicaklik_duzenli.xlsx", "Ortalama Sıcaklık", "sicaklik")
    nem   = _load_excel("yil_ay_nispi_nem_duzenli.xlsx", "Nispi Nem", "nem")
    buh   = _load_excel("yil_ay_toplam_buharlasma_duzenli.xlsx", "Toplam Buharlaşma", "buharlasma")

    # 1) Hepsini birleştir (inner yerine outer alıyoruz)
    df = yagis.join([sic, nem, buh], how="outer")

    # 2) Düzenli aylık index üret
    full_index = pd.date_range(df.index.min(), df.index.max(), freq="MS")
    df = df.reindex(full_index)

    # 3) Covariate eksiklerini doldurduk (geçmişten ileri doldurma)
    # (sıcaklık/nem/buharlaşma bazı aylarda eksikse modeli bozmasın)
    for c in ["sicaklik", "nem", "buharlasma"]:
        df[c] = df[c].interpolate(limit_direction="both")

    # 4) Target (yağış) eksikse o ayı kullanamayız -> drop
    df = df.dropna(subset=["yagis"])

    # 5) Ay sin/cos (known covariates)
    m = df.index.month.astype(int)
    df["month_sin"] = np.sin(2 * np.pi * m / 12).astype("float32")
    df["month_cos"] = np.cos(2 * np.pi * m / 12).astype("float32")

    df.index.name = "tarih"
    return df

def get_train_test_df(test_ratio=0.2):
    df = load_merged_dataframe()
    test_size = int(len(df) * test_ratio)
    train_df = df.iloc[:-test_size].copy()
    test_df  = df.iloc[-test_size:].copy()
    return train_df, test_df
