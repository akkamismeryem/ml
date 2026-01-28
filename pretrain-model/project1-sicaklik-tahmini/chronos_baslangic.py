import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
excel_path = os.path.join(BASE_DIR, "yil_ay_ortalama_sicaklik_duzenli.xlsx")

print("Excel yolu:", excel_path)

df = pd.read_excel(excel_path)

# burada kolon adlarını düzenliyorum
df = df.rename(columns={
    "Yıl/Ay": "yil_ay",
    "Ortalama Sıcaklık": "sicaklik"
})

# değerlerimizi chronos modeli için datetime'a çeviriyorum
df["tarih"] = pd.to_datetime(df["yil_ay"].astype(str), format="%Y/%m")

# sıralama/indezx
df = df.sort_values("tarih").set_index("tarih")

# hedef seri 
y = df["sicaklik"].astype("float32")

# print(df.columns)
print(df.head())

y = y.dropna()

print("NaN temizlendikten sonra:")
print("Başlangıç:", y.index.min(), "Bitiş:", y.index.max())
print("Toplam uzunluk:", len(y))
print("Eksik değer var mı?:", y.isna().any())

# # eksik değerler / sadece 2025 in mayıstan sonrası 
# print(df[df["sicaklik"].isna()])

# train / test (%20)
# test_size = int(len(y) * 0.2)
# train = y.iloc[:-test_size]
# test = y.iloc[-test_size:]

# print("Train:", train.index.min(), "→", train.index.max(), "(", len(train), ")")
# print("Test :", test.index.min(), "→", test.index.max(), "(", len(test), ")")

def get_series(test_ratio=0.2):
    test_size = int(len(y) * test_ratio)
    train = y.iloc[:-test_size]
    test = y.iloc[-test_size:]
    return train, test