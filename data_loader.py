"""
data_loader.py
--------------
Excel metadata okuma ve dataset klasörü tarama fonksiyonları.
Grup_05_MetaVeri.xlsx sütun yapısına göre yazılmıştır:
  Dosya_Adi, Denek_ID, Cinsiyet, Yas, Duygu,
  Cumle_No, Kayit_Cihazi, ORTAM, gürültü seyiyesi
"""

import os
import glob
import pandas as pd


# ──────────────────────────────────────────────
# TEK GRUP EXCEL OKUMA
# ──────────────────────────────────────────────

def load_single_excel(excel_path):
    """
    Tek bir grubun metadata Excel'ini okur.
    Döndürür: DataFrame (boş satırlar temizlenmiş)
    """
    df = pd.read_excel(excel_path)
    df = df.dropna(subset=["Dosya_Adi", "Cinsiyet"])
    df = df.reset_index(drop=True)
    return df


# ──────────────────────────────────────────────
# TÜM GRUPLARI BİRLEŞTİRME
# ──────────────────────────────────────────────

def load_all_metadata(dataset_root):
    """
    Dataset/ klasörü altındaki tüm Grup_XX klasörlerini tarar,
    bulduğu tüm .xlsx dosyalarını birleştirir.

    dataset_root : Örn. "Dataset/"

    Döndürür: birleşik DataFrame + her satıra "wav_path" sütunu eklenir
    """
    excel_pattern = os.path.join(dataset_root, "**", "*.xlsx")
    excel_files   = glob.glob(excel_pattern, recursive=True)

    if not excel_files:
        return pd.DataFrame()

    dfs = []
    for ef in excel_files:
        grup_klasoru = os.path.dirname(ef)
        df = load_single_excel(ef)
        # Her dosyanın tam yolunu ekle
        df["wav_path"]   = df["Dosya_Adi"].apply(
            lambda fn: os.path.join(grup_klasoru, fn)
        )
        df["grup_klasor"] = os.path.basename(grup_klasoru)
        dfs.append(df)

    merged = pd.concat(dfs, ignore_index=True)
    return merged


# ──────────────────────────────────────────────
# TEK KLASÖR MODU (EXCEL + SES AYNI YERDE)
# ──────────────────────────────────────────────

def load_single_group(excel_path, wav_dir=None):
    """
    Tek grup klasöründen çalışmak için.
    wav_dir belirtilmezse Excel ile aynı klasör varsayılır.
    """
    df = load_single_excel(excel_path)
    if wav_dir is None:
        wav_dir = os.path.dirname(excel_path)

    df["wav_path"] = df["Dosya_Adi"].apply(
        lambda fn: os.path.join(wav_dir, fn)
    )
    return df


# ──────────────────────────────────────────────
# DOSYA VARLIK KONTROLÜ
# ──────────────────────────────────────────────

def check_files(df):
    """
    DataFrame'deki wav_path sütununu kontrol eder.
    Döndürür: (mevcut sayısı, eksik dosya listesi)
    """
    existing = df["wav_path"].apply(os.path.exists)
    missing  = df[~existing]["Dosya_Adi"].tolist()
    return int(existing.sum()), missing


# ──────────────────────────────────────────────
# VERİ SETİ ÖZET İSTATİSTİKLERİ
# ──────────────────────────────────────────────

def dataset_summary(df):
    """
    Basit özet istatistikler döndürür (dict).
    """
    summary = {
        "toplam_kayit"   : len(df),
        "cinsiyet_dagilim": df["Cinsiyet"].value_counts().to_dict(),
        "yas_aralik"     : (int(df["Yas"].min()), int(df["Yas"].max())),
        "duygu_dagilim"  : df["Duygu"].value_counts().to_dict() if "Duygu" in df.columns else {},
    }
    return summary
