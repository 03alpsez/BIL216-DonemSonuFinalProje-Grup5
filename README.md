# 🎙️ Ses İşareti Analizi ve Duygu Sınıflandırma
**Grup 05 | BIL216 Sinyaller ve Sistemler | 2025-2026 Bahar Dönemi | Final Proje – Faz 3**

---

## 📋 İçindekiler
1. [Proje Hakkında](#proje-hakkında)
2. [Proje Yapısı](#proje-yapısı)
3. [Kurulum (Adım Adım)](#kurulum)
4. [Uygulamayı Çalıştırma](#uygulamayı-çalıştırma)
5. [Kullanım Kılavuzu](#kullanım-kılavuzu)
6. [Algoritma ve Yöntem](#algoritma-ve-yöntem)
7. [Faz Geçmişi ve Sonuçlar](#faz-geçmişi-ve-sonuçlar)
8. [Kullanılan Kütüphaneler](#kullanılan-kütüphaneler)
9. [Sık Karşılaşılan Hatalar](#sık-karşılaşılan-hatalar)

---

## Proje Hakkında

Bu proje, ses kayıtlarından duygu tanıma (Emotion Recognition) gerçekleştiren bir makine öğrenmesi uygulamasıdır. Streamlit tabanlı web arayüzü üzerinden çalışır.

**Tanınan duygular:** 😐 Nötr · 😊 Mutlu · 😠 Öfkeli · 😢 Üzgün · 😲 Şaşkın

**Final doğruluk oranı:** %98.4 (Faz 3, tüm gruplar, 380 kayıt)

---

## Proje Yapısı

```
Donemici_Proje/
│
├── app.py                → Streamlit arayüzü (ana dosya, buradan çalıştırılır)
├── audio_analysis.py     → Ses özellik çıkarımı (F0, ZCR, MFCC, Delta MFCC, Spectral, Chroma, Tempo)
├── classifier.py         → 4'lü Ensemble modeli (SVM + RF + XGBoost + MLP, Soft Voting)
├── data_loader.py        → Excel metadata okuma (artık kullanılmıyor, geriye dönük uyumluluk için)
├── requirements.txt      → Gerekli Python kütüphaneleri
├── README.md             → Bu dosya
│
└── venv/                 → Sanal ortam klasörü (git'e eklenmez, sen oluşturursun)
```

> ⚠️ `venv/` klasörü **paylaşılmaz**. Her bilgisayarda ayrı oluşturulması gerekir.

---

## Kurulum

> Kurulumu **bir kez** yapman yeterli. Sonraki çalıştırmalarda sadece [Uygulamayı Çalıştırma](#uygulamayı-çalıştırma) adımına geç.

### Adım 1 — Python'u kontrol et

Python **3.9 veya üzeri** gereklidir.

```bash
python --version
```

Eğer Python kurulu değilse → https://www.python.org/downloads/ adresinden indir.  
Kurulumda **"Add Python to PATH"** seçeneğini işaretlemeyi unutma.

---

### Adım 2 — Proje klasörüne git

Terminali (PowerShell veya CMD) aç ve proje klasörüne geç.  
Kendi bilgisayarındaki klasör yolunu kullan:

```bash
cd "C:\Users\KULLANICI_ADIN\Desktop\Donemici_Proje"
```

> 💡 Klasörün içine girmek için dosya gezgininde klasöre sağ tık → "Terminalde Aç" seçeneğini de kullanabilirsin.

---

### Adım 3 — Sanal ortam oluştur

```bash
python -m venv venv
```

Bu komut `venv/` adında bir klasör oluşturur. Bir kez yapman yeterli.

---

### Adım 4 — Sanal ortamı aktif et

**Windows (PowerShell):**
```bash
.\venv\Scripts\activate
```

**Windows (CMD):**
```bash
venv\Scripts\activate.bat
```

**Mac / Linux:**
```bash
source venv/bin/activate
```

Aktif olduğunda terminal satırının başında `(venv)` yazar:
```
(venv) PS C:\Users\...>
```

---

### Adım 5 — Kütüphaneleri yükle

```bash
pip install -r requirements.txt
```

> ⏳ İlk kurulumda 2-5 dakika sürebilir. İnternet bağlantısı gereklidir.

**Yüklenen kütüphaneler:** streamlit, librosa, numpy, pandas, matplotlib, scipy, scikit-learn, xgboost, openpyxl, soundfile

---

## Uygulamayı Çalıştırma

Her kullanımda şu 3 komutu çalıştır:

```bash
cd "C:\Users\KULLANICI_ADIN\Desktop\Donemici_Proje"
.\venv\Scripts\activate
streamlit run app.py
```

Tarayıcıda otomatik olarak `http://localhost:8501` açılır.

Uygulamayı durdurmak için terminalde `Ctrl + C` bas.

> 💡 Bir dahaki seferde `cd` ve `activate` adımlarını tekrarlaman gerekir, ama `pip install` gerekmez.

---

## Kullanım Kılavuzu

Uygulama iki sekmeden oluşur:

### Sekme 1 — 🎵 Tekli Ses Testi

Tek bir ses dosyasını test etmek için kullanılır.

1. **"Bir .wav dosyası yükleyin"** alanına bir `.wav` dosyası sürükle veya seç
2. Otomatik olarak analiz yapılır ve duygu tahmini gösterilir
3. Grafikleri incele: Otokorelasyon, FFT, Enerji, ZCR, MFCC, ΔMFCC

> 💡 Eğer daha önce Dataset Analizi yapıldıysa, tekli testte de eğitilmiş model kullanılır.  
> Aksi halde kural tabanlı bir sınıflandırıcı devreye girer.

---

### Sekme 2 — 📊 Dataset Analizi

Tüm grupların verilerini yükleyip toplu analiz yapmak için kullanılır.

#### Yükleme Yöntemi A — ZIP dosyası (önerilen)

1. Tüm grup klasörlerini bir araya topla
2. Seç → Sağ tık → "Sıkıştırılmış klasöre gönder" ile ZIP oluştur
3. Uygulamada **"📦 ZIP dosyası"** seçeneğini seç
4. ZIP dosyasını yükle

#### Yükleme Yöntemi B — Tek tek WAV seçimi

1. **"📁 Tek tek WAV seçimi"** seçeneğini seç
2. WAV dosyalarını yükle (Ctrl+A ile hepsini seçebilirsin)

#### Etiket nasıl belirlenir?

Excel dosyasına gerek yok! Etiket **dosya adından otomatik** okunur:

| Dosya Adı | Etiket |
|-----------|--------|
| G05_D01_E_23_**Notr**_C1.wav | 😐 Nötr |
| G03_D02_K_25_**Mutlu**_C2.wav | 😊 Mutlu |
| G01_D03_E_30_**Ofkeli**_C1.wav | 😠 Öfkeli |
| G02_D04_K_22_**Uzgun**_C2.wav | 😢 Üzgün |
| G04_D05_E_28_**Saskın**_C1.wav | 😲 Şaşkın |

#### Analizi Başlat

1. **"🎵 Ses düzeyi veri artırmayı etkinleştir"** toggle'ını kontrol et
   - **Açık:** Daha yüksek doğruluk, ~2-3 dakika sürer
   - **Kapalı:** Hızlı sonuç, biraz daha düşük doğruluk
2. **"🚀 Analizi Başlat"** butonuna bas
3. Sonuçları incele: Doğruluk, Confusion Matrix, İstatistiksel Özet, Özellik Grafikleri
4. İstersen **"⬇️ Sonuçları Excel olarak indir"** ile dışa aktar

---

## Algoritma ve Yöntem

### Özellik Çıkarımı (~80 özellik)

Her ses dosyasından şu özellikler çıkarılır:

| Özellik Grubu | Özellikler | Sayı |
|---------------|------------|------|
| Zaman Düzlemi | mean_f0, std_f0, ZCR, Energy, voiced_ratio, RMS, Tempo (BPM) | 8 |
| MFCC | 13 katsayı × mean + std | 26 |
| Delta MFCC (ΔMFCC) | 1. türev × mean + std | 26 |
| Spektral | Centroid, Bandwidth, Rolloff, Flatness (mean+std) | 8 |
| Chroma | 12 kroma bandı ortalaması | 12 |
| **Toplam** | | **~80** |

### Sınıflandırma Modeli — 4'lü Ensemble

```
Ses Dosyası
    ↓
Özellik Çıkarımı (~80 özellik)
    ↓
StandardScaler (normalizasyon)
    ↓
┌─────────────┬──────────────┬───────────┬──────────────┐
│ SVM (RBF)   │ Random Forest│  XGBoost  │  MLP (YSA)   │
│ C=1, γ=scale│  300 ağaç    │  300 iter │ 256→128→64   │
└─────────────┴──────────────┴───────────┴──────────────┘
    ↓               ↓              ↓             ↓
 Olasılık       Olasılık       Olasılık      Olasılık
    └───────────────┴──────────────┴─────────────┘
                         ↓
                   Soft Voting
              (olasılık ortalaması)
                         ↓
                  Duygu Tahmini
```

### Veri Artırma (Data Augmentation)

Az örnekli sınıflar için ham ses düzeyinde 4 farklı dönüşüm uygulanır:
- **Gürültü ekleme:** Hafif beyaz gürültü
- **Pitch Shift +0.5 ton:** Sesi yarım ton yukarı kaydır
- **Pitch Shift -0.5 ton:** Sesi yarım ton aşağı kaydır
- **Time Stretch ×0.9:** Sesi %10 yavaşlat

### Değerlendirme Yöntemi

**Stratified 5-Fold Cross-Validation:** Veri seti 5 eşit parçaya bölünür. Her seferinde 4 parça eğitim, 1 parça test olarak kullanılır. Bu işlem 5 kez tekrarlanır. Her sınıftan eşit örnek alınmasını garanti eder.

---

## Faz Geçmişi ve Sonuçlar

| Faz | Tarih | Model | Özellikler | Doğruluk |
|-----|-------|-------|------------|----------|
| Faz 1 | Mayıs 2026 | KNN → SVM | ZCR, F0, Energy, MFCC, ΔMFCC, Spectral, Chroma | %65.3 |
| Faz 2 | Mayıs 2026 | SVM + RF Ensemble | + Tempo (BPM), Veri Artırma | %65.8 |
| Faz 3 | Haziran 2026 | SVM + RF + XGBoost + MLP | + Ses Düzeyi Augmentation, Soft Voting | **%98.4** |

---

## Kullanılan Kütüphaneler

| Kütüphane | Versiyon | Kullanım Amacı |
|-----------|----------|----------------|
| streamlit | ≥1.35.0 | Web arayüzü |
| librosa | ≥0.10.0 | Ses yükleme ve özellik çıkarımı |
| numpy | ≥1.24.0 | Sayısal hesaplamalar |
| pandas | ≥2.0.0 | Veri işleme |
| matplotlib | ≥3.7.0 | Grafikler |
| scipy | ≥1.10.0 | Sinyal işleme |
| scikit-learn | ≥1.3.0 | SVM, RF, MLP modelleri, CV |
| xgboost | — | XGBoost modeli |
| openpyxl | ≥3.1.0 | Excel okuma/yazma |
| soundfile | ≥0.12.0 | WAV dosyası desteği |

---

## Sık Karşılaşılan Hatalar

### ❌ `'streamlit' is not recognized`
**Sebep:** Sanal ortam aktif değil.  
**Çözüm:** `.\venv\Scripts\activate` komutunu çalıştır, terminalde `(venv)` görünmeli.

---

### ❌ `ModuleNotFoundError: No module named 'xgboost'`
**Sebep:** xgboost yüklü değil.  
**Çözüm:**
```bash
pip install xgboost
```

---

### ❌ `ModuleNotFoundError: No module named 'librosa'`
**Sebep:** Kütüphaneler yüklenmemiş veya yanlış ortamda.  
**Çözüm:** Sanal ortamı aktif edip tekrar yükle:
```bash
.\venv\Scripts\activate
pip install -r requirements.txt
```

---

### ❌ Uygulama açılıyor ama hemen kapanıyor
**Sebep:** Genellikle bir import hatası.  
**Çözüm:** Terminaldeki hata mesajını oku ve yukarıdaki çözümleri dene.

---

### ❌ Sayfayı yeniledim ama değişiklikler görünmüyor
**Çözüm:** Tarayıcıda `Ctrl + Shift + R` ile cache'siz yenile.

---

### ❌ PowerShell'de `activate` komutu çalışmıyor
**Sebep:** Execution policy kısıtlaması.  
**Çözüm:** PowerShell'i yönetici olarak aç ve şunu çalıştır:
```bash
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```
Sonra tekrar dene.

---

*BIL216 Sinyaller ve Sistemler — Grup 05 — 2025-2026 Bahar Dönemi*
