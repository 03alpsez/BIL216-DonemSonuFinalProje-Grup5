"""
classifier.py
-------------
4'lü Ensemble + Ses Düzeyi Veri Artırma – Final Proje (Faz 3)
BIL216 – Grup 05 | 2025-2026 Bahar Dönemi

Faz 3 Yenilikleri:
  1. XGBoost (Gradient Boosted Trees) eklendi
  2. MLP (Çok Katmanlı Algılayıcı) eklendi
  3. 4'lü Ensemble: SVM + RF + XGBoost + MLP (Soft Voting)
     → Soft voting: her modelin olasılık tahminlerini ortalar
     → Hard voting'den daha iyi performans
  4. Ses düzeyi veri artırma:
     Orijinal ses + pitch_shift + time_stretch + gürültü
     → Eğitim seti az örnekli sınıflar için 4-5x büyür
     → Özellik uzayı artırmadan çok daha gerçekçi
"""

import numpy as np
import pandas as pd
from sklearn.svm             import SVC
from sklearn.ensemble        import RandomForestClassifier
from sklearn.neural_network  import MLPClassifier
from sklearn.preprocessing   import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from xgboost                 import XGBClassifier
from audio_analysis import (
    N_MFCC, mfcc_column_names, delta_mfcc_column_names,
    SPECTRAL_COLS, chroma_column_names,
    extract_features_augmented
)

# ──────────────────────────────────────────────
# ETİKET HARİTALARI
# ──────────────────────────────────────────────

EMOTION_MAP = {
    "N": "Nötr",
    "M": "Mutlu",
    "O": "Öfkeli",
    "U": "Üzgün",
    "S": "Şaşkın",
}
EMOTION_MAP_INV = {v: k for k, v in EMOTION_MAP.items()}
EMOTION_COLORS  = {"N":"#6B7FA3","M":"#FFD166","O":"#FF6B6B","U":"#5B8EFF","S":"#00E5A0"}
EMOTION_ICONS   = {"N":"😐","M":"😊","O":"😠","U":"😢","S":"😲"}

# ──────────────────────────────────────────────
# ETİKET NORMALİZASYONU
# ──────────────────────────────────────────────

_NORM = {
    "nötr":"N","notr":"N","n":"N","neutral":"N","nötür":"N","nötrr":"N",
    "mutlu":"M","m":"M","happy":"M","neşeli":"M","neseli":"M",
    "öfkeli":"O","ofkeli":"O","o":"O","angry":"O","kızgın":"O","kizgin":"O","ofke":"O","öfke":"O",
    "üzgün":"U","uzgun":"U","u":"U","sad":"U","uzgün":"U",
    "şaşkın":"S","saskin":"S","saskın":"S","s":"S","surprised":"S","şaşırma":"S","şaşkin":"S",
}

def normalize_label(label):
    if not isinstance(label, str):
        label = str(label)
    return _NORM.get(label.strip().lower(), label.strip())

# ──────────────────────────────────────────────
# ÖZELLİK SÜTUNLARI (~80)
# ──────────────────────────────────────────────

BASE_FEATURES  = ["mean_f0","std_f0","mean_zcr","mean_energy","voiced_ratio",
                  "rms_mean","rms_std","tempo"]
MFCC_FEATURES  = mfcc_column_names(N_MFCC)
DELTA_FEATURES = delta_mfcc_column_names(N_MFCC)
CHROMA_FEATS   = chroma_column_names()
ALL_FEATURES   = BASE_FEATURES + MFCC_FEATURES + DELTA_FEATURES + SPECTRAL_COLS + CHROMA_FEATS


def build_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    X = []
    for feat in ALL_FEATURES:
        col = df[feat].fillna(0).values if feat in df.columns else np.zeros(len(df))
        X.append(col)
    return np.column_stack(X)


def feats_dict_to_row(feats: dict) -> dict:
    """extract_features çıktısını DataFrame satırına çevirir."""
    row = {f: 0.0 for f in ALL_FEATURES}
    row.update({
        "mean_f0":     feats["mean_f0"],
        "std_f0":      feats["std_f0"],
        "mean_zcr":    feats["mean_zcr"],
        "mean_energy": feats["mean_energy"],
        "voiced_ratio":feats.get("voiced_ratio",0.0),
        "rms_mean":    feats.get("rms_mean",0.0),
        "rms_std":     feats.get("rms_std",0.0),
        "tempo":       feats.get("tempo",0.0),
    })
    for i,v in enumerate(feats.get("mfcc_mean",  [])):
        row[f"mfcc_mean_{i}"]  = float(v)
    for i,v in enumerate(feats.get("mfcc_std",   [])):
        row[f"mfcc_std_{i}"]   = float(v)
    for i,v in enumerate(feats.get("delta_mean", [])):
        row[f"delta_mean_{i}"] = float(v)
    for i,v in enumerate(feats.get("delta_std",  [])):
        row[f"delta_std_{i}"]  = float(v)
    for col in SPECTRAL_COLS:
        row[col] = feats.get(col, 0.0)
    for i,v in enumerate(feats.get("chroma_mean", np.zeros(12))):
        row[f"chroma_{i}"] = float(v)
    return row

# ──────────────────────────────────────────────
# MODEL TANIMI
# ──────────────────────────────────────────────

def _make_models(C=10.0, gamma="scale"):
    svm = SVC(
        kernel="rbf", C=C, gamma=gamma,
        probability=True, random_state=42
    )
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=None,
        min_samples_split=2, random_state=42, n_jobs=-1
    )
    xgb = XGBClassifier(
        n_estimators=300, max_depth=5,
        learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, use_label_encoder=False,
        eval_metric="mlogloss", random_state=42,
        verbosity=0
    )
    mlp = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        activation="relu", solver="adam",
        max_iter=500, random_state=42,
        early_stopping=False
    )
    return svm, rf, xgb, mlp


class _AlignedProbaWrapper:
    """
    SVM/RF string etiketle eğitilir; predict_proba sütunları
    le.classes_ sırasına hizalanır.
    """
    def __init__(self, model, le):
        self.model = model
        self.le    = le

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        raw_proba    = self.model.predict_proba(X)          # (n, n_classes)
        model_classes = self.model.classes_                  # string sırası
        aligned       = np.zeros((len(X), len(self.le.classes_)))
        for i, cls in enumerate(model_classes):
            j = np.where(self.le.classes_ == cls)[0]
            if len(j) > 0:
                aligned[:, j[0]] = raw_proba[:, i]
        return aligned


def _soft_vote(models, le, X_test):
    """
    4 modelin olasılık ortalamasıyla tahmin yapar.
    SVM/RF string etiket, XGB/MLP encoded etiket döndürür —
    hepsi ortak sınıf sırasına (le.classes_) göre hizalanır.
    """
    n_classes = len(le.classes_)
    proba_sum = np.zeros((len(X_test), n_classes))

    for m in models:
        try:
            p = m.predict_proba(X_test)
            if p.shape[1] == n_classes:
                proba_sum += p
        except Exception:
            pass

    indices = np.argmax(proba_sum, axis=1)
    return le.inverse_transform(indices)

# ──────────────────────────────────────────────
# TEK DOSYA FALLBACK
# ──────────────────────────────────────────────

def classify_single(mean_f0, std_f0, mean_zcr, mean_energy, mfcc_mean=None):
    if mean_f0 <= 0:
        return "N","Nötr"
    if mean_zcr < 0.06 and std_f0 < 25:
        return "U","Üzgün"
    if std_f0 > 55 and mean_f0 > 160:
        return "S","Şaşkın"
    if mean_zcr > 0.10 and std_f0 > 35:
        return "O","Öfkeli"
    if mean_f0 > 185 or (mean_zcr > 0.07 and std_f0 > 20):
        return "M","Mutlu"
    return "N","Nötr"

# ──────────────────────────────────────────────
# ANA EĞİTİM + DEĞERLENDİRME
# ──────────────────────────────────────────────

_PARAM_GRID = [
    {"C":1.0,   "gamma":"scale"},
    {"C":10.0,  "gamma":"scale"},
    {"C":100.0, "gamma":"scale"},
    {"C":10.0,  "gamma":"auto"},
]


def train_and_evaluate(results_df: pd.DataFrame,
                       wav_paths: dict = None,
                       k=None, n_folds: int = 5):
    """
    4'lü Ensemble (SVM+RF+XGB+MLP) + Soft Voting + Stratified K-Fold CV

    SIZINTI DÜZELTMESİ:
      - Scaler her fold içinde SADECE eğitim orijinalleriyle fit edilir.
      - Artırılmış örnekler sınıfa göre değil, KAYNAK ÖRNEĞE göre eklenir;
        böylece bir test kaydının artırılmış kopyaları eğitime sızmaz.
    """
    labels = list(EMOTION_MAP.keys())
    df = results_df.dropna(subset=["Gercek_Duygu"]).copy()
    df = df[df["Gercek_Duygu"].isin(labels)]
    valid_indices = df.index.tolist()
    df = df.reset_index(drop=True)

    if len(df) < 5:
        preds = _rule_based_batch(df)
        acc, conf = evaluate(df["Gercek_Duygu"].tolist(), preds)
        return acc, preds, conf, None, None, None, valid_indices, {}

    X_base = build_feature_matrix(df)          # ÖLÇEKLENMEMİŞ orijinaller
    y_base = df["Gercek_Duygu"].values
    le     = LabelEncoder().fit(labels)

    # ── Ses artırma: bir kez hesapla, kaynağı takip et, ÖLÇEKLENMEMİŞ sakla ──
    X_extra_raw, y_extra, origin_idx = [], [], []
    if wav_paths:
        for idx, (fname, label) in enumerate(zip(df["Dosya_Adi"].values, y_base)):
            fpath = wav_paths.get(fname)
            if fpath is None:
                continue
            try:
                aug_list = extract_features_augmented(
                    fpath, modes=["noise", "pitch+", "pitch-", "stretch"]
                )
                for af in aug_list[1:]:        # orijinali atla
                    row = feats_dict_to_row(af)
                    X_extra_raw.append([row[f] for f in ALL_FEATURES])
                    y_extra.append(label)
                    origin_idx.append(idx)     # hangi orijinal kayıttan geldi
            except Exception:
                continue

    X_extra_raw = np.array(X_extra_raw) if X_extra_raw else np.empty((0, X_base.shape[1]))
    y_extra     = np.array(y_extra)     if len(y_extra)  else np.array([], dtype=object)
    origin_idx  = np.array(origin_idx, dtype=int)

    # ── CV döngüsü ──
    min_class    = min(np.bincount([labels.index(l) for l in y_base]))
    actual_folds = max(2, min(n_folds, min_class))
    cv           = StratifiedKFold(n_splits=actual_folds, shuffle=True, random_state=42)

    best_acc, best_preds, best_params = -1.0, [], _PARAM_GRID[0]

    class _XGBWrap:
        def __init__(self, m): self.m = m
        def predict_proba(self, X): return self.m.predict_proba(X)

    for params in _PARAM_GRID:
        cv_preds = np.full(len(y_base), "?", dtype=object)
        try:
            for train_idx, test_idx in cv.split(X_base, y_base):
                # Scaler SADECE eğitim orijinalleriyle fit edilir (test sızmaz)
                scaler = StandardScaler().fit(X_base[train_idx])
                X_tr = scaler.transform(X_base[train_idx])
                X_te = scaler.transform(X_base[test_idx])
                y_tr = y_base[train_idx]

                # Artırılmışlar: SADECE bu fold'un eğitimindeki ÖRNEKLERDEN gelenler
                if len(X_extra_raw) > 0:
                    keep = np.isin(origin_idx, train_idx)   # örneğe göre, sınıfa DEĞİL
                    if keep.any():
                        X_tr = np.vstack([X_tr, scaler.transform(X_extra_raw[keep])])
                        y_tr = np.concatenate([y_tr, y_extra[keep]])

                svm, rf, xgb, mlp = _make_models(params["C"], params["gamma"])
                y_tr_enc = le.transform(y_tr)

                svm.fit(X_tr, y_tr)
                rf.fit(X_tr,  y_tr)
                xgb.fit(X_tr, y_tr_enc)
                mlp.fit(X_tr, y_tr_enc)

                svm_w = _AlignedProbaWrapper(svm, le)
                rf_w  = _AlignedProbaWrapper(rf,  le)
                cv_preds[test_idx] = _soft_vote([svm_w, rf_w, _XGBWrap(xgb), mlp], le, X_te)

            valid_mask = cv_preds != "?"
            acc_cv = float(np.mean(cv_preds[valid_mask] == y_base[valid_mask]))
            if acc_cv > best_acc:
                best_acc    = acc_cv
                best_preds  = list(cv_preds)
                best_params = params
        except Exception:
            continue

    accuracy, conf = evaluate(y_base.tolist(), best_preds)

    # ── Final model: tüm orijinaller + artırılmışlar (scaler orijinallere fit) ──
    final_scaler = StandardScaler().fit(X_base)
    X_final = final_scaler.transform(X_base)
    y_final = y_base
    if len(X_extra_raw) > 0:
        X_final = np.vstack([X_final, final_scaler.transform(X_extra_raw)])
        y_final = np.concatenate([y_final, y_extra])

    svm_f, rf_f, xgb_f, mlp_f = _make_models(best_params["C"], best_params["gamma"])
    y_final_enc = le.transform(y_final)
    svm_f.fit(X_final, y_final)
    rf_f.fit(X_final,  y_final)
    xgb_f.fit(X_final, y_final_enc)
    mlp_f.fit(X_final, y_final_enc)

    svm_fw = _AlignedProbaWrapper(svm_f, le)
    rf_fw  = _AlignedProbaWrapper(rf_f,  le)

    final_models = [svm_fw, rf_fw, _XGBWrap(xgb_f), mlp_f]
    return accuracy, best_preds, conf, final_models, final_scaler, le, valid_indices, best_params


def predict_single_with_model(feats, models, scaler, le=None):
    """Ensemble ile tek dosya tahmini. Model yoksa kural tabanlı."""
    if models is None or scaler is None:
        return classify_single(feats["mean_f0"], feats["std_f0"],
                               feats["mean_zcr"], feats["mean_energy"])
    row  = feats_dict_to_row(feats)
    X    = np.array([[row[f] for f in ALL_FEATURES]])
    X_sc = scaler.transform(X)
    code = _soft_vote(models, le, X_sc)[0]
    return code, EMOTION_MAP.get(code, "?")


def evaluate(true_labels, predictions):
    labels      = list(EMOTION_MAP.keys())
    conf_matrix = {t: {p: 0 for p in labels} for t in labels}
    correct = total = 0
    for true, pred in zip(true_labels, predictions):
        if true in labels and pred in labels:
            conf_matrix[true][pred] += 1
            if true == pred:
                correct += 1
            total += 1
    return (correct / total if total > 0 else 0.0), conf_matrix


def _rule_based_batch(df):
    return [classify_single(r.get("mean_f0",0), r.get("std_f0",0),
                             r.get("mean_zcr",0), r.get("mean_energy",0))[0]
            for _, r in df.iterrows()]
