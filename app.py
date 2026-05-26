"""
app.py – Final Proje Faz 3
BIL216 – Grup 05 | 2025-2026 Bahar Dönemi

Faz 3: SVM + RF + XGBoost + MLP Ensemble (Soft Voting)
       + Ses Düzeyi Veri Artırma (pitch shift, time stretch, gürültü)
       
cd "C:\Users\KULLANICI_ADIN\Desktop\Donemici_Proje"
.\venv\Scripts\activate
streamlit run app.py


"""

import os, io, zipfile, tempfile
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from audio_analysis import (
    extract_features, get_autocorr_array, compute_fft,
    N_MFCC, SPECTRAL_COLS
)
from classifier import (
    classify_single, train_and_evaluate, predict_single_with_model,
    evaluate, normalize_label,
    EMOTION_MAP, EMOTION_COLORS, EMOTION_ICONS, ALL_FEATURES,
)

# ──────────────────────────────────────────────
_FNAME_KEYWORDS = [
    ("saskın","S"),("saskin","S"),("saşkın","S"),("şaşkın","S"),
    ("şaşkin","S"),("surprised","S"),
    ("ofkeli","O"),("öfkeli","O"),("angry","O"),("kizgin","O"),
    ("uzgun","U"),("üzgün","U"),("sad","U"),
    ("mutlu","M"),("happy","M"),
    ("notr","N"),("nötr","N"),("neutral","N"),
]

def label_from_filename(fn: str) -> str:
    f = fn.lower()
    for kw, code in _FNAME_KEYWORDS:
        if kw in f:
            return code
    return "?"

# ══════════════════════════════════════════════
st.set_page_config(page_title="Duygu Analizi – Grup 05", page_icon="🎙️",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
html,body,[data-testid="stAppViewContainer"]{background:#070B14!important;font-family:'Plus Jakarta Sans',sans-serif!important;}
[data-testid="stHeader"]{background:transparent!important;}
[data-testid="stSidebar"]{background:#0D1220!important;border-right:1px solid #1E2940!important;min-width:270px!important;}
[data-testid="stSidebarContent"]{padding:1rem 1.2rem!important;}
p,li,label{color:#B8C4D8!important;}
h1,h2,h3,h4{color:#E8EFF8!important;font-weight:700!important;}
[data-testid="stTabs"] button{font-weight:600!important;color:#6B7FA3!important;border-radius:8px 8px 0 0!important;}
[data-testid="stTabs"] button[aria-selected="true"]{color:#00E5CC!important;border-bottom:2px solid #00E5CC!important;background:#0D1829!important;}
[data-testid="stMetric"]{background:#0D1829!important;border:1px solid #1E2F4A!important;border-radius:12px!important;padding:16px!important;}
[data-testid="stMetric"]:hover{border-color:#00E5CC!important;}
[data-testid="stMetricLabel"]{color:#6B7FA3!important;font-size:0.78rem!important;letter-spacing:0.06em!important;text-transform:uppercase!important;}
[data-testid="stMetricValue"]{color:#E8EFF8!important;font-family:'JetBrains Mono',monospace!important;font-size:1.5rem!important;}
[data-testid="stMetricDelta"]{color:#00E5CC!important;}
[data-testid="stButton"]>button{background:linear-gradient(135deg,#00C9AF,#006BFF)!important;color:#fff!important;font-weight:700!important;border:none!important;border-radius:10px!important;padding:12px 28px!important;box-shadow:0 4px 20px #00C9AF33!important;}
[data-testid="stButton"]>button:hover{opacity:0.88!important;transform:translateY(-1px)!important;}
[data-testid="stDownloadButton"]>button{background:#0D1829!important;border:1px solid #00E5CC!important;color:#00E5CC!important;font-weight:600!important;border-radius:10px!important;}
[data-testid="stFileUploader"]{background:#0D1829!important;border:1.5px dashed #1E3050!important;border-radius:14px!important;padding:8px!important;}
[data-testid="stFileUploader"]:hover{border-color:#00E5CC!important;}
[data-testid="stExpander"]{background:#0D1829!important;border:1px solid #1E2F4A!important;border-radius:12px!important;}
[data-testid="stDataFrame"]{border-radius:12px!important;overflow:hidden!important;border:1px solid #1E2F4A!important;}
hr{border-color:#1E2940!important;}code{color:#00E5CC!important;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style="padding:36px 0 24px 0;border-bottom:1px solid #1E2940;margin-bottom:28px;">
<div style="display:flex;align-items:center;gap:16px;">
<div style="width:52px;height:52px;background:linear-gradient(135deg,#00C9AF22,#006BFF22);border:1px solid #00C9AF55;border-radius:14px;display:flex;align-items:center;justify-content:center;font-size:26px;">🎙️</div>
<div>
<h1 style="margin:0;font-size:1.65rem;font-weight:800;background:linear-gradient(90deg,#E8EFF8 30%,#00E5CC);-webkit-background-clip:text;-webkit-text-fill-color:transparent;">
Ses İşareti Analizi ve Duygu Sınıflandırma</h1>
<p style="margin:4px 0 0 0;color:#4A6080;font-size:0.8rem;letter-spacing:0.05em;text-transform:uppercase;">
Grup 05 &nbsp;·&nbsp; 2025–2026 Bahar Dönemi &nbsp;·&nbsp; Final Proje – Faz 3</p>
</div></div></div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════
with st.sidebar:
    st.markdown("""<div style="padding:20px 0 8px 0;">
    <p style="font-size:0.7rem;letter-spacing:0.1em;text-transform:uppercase;color:#3A5070;margin:0 0 6px 0;">Kontrol Paneli</p>
    <h2 style="font-size:1.05rem;font-weight:700;color:#E8EFF8;margin:0;">Duygu Sınıflandırma</h2>
    </div>""", unsafe_allow_html=True)

    for code, name in EMOTION_MAP.items():
        st.markdown(f"""<div style="display:flex;align-items:center;gap:10px;margin-bottom:7px;">
        <div style="width:10px;height:10px;border-radius:50%;background:{EMOTION_COLORS[code]};flex-shrink:0;"></div>
        <span style="color:#B8C4D8;font-size:0.85rem;">{EMOTION_ICONS[code]} {name}</span></div>""", unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#1E2940;margin:16px 0;'>", unsafe_allow_html=True)
    bp     = st.session_state.get("best_params", {})
    bp_str = f"C={bp.get('C','?')}, γ={bp.get('gamma','?')}" if bp else "henüz eğitilmedi"

    st.markdown(f"""
    <p style="font-size:0.7rem;letter-spacing:0.1em;text-transform:uppercase;color:#3A5070;margin:0 0 10px 0;">Algoritma (Faz 3)</p>
    <div style="background:#111D30;border:1px solid #1E3050;border-radius:10px;padding:14px 16px;font-size:0.80rem;line-height:2.0;">
    <span style="color:#00E5CC;font-weight:700;">{len(ALL_FEATURES)} Özellik</span><br>
    <span style="color:#6B7FA3;">▸ F0, ZCR, Energy, RMS, Tempo</span><br>
    <span style="color:#6B7FA3;">▸ MFCC + ΔMFCC × 13</span><br>
    <span style="color:#6B7FA3;">▸ Spectral × 8, Chroma × 12</span><br><br>
    <span style="color:#00E5CC;font-weight:700;">Model: 4'lü Ensemble</span><br>
    <span style="color:#6B7FA3;">▸ SVM (RBF) · {bp_str}</span><br>
    <span style="color:#FFD166;font-weight:600;">▸ XGBoost ← YENİ</span><br>
    <span style="color:#FFD166;font-weight:600;">▸ MLP (256→128→64) ← YENİ</span><br>
    <span style="color:#6B7FA3;">▸ Random Forest (300 ağaç)</span><br>
    <span style="color:#FFD166;font-weight:600;">▸ Soft Voting ← YENİ</span><br><br>
    <span style="color:#00E5CC;font-weight:700;">Veri Artırma</span><br>
    <span style="color:#FFD166;font-weight:600;">▸ Pitch Shift ±0.5 ton ← YENİ</span><br>
    <span style="color:#FFD166;font-weight:600;">▸ Time Stretch ×0.9 ← YENİ</span><br>
    <span style="color:#6B7FA3;">▸ Beyaz Gürültü</span>
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
tab1, tab2 = st.tabs(["🎵 Tekli Ses Testi", "📊 Dataset Analizi"])

# ──────────────────────────────────────────────
# SEKME 1
# ──────────────────────────────────────────────
with tab1:
    st.markdown("""<div style="padding:20px 0 16px 0;">
    <h2 style="font-size:1.2rem;font-weight:800;color:#E8EFF8;margin:0 0 4px 0;">Tek Ses Dosyası Duygu Analizi</h2>
    <p style="color:#4A6080;font-size:0.82rem;margin:0;">4'lü Ensemble (SVM+RF+XGBoost+MLP) ile ~80 özellik üzerinden sınıflandırma.</p>
    </div>""", unsafe_allow_html=True)

    uploaded = st.file_uploader("Bir .wav dosyası yükleyin", type=["wav"])
    if uploaded:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(uploaded.read()); tmp_path = tmp.name
        with st.spinner("Analiz yapılıyor..."):
            try:
                feats = extract_features(tmp_path)
            except Exception as e:
                st.error(f"Hata: {e}"); os.unlink(tmp_path); st.stop()
        os.unlink(tmp_path)

        auto_label = label_from_filename(uploaded.name)
        models = st.session_state.get("final_models")
        scaler = st.session_state.get("scaler")
        le     = st.session_state.get("le")
        pred_code, pred_name = predict_single_with_model(feats, models, scaler, le)
        color = EMOTION_COLORS.get(pred_code, "#6B7FA3")
        icon  = EMOTION_ICONS.get(pred_code, "?")
        model_tag = "(4'lü Ensemble)" if models else "(kural tabanlı)"

        badge = ""
        if auto_label != "?":
            if auto_label == pred_code:
                badge = "<span style='background:#00E5A022;border:1px solid #00E5A055;color:#00E5A0;border-radius:8px;padding:3px 10px;font-size:0.78rem;font-weight:700;margin-left:12px;'>✅ Doğru</span>"
            else:
                badge = f"<span style='background:#FF6B6B22;border:1px solid #FF6B6B55;color:#FF6B6B;border-radius:8px;padding:3px 10px;font-size:0.78rem;font-weight:700;margin-left:12px;'>❌ Gerçek: {EMOTION_ICONS.get(auto_label,'')} {EMOTION_MAP.get(auto_label,auto_label)}</span>"

        st.markdown(f"""
        <div style="background:linear-gradient(135deg,{color}12,{color}06);border:1px solid {color}40;border-left:4px solid {color};padding:20px 28px;border-radius:14px;margin:20px 0 24px 0;display:flex;align-items:center;gap:20px;">
        <div style="width:56px;height:56px;background:{color}20;border:1.5px solid {color}50;border-radius:14px;display:flex;align-items:center;justify-content:center;font-size:2rem;">{icon}</div>
        <div>
        <p style="margin:0;font-size:0.72rem;letter-spacing:0.1em;text-transform:uppercase;color:{color}99;">Tahmin {model_tag}</p>
        <div style="display:flex;align-items:center;gap:4px;">
        <h2 style="margin:2px 0 4px 0;font-size:1.8rem;font-weight:800;color:{color};">{pred_name}</h2>{badge}</div>
        <p style="margin:0;color:#4A6080;font-size:0.85rem;font-family:'JetBrains Mono',monospace;">
        F0=<span style="color:{color};font-weight:600;">{feats['mean_f0']:.1f}Hz</span> &nbsp;|&nbsp;
        ZCR=<span style="color:{color};font-weight:600;">{feats['mean_zcr']:.4f}</span> &nbsp;|&nbsp;
        BPM=<span style="color:{color};font-weight:600;">{feats.get('tempo',0):.0f}</span></p>
        </div></div>""", unsafe_allow_html=True)

        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric("Ort. F0",    f"{feats['mean_f0']:.1f} Hz")
        c2.metric("F0 Std",     f"{feats['std_f0']:.1f} Hz")
        c3.metric("ZCR",        f"{feats['mean_zcr']:.4f}")
        c4.metric("Sesli Oran", f"%{feats['voiced_ratio']*100:.1f}")
        c5.metric("Tempo",      f"{feats.get('tempo',0):.0f} BPM")

        st.markdown("<hr style='border-color:#1E2940;margin:8px 0 20px 0;'>", unsafe_allow_html=True)
        frame = feats["sample_frame"]; sr_val = feats["sr"]
        col_l, col_r = st.columns(2)

        with col_l:
            st.markdown("<p style='color:#6B7FA3;font-size:0.82rem;font-weight:600;margin-bottom:8px;'>Otokorelasyon & FFT</p>", unsafe_allow_html=True)
            fig,(ax1,ax2) = plt.subplots(2,1,figsize=(6,5))
            fig.patch.set_facecolor("#0A0F1E")
            for ax in (ax1,ax2):
                ax.set_facecolor("#0D1829"); ax.tick_params(colors="#6B7FA3",labelsize=7)
                for s in ax.spines.values(): s.set_edgecolor("#1E2F4A")
            ac = get_autocorr_array(frame)
            ax1.plot(np.arange(len(ac))[:len(ac)//2],ac[:len(ac)//2],color="#5B8EFF",lw=1.2)
            ax1.set_title("Otokorelasyon R(τ)",color="#B8C4D8",fontsize=9,pad=6)
            ax1.set_xlabel("Lag",color="#4A6080",fontsize=7); ax1.set_ylabel("R(τ)",color="#4A6080",fontsize=7)
            freqs,spec = compute_fft(frame,sr_val)
            ax2.plot(freqs[freqs<800],spec[freqs<800],color="#FF6B8A",lw=1.2)
            if feats['mean_f0']>0:
                ax2.axvline(feats['mean_f0'],color="#00E5A0",ls="--",lw=1.5,label=f"F0={feats['mean_f0']:.0f}Hz")
                ax2.legend(fontsize=7,facecolor="#0D1829",labelcolor="#B8C4D8",edgecolor="#1E2F4A")
            ax2.set_title("FFT |X(f)|",color="#B8C4D8",fontsize=9,pad=6)
            ax2.set_xlabel("Hz",color="#4A6080",fontsize=7); ax2.set_ylabel("|X(f)|",color="#4A6080",fontsize=7)
            fig.tight_layout(pad=1.5); st.pyplot(fig); plt.close(fig)

        with col_r:
            st.markdown("<p style='color:#6B7FA3;font-size:0.82rem;font-weight:600;margin-bottom:8px;'>Enerji & ZCR</p>", unsafe_allow_html=True)
            fig2,(ax3,ax4) = plt.subplots(2,1,figsize=(6,5))
            fig2.patch.set_facecolor("#0A0F1E")
            for ax in (ax3,ax4):
                ax.set_facecolor("#0D1829"); ax.tick_params(colors="#6B7FA3",labelsize=7)
                for s in ax.spines.values(): s.set_edgecolor("#1E2F4A")
            hop   = feats["hop_length"]
            times = np.arange(len(feats["energy_arr"]))*hop/sr_val
            ax3.plot(times,feats["energy_arr"],color="#5B8EFF",lw=1.2)
            ax3.fill_between(times,feats["energy_arr"],where=feats["voiced_mask"][:len(times)],alpha=0.25,color="#00E5A0",label="Voiced")
            ax3.set_title("Kısa Süreli Enerji",color="#B8C4D8",fontsize=9,pad=6)
            ax3.set_xlabel("Zaman (s)",color="#4A6080",fontsize=7)
            ax3.legend(fontsize=7,facecolor="#0D1829",labelcolor="#B8C4D8",edgecolor="#1E2F4A")
            ax4.plot(times,feats["zcr_arr"][:len(times)],color="#FFB347",lw=1.2)
            ax4.axhline(0.15,color="#FF6B8A",ls="--",lw=1,label="Eşik=0.15")
            ax4.set_title("ZCR",color="#B8C4D8",fontsize=9,pad=6)
            ax4.set_xlabel("Zaman (s)",color="#4A6080",fontsize=7)
            ax4.legend(fontsize=7,facecolor="#0D1829",labelcolor="#B8C4D8",edgecolor="#1E2F4A")
            fig2.tight_layout(pad=1.5); st.pyplot(fig2); plt.close(fig2)

        if feats.get("mfcc_mean") is not None:
            st.markdown("<p style='color:#6B7FA3;font-size:0.82rem;font-weight:600;margin:16px 0 8px 0;'>MFCC ve ΔMFCC</p>", unsafe_allow_html=True)
            fig3,axes3 = plt.subplots(1,2,figsize=(10,2.8))
            fig3.patch.set_facecolor("#0A0F1E")
            for ax in axes3:
                ax.set_facecolor("#0D1829"); ax.tick_params(colors="#6B7FA3",labelsize=7)
                for s in ax.spines.values(): s.set_edgecolor("#1E2F4A")
            x = np.arange(N_MFCC)
            axes3[0].bar(x,feats["mfcc_mean"],color=EMOTION_COLORS.get(pred_code,"#5B8EFF"),alpha=0.8)
            axes3[0].set_xticks(x); axes3[0].set_xticklabels([f"C{i}" for i in range(N_MFCC)],fontsize=6,color="#6B7FA3")
            axes3[0].set_title("MFCC",color="#B8C4D8",fontsize=9)
            axes3[1].bar(x,feats["delta_mean"],color="#FFD166",alpha=0.8)
            axes3[1].set_xticks(x); axes3[1].set_xticklabels([f"C{i}" for i in range(N_MFCC)],fontsize=6,color="#6B7FA3")
            axes3[1].set_title("ΔMFCC",color="#B8C4D8",fontsize=9)
            fig3.tight_layout(); st.pyplot(fig3); plt.close(fig3)


# ──────────────────────────────────────────────
# SEKME 2
# ──────────────────────────────────────────────
with tab2:
    st.markdown("""<div style="padding:20px 0 16px 0;">
    <h2 style="font-size:1.2rem;font-weight:800;color:#E8EFF8;margin:0 0 4px 0;">Çok Gruplu Dataset Analizi — Faz 3</h2>
    <p style="color:#4A6080;font-size:0.82rem;margin:0;">
    4'lü Ensemble + Ses Düzeyi Veri Artırma + Stratified 5-Fold CV<br>
    <span style="color:#FFD166;font-size:0.78rem;">⚠️ Ses artırma açıkken analiz daha uzun sürer (~2-3 dk)</span></p>
    </div>""", unsafe_allow_html=True)

    st.markdown("""<div style="background:#0D1829;border:1px solid #1E3050;border-radius:12px;padding:14px 18px;margin:0 0 20px 0;font-size:0.82rem;line-height:1.9;">
    <span style="color:#00E5CC;font-weight:700;">Etiket dosya adından okunur:</span><br>
    <span style="color:#B8C4D8;">G05_D01_E_23_<b>Notr</b>_C1.wav → 😐 Nötr &nbsp;|&nbsp; G03_D02_K_25_<b>Mutlu</b>_C2.wav → 😊 Mutlu</span>
    </div>""", unsafe_allow_html=True)

    # Ses artırma açma/kapama
    use_aug = st.toggle("🎵 Ses düzeyi veri artırmayı etkinleştir (daha yüksek doğruluk, daha uzun süre)", value=True)

    upload_mode = st.radio("Yükleme yöntemi:",
        ["📦 ZIP dosyası (tüm gruplar)", "📁 Tek tek WAV seçimi"], horizontal=True)

    tmp_dir   = tempfile.mkdtemp()
    wav_paths = []

    if upload_mode == "📦 ZIP dosyası (tüm gruplar)":
        zip_file = st.file_uploader("ZIP dosyasını yükleyin", type=["zip"], key="zip_upload")
        if zip_file:
            with st.spinner("ZIP açılıyor..."):
                try:
                    with zipfile.ZipFile(io.BytesIO(zip_file.read())) as zf:
                        zf.extractall(tmp_dir)
                    for root,_,files in os.walk(tmp_dir):
                        for fn in files:
                            if fn.lower().endswith(".wav"):
                                wav_paths.append((fn, os.path.join(root,fn)))
                    st.success(f"✅ ZIP açıldı — {len(wav_paths)} WAV bulundu.")
                except Exception as e:
                    st.error(f"ZIP hatası: {e}")
    else:
        wav_ups = st.file_uploader(
            "🎵 WAV dosyaları — çoklu seçim",
            type=["wav"], accept_multiple_files=True, key="wav_upload"
        )
        if wav_ups:
            with st.spinner("Kaydediliyor..."):
                for wf in wav_ups:
                    path = os.path.join(tmp_dir,wf.name)
                    open(path,"wb").write(wf.read())
                    wav_paths.append((wf.name,path))
            st.success(f"✅ {len(wav_paths)} WAV yüklendi.")

    if wav_paths:
        labeled   = [(fn,p,label_from_filename(fn)) for fn,p in wav_paths]
        unknown   = [fn for fn,_,lbl in labeled if lbl=="?"]
        known_cnt = len(labeled)-len(unknown)
        if unknown:
            with st.expander(f"⚠️ {len(unknown)} dosyada etiket bulunamadı"):
                st.write(unknown)
        st.info(f"📊 {known_cnt} / {len(labeled)} dosyada etiket var — tümü analiz edilecek.")

        if st.button("🚀 Analizi Başlat", type="primary"):
            results  = []
            progress = st.progress(0)
            status   = st.empty()
            wav_dict = {}   # dosya_adı → tam_yol (ses artırma için)

            for i,(fn,fpath,true_label) in enumerate(labeled):
                status.text(f"Özellik çıkarılıyor: {fn} ({i+1}/{len(labeled)})")
                wav_dict[fn] = fpath
                try:
                    feats = extract_features(fpath)
                    entry = {
                        "Dosya_Adi"   : fn,
                        "Gercek_Duygu": true_label,
                        "mean_f0"     : round(feats["mean_f0"],1),
                        "std_f0"      : round(feats["std_f0"],1),
                        "mean_zcr"    : round(feats["mean_zcr"],4),
                        "mean_energy" : round(float(feats["mean_energy"]),6),
                        "voiced_ratio": round(feats["voiced_ratio"],3),
                        "rms_mean"    : round(feats.get("rms_mean",0),4),
                        "rms_std"     : round(feats.get("rms_std",0),4),
                        "tempo"       : round(feats.get("tempo",0),2),
                    }
                    for j in range(N_MFCC):
                        entry[f"mfcc_mean_{j}"]  = round(float(feats["mfcc_mean"][j]),4)
                        entry[f"mfcc_std_{j}"]   = round(float(feats["mfcc_std"][j]),4)
                        entry[f"delta_mean_{j}"] = round(float(feats["delta_mean"][j]),4)
                        entry[f"delta_std_{j}"]  = round(float(feats["delta_std"][j]),4)
                    for col in SPECTRAL_COLS:
                        entry[col] = round(float(feats.get(col,0)),4)
                    for j,v in enumerate(feats.get("chroma_mean",np.zeros(12))):
                        entry[f"chroma_{j}"] = round(float(v),4)
                    results.append(entry)
                except Exception as e:
                    results.append({"Dosya_Adi":fn,"Gercek_Duygu":true_label,
                        "mean_f0":0,"std_f0":0,"mean_zcr":0,"mean_energy":0,
                        "voiced_ratio":0,"rms_mean":0,"rms_std":0,"tempo":0,"Hata":str(e)})
                progress.progress((i+1)/len(labeled))

            status.empty(); progress.empty()
            res_df = pd.DataFrame(results)
            labeled_df = res_df[res_df["Gercek_Duygu"]!="?"].copy().reset_index(drop=True)

            spinner_msg = "4'lü Ensemble eğitiliyor (Ses Artırma + 5-Fold CV)... ~2-3 dk" if use_aug \
                          else "4'lü Ensemble eğitiliyor (5-Fold CV)..."
            with st.spinner(spinner_msg):
                accuracy,cv_preds,conf,final_models,scaler,le,valid_indices,best_params = \
                    train_and_evaluate(
                        labeled_df,
                        wav_paths=wav_dict if use_aug else None
                    )

            labeled_df["Tahmin"]   = "?"
            labeled_df["Dogru_mu"] = "⚠️"
            for vi,pred in zip(valid_indices,cv_preds):
                labeled_df.at[vi,"Tahmin"]   = pred
                labeled_df.at[vi,"Dogru_mu"] = "✅" if pred==labeled_df.at[vi,"Gercek_Duygu"] else "❌"

            st.session_state["final_models"] = final_models
            st.session_state["scaler"]       = scaler
            st.session_state["le"]           = le
            st.session_state["results_df"]   = labeled_df
            st.session_state["best_params"]  = best_params

        if "results_df" in st.session_state:
            res_df   = st.session_state["results_df"]
            valid    = res_df[res_df["Tahmin"]!="?"]
            accuracy,conf = evaluate(valid["Gercek_Duygu"].tolist(),valid["Tahmin"].tolist())
            bp  = st.session_state.get("best_params",{})
            bp_txt = f"C={bp.get('C','?')}, γ={bp.get('gamma','?')}" if bp else ""

            st.markdown("<hr style='border-color:#1E2940;margin:20px 0;'>", unsafe_allow_html=True)
            st.markdown(f"""<p style="font-size:0.72rem;letter-spacing:0.1em;text-transform:uppercase;
            color:#3A5070;margin:0 0 14px 0;">📊 4'lü Ensemble — 5-Fold CV
            <span style='color:#00E5CC;font-size:0.85rem;'>&nbsp;{bp_txt}</span></p>""", unsafe_allow_html=True)

            cols = st.columns(len(EMOTION_MAP)+1)
            cols[0].metric("Genel Doğruluk", f"%{accuracy*100:.1f}")
            for cw,(code,name) in zip(cols[1:],EMOTION_MAP.items()):
                sub = valid[valid["Gercek_Duygu"]==code]
                if len(sub)>0:
                    cw.metric(f"{EMOTION_ICONS[code]} {name}",
                              f"%{(sub['Tahmin']==sub['Gercek_Duygu']).mean()*100:.1f}",
                              delta=f"{len(sub)} kayıt")

            st.markdown("""<p style="font-size:0.72rem;letter-spacing:0.1em;text-transform:uppercase;color:#3A5070;margin:20px 0 10px 0;">🔲 Karışıklık Matrisi</p>""", unsafe_allow_html=True)
            lo = list(EMOTION_MAP.keys())
            ln = [f"{EMOTION_ICONS[c]} {EMOTION_MAP[c]}" for c in lo]
            conf_display = pd.DataFrame(
                [[conf.get(t,{}).get(p,0) for p in lo] for t in lo],
                index=[f"Gerçek: {n}" for n in ln],
                columns=[f"Tahmin: {n}" for n in ln],
            )
            st.dataframe(conf_display.style.background_gradient(
                cmap=__import__("matplotlib.colors",fromlist=["LinearSegmentedColormap"])
                    .LinearSegmentedColormap.from_list("c",["#0D1829","#00C9AF"]),axis=None
            ).format("{:d}"), use_container_width=True)

            st.markdown("""<p style="font-size:0.72rem;letter-spacing:0.1em;text-transform:uppercase;color:#3A5070;margin:20px 0 10px 0;">📋 İstatistiksel Özet</p>""", unsafe_allow_html=True)
            stat_rows=[]
            for code,name in EMOTION_MAP.items():
                sub=valid[valid["Gercek_Duygu"]==code]
                if len(sub)>0:
                    stat_rows.append({"Duygu":f"{EMOTION_ICONS[code]} {name}","Örnek":len(sub),
                        "Ort. F0":round(sub["mean_f0"].mean(),1),
                        "F0 Std":round(sub["std_f0"].mean(),1),
                        "Ort. ZCR":round(sub["mean_zcr"].mean(),4),
                        "Ort. BPM":round(sub["tempo"].mean(),1) if "tempo" in sub.columns else "-",
                        "Başarı":f"%{(sub['Tahmin']==sub['Gercek_Duygu']).mean()*100:.0f}"})
            st.dataframe(pd.DataFrame(stat_rows),use_container_width=True,hide_index=True)

            st.markdown("""<p style="font-size:0.72rem;letter-spacing:0.1em;text-transform:uppercase;color:#3A5070;margin:16px 0 10px 0;">📦 Özellik Dağılımları</p>""", unsafe_allow_html=True)
            fig4,axes=plt.subplots(1,4,figsize=(16,4))
            fig4.patch.set_facecolor("#0A0F1E")
            present=[c for c in lo if c in valid["Gercek_Duygu"].values]
            for ax,(feat,title) in zip(axes,[("mean_f0","F0 (Hz)"),("std_f0","F0 Std"),
                                             ("mean_zcr","ZCR"),("tempo","Tempo (BPM)")]):
                ax.set_facecolor("#0D1829"); ax.tick_params(colors="#6B7FA3",labelsize=8)
                for sp in ax.spines.values(): sp.set_edgecolor("#1E2F4A")
                data_plot=[valid[valid["Gercek_Duygu"]==c][feat].values
                           for c in present if feat in valid.columns]
                if data_plot:
                    bps=ax.boxplot([d for d in data_plot if len(d)>0],patch_artist=True,
                                   medianprops=dict(color="white",linewidth=2))
                    for patch,clr in zip(bps["boxes"],[EMOTION_COLORS[c] for c in present]):
                        patch.set_facecolor(clr); patch.set_alpha(0.7)
                ax.set_xticklabels([f"{EMOTION_ICONS[c]}\n{EMOTION_MAP[c]}" for c in present],
                                   color="#B8C4D8",fontsize=8)
                ax.set_title(title,color="#B8C4D8",fontsize=9)
            fig4.tight_layout(); st.pyplot(fig4); plt.close(fig4)

            st.markdown("<hr style='border-color:#1E2940;margin:20px 0;'>", unsafe_allow_html=True)
            dcols=["Dosya_Adi","Gercek_Duygu","Tahmin","Dogru_mu","mean_f0","std_f0","mean_zcr","tempo","voiced_ratio"]
            st.dataframe(res_df[[c for c in dcols if c in res_df.columns]],use_container_width=True,hide_index=True)

            buf=io.BytesIO()
            res_df.to_excel(buf,index=False,sheet_name="Sonuclar"); buf.seek(0)
            st.download_button("⬇️ Sonuçları Excel olarak indir",data=buf,
                file_name="Grup05_Duygu_Faz3_Sonuclar.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
