import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# konfigurasi
JUDUL = "Prediksi Penyakit Diabetes"
SUBJUDUL = ("Evaluasi Kinerja Decision Tree, Random Forest, XGBoost, dan LightGBM "
            "dengan Penanganan Imbalanced Data Menggunakan SMOTE")
MATA_KULIAH = "Data Mining 2026"           
KELAS_DOSEN = "Dr. Wiranto M.Kom., M.Cs."     
INSTITUSI = "Fakultas Teknologi Informasi dan Sains Data Universitas Sebelas Maret"  

TIM = [  # 
    {"nama": "Benedhictus Kevin Doni", "nim": "L0124006", "peran": "Anggota 1"},
    {"nama": "Shafa Rifkika Nur Fauziah", "nim": "L0124031", "peran": "Anggota 2"},
    {"nama": "Aisyah Nurul Sholikhah", "nim": "L0124085", "peran": "Anggota 3"},
    {"nama": "Aisyah Zahrotul Jannah", "nim": "L0124086", "peran": "Anggota 4"},
    {"nama": "Az Zahra Sam Putri", "nim": "L0124089", "peran": "Anggota 5"},
]

ARTIFACT_DIR = "artifacts"

st.set_page_config(page_title=JUDUL, layout="wide", initial_sidebar_state="collapsed")

PRIMARY, ACCENT = "#0F4C81", "#E07A3F"
BLUE = "#3A86C8"   # biru untuk teks/aksen: kontras baik di tema terang maupun gelap
MODEL_COLORS = {"Decision Tree": "#0F4C81", "Random Forest": "#2A9D8F",
                "XGBoost": "#E07A3F", "LightGBM": "#8E5BA8"}
PAGES = ["Tentang Proyek", "EDA", "Data Preprocessing", "Stratified Split", "SMOTE",
         "Model dan Evaluasi", "Prediksi"]

#   CSS
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}
[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stSidebar"], [data-testid="collapsedControl"] {{ display: none; }}
#MainMenu, footer {{ visibility: hidden; }}
.block-container {{ padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1250px; }}

/* Catatan tema: teks memakai warna bawaan tema (inherit), sedangkan latar/garis memakai abu-abu
   transparan sehingga otomatis terbaca di mode terang maupun gelap. */
.hero {{ background: linear-gradient(135deg, {PRIMARY} 0%, #1B6CA8 100%);
        color: #FFFFFF; padding: 1.8rem 2rem; border-radius: 16px; margin-bottom: 1rem;
        box-shadow: 0 6px 20px rgba(15,76,129,.25); }}
.hero h1 {{ margin: 0; font-size: 1.9rem; font-weight: 700; color: #FFFFFF; }}
.hero p {{ margin: .4rem 0 0 0; font-size: .98rem; opacity: .92; color: #FFFFFF; }}

/* navigasi halaman (radio horizontal dibuat seperti tombol) */
div[role="radiogroup"] {{ gap: .4rem; flex-wrap: wrap; }}
div[role="radiogroup"] > label {{ background: rgba(128,128,128,.12); border: 1px solid rgba(128,128,128,.28);
        border-radius: 10px; padding: .45rem .9rem; cursor: pointer; transition: all .15s; }}
div[role="radiogroup"] > label:hover {{ background: rgba(128,128,128,.22); }}
div[role="radiogroup"] > label > div:first-child {{ display: none; }}
div[role="radiogroup"] > label:has(input:checked) {{ background: {PRIMARY}; border-color: {PRIMARY}; }}
div[role="radiogroup"] > label:has(input:checked) p {{ color: #FFFFFF !important; font-weight: 600; }}
div[role="radiogroup"] p {{ font-size: .9rem; margin: 0; }}

.kpi, .card, .member {{ background: rgba(128,128,128,.07); border: 1px solid rgba(128,128,128,.25);
        border-radius: 14px; }}
.kpi {{ padding: 1rem 1.2rem; height: 100%; }}
.kpi .label {{ font-size: .78rem; opacity: .7; text-transform: uppercase; letter-spacing: .04em; }}
.kpi .value {{ font-size: 1.7rem; font-weight: 700; color: {BLUE}; margin-top: .15rem; }}
.kpi .note {{ font-size: .78rem; opacity: .7; }}

.section-title {{ font-size: 1.15rem; font-weight: 600; margin: 1.4rem 0 .6rem 0;
                 padding-left: .7rem; border-left: 4px solid {ACCENT}; }}

.card {{ padding: 1.1rem 1.3rem; height: 100%; }}
.card h4 {{ margin: 0 0 .4rem 0; font-size: 1rem; color: {BLUE}; }}
.card p, .card li {{ font-size: .92rem; line-height: 1.55; }}
.step {{ display: flex; gap: .8rem; align-items: flex-start; margin-bottom: .7rem; }}
.step .num {{ min-width: 30px; height: 30px; border-radius: 50%; background: {PRIMARY}; color: #FFFFFF;
             display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: .85rem; }}
.step .txt {{ font-size: .92rem; line-height: 1.5; }}

.result-card {{ border-radius: 14px; padding: 1.4rem 1.6rem; color: #FFFFFF; margin-bottom: .8rem; }}
.result-pos {{ background: linear-gradient(135deg, #B42318, #E5484D); }}
.result-neg {{ background: linear-gradient(135deg, #1B7F5C, #2A9D8F); }}
.result-card .title {{ font-size: .85rem; opacity: .9; text-transform: uppercase; letter-spacing: .05em; }}
.result-card .main {{ font-size: 1.8rem; font-weight: 700; margin: .2rem 0; }}

.member {{ padding: 1.1rem .8rem; text-align: center; }}
.member .avatar {{ width: 56px; height: 56px; border-radius: 50%; background: {PRIMARY}; color: #FFFFFF;
                  display: flex; align-items: center; justify-content: center; font-weight: 700;
                  font-size: 1.2rem; margin: 0 auto .6rem auto; }}
.member .name {{ font-weight: 600; }}
.member .meta {{ font-size: .82rem; opacity: .7; }}
.member .role {{ display: inline-block; margin-top: .5rem; font-size: .75rem; padding: .15rem .6rem;
                background: rgba(58,134,200,.18); color: {BLUE}; border-radius: 999px; }}

.note-box {{ background: rgba(58,134,200,.12); border-left: 4px solid {BLUE}; padding: .8rem 1rem;
            border-radius: 8px; font-size: .9rem; margin: .6rem 0 1rem 0; }}
.warn-box {{ background: rgba(224,122,63,.14); border-left: 4px solid {ACCENT}; padding: .8rem 1rem;
            border-radius: 8px; font-size: .88rem; margin-top: 1rem; }}
.ctrl {{ background: rgba(128,128,128,.07); border: 1px solid rgba(128,128,128,.25); border-radius: 12px;
        padding: .8rem 1.2rem .2rem 1.2rem; margin: .8rem 0 1rem 0; }}
.footer {{ text-align: center; opacity: .6; font-size: .8rem; margin-top: 2.5rem; }}

.stTabs [data-baseweb="tab-list"] {{ gap: 6px; flex-wrap: wrap; }}
.stTabs [data-baseweb="tab"] {{ background: rgba(128,128,128,.12); border-radius: 8px 8px 0 0; padding: .5rem 1rem; }}
.stTabs [aria-selected="true"] {{ background: {PRIMARY} !important; }}
.stTabs [aria-selected="true"] p {{ color: #FFFFFF !important; }}
.stButton > button, .stFormSubmitButton > button {{ background: {PRIMARY}; color: #FFFFFF; border: none;
        border-radius: 10px; padding: .6rem 1.4rem; font-weight: 600; }}
.stButton > button:hover, .stFormSubmitButton > button:hover {{ background: #0B3A63; color: #FFFFFF; }}
</style>
""", unsafe_allow_html=True)


# ------------------------------ DATA ---------------------------------
@st.cache_resource
def load_results():
    return joblib.load(f"{ARTIFACT_DIR}/results.pkl")


@st.cache_resource
def load_models(n_drop: int):
    return joblib.load(f"{ARTIFACT_DIR}/models_drop_{n_drop}.pkl")


try:
    bundle = load_results()
except FileNotFoundError:
    st.error(f"Folder '{ARTIFACT_DIR}' belum ada. Jalankan sel export di notebook, "
             f"lalu salin folder 'artifacts' ke samping app.py.")
    st.stop()

R, META = bundle["results"], bundle["meta"]
if "eda" not in META:
    st.error("File results.pkl berasal dari versi lama. Jalankan ulang sel export terbaru di notebook.")
    st.stop()
EDA = META["eda"]
MAX_DROP = META["max_drop"]
ALL_FEATURES = META["features"]
MODEL_NAMES = list(R[0].keys())
METRICS = ["Accuracy", "Precision", "Recall", "Specificity", "F1-Score", "NPV", "ROC-AUC"]
NUM_COLS = ["age", "bmi", "HbA1c_level", "blood_glucose_level"]


def kpi(col, label, value, note=""):
    col.markdown(f'<div class="kpi"><div class="label">{label}</div><div class="value">{value}</div>'
                 f'<div class="note">{note}</div></div>', unsafe_allow_html=True)


def section(text):
    st.markdown(f'<div class="section-title">{text}</div>', unsafe_allow_html=True)


def note(text):
    st.markdown(f'<div class="note-box">{text}</div>', unsafe_allow_html=True)


def show(fig, h=380, container=None):
    # latar transparan + tema Streamlit bawaan (theme="streamlit") => teks, grid, dan legend
    # otomatis menyesuaikan mode terang/gelap
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=40, b=10), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", legend=dict(orientation="h", y=-0.2))
    (container or st).plotly_chart(fig, theme="streamlit")


# --------------------------- HEADER & NAVIGASI -----------------------
st.markdown(f'<div class="hero"><h1>{JUDUL}</h1><p>{SUBJUDUL}</p></div>', unsafe_allow_html=True)
page = st.radio("Navigasi", PAGES, horizontal=True, label_visibility="collapsed", key="page")

# jumlah fitur yang dihapus (disimpan di session_state agar tidak hilang saat pindah halaman)
if "n_drop" not in st.session_state:
    st.session_state.n_drop = 0
if page in ("Model dan Evaluasi", "Prediksi"):
    st.markdown('<div class="ctrl">', unsafe_allow_html=True)
    c1, c2 = st.columns([2, 3])
    with c1:
        st.session_state.n_drop = st.slider(
            "Jumlah fitur yang dihapus", 0, MAX_DROP, st.session_state.n_drop,
            help="Fitur dengan importance terendah (menurut masing-masing model) dibuang. "
                 "0 berarti seluruh 8 fitur dipakai.")
    with c2:
        st.caption(f"Fitur yang dipakai model: {len(ALL_FEATURES) - st.session_state.n_drop} dari "
                   f"{len(ALL_FEATURES)}. Geser slider untuk mencoba pengurangan fitur; seluruh tabel, "
                   f"grafik, dan prediksi pada halaman ini ikut berubah.")
    st.markdown('</div>', unsafe_allow_html=True)
n_drop = st.session_state.n_drop


# 1. TENTANG PROYEK
if page == "Tentang Proyek":
    section("Identitas Proyek")
    st.markdown(f"**Judul:** {SUBJUDUL}  \n**Mata kuliah:** {MATA_KULIAH}  \n"
                f"**Dosen:** {KELAS_DOSEN}  \n**Institusi:** {INSTITUSI}")

    section("Anggota Kelompok")
    cols = st.columns(5)
    for col, mbr in zip(cols, TIM):
        ini = "".join(w[0] for w in mbr["nama"].split()[:2]).upper()
        col.markdown(f'<div class="member"><div class="avatar">{ini}</div><div class="name">{mbr["nama"]}</div>'
                     f'<div class="meta">{mbr["nim"]}</div><div class="role">{mbr["peran"]}</div></div>',
                     unsafe_allow_html=True)

    section("Latar Belakang dan Tujuan")
    a, b = st.columns(2)
    a.markdown('<div class="card"><h4>Latar Belakang</h4><p>Diabetes adalah penyakit kronis yang sering '
               'terdeteksi terlambat. Data klinis sederhana seperti usia, BMI, HbA1c, dan glukosa darah dapat '
               'dimanfaatkan model pembelajaran mesin untuk membantu skrining awal. Tantangannya, data diabetes '
               'bersifat tidak seimbang: pasien positif jauh lebih sedikit daripada negatif, sehingga model '
               'cenderung mengabaikan kelas positif.</p></div>', unsafe_allow_html=True)
    b.markdown('<div class="card"><h4>Tujuan</h4><ul>'
               '<li>Membandingkan kinerja Decision Tree, Random Forest, XGBoost, dan LightGBM.</li>'
               '<li>Menguji pengaruh SMOTE terhadap data yang tidak seimbang.</li>'
               '<li>Menilai kestabilan model dengan 10-Fold Cross-Validation.</li>'
               '<li>Mengetahui fitur paling berpengaruh dan dampak pengurangan fitur terhadap kinerja.</li>'
               '</ul></div>', unsafe_allow_html=True)

    section("Dataset")
    st.markdown("Dataset: *Diabetes Prediction Dataset* (Kaggle, iammustafatz) dengan "
                f"{EDA['n_raw']:,} baris. Setelah pembersihan tersisa {META['n_clean']:,} baris dengan 8 fitur "
                "dan 1 target.")
    fitur = pd.DataFrame([
        ["gender", "Jenis kelamin (Female/Male)", "Kategorikal"],
        ["age", "Usia pasien (tahun)", "Numerik"],
        ["hypertension", "Riwayat hipertensi (0 = tidak, 1 = ya)", "Biner"],
        ["heart_disease", "Riwayat penyakit jantung (0 = tidak, 1 = ya)", "Biner"],
        ["smoking_history", "Riwayat merokok (never, former, not current, current, ever)", "Kategorikal"],
        ["bmi", "Indeks massa tubuh (kg/m²)", "Numerik"],
        ["HbA1c_level", "Rata-rata kadar gula darah 2-3 bulan terakhir (%)", "Numerik"],
        ["blood_glucose_level", "Kadar glukosa darah (mg/dL)", "Numerik"],
        ["diabetes (target)", "0 = tidak diabetes, 1 = diabetes", "Biner"],
    ], columns=["Variabel", "Keterangan", "Tipe"])
    st.dataframe(fitur, hide_index=True)

    section("Alur Program")
    steps = [
        ("EDA", "Statistik deskriptif, distribusi fitur, perbandingan antar kelas, dan korelasi."),
        ("Data Preprocessing", "Menghapus duplikat, gender 'Other', dan smoking_history 'No Info', lalu encoding."),
        ("Stratified Split", "Membagi data latih dan data uji 80:20 dengan proporsi kelas terjaga."),
        ("SMOTE", "Menyeimbangkan kelas pada data latih saja."),
        ("Model dan Evaluasi", "Melatih empat algoritma, 10-Fold CV, confusion matrix, 7 metrik, kurva ROC, "
                               "feature importance, dan reduksi fitur."),
        ("Prediksi", "Memasukkan data pasien dan melihat hasil dari keempat model."),
    ]
    for i, (t, d) in enumerate(steps, 1):
        st.markdown(f'<div class="step"><div class="num">{i}</div><div class="txt"><b>{t}</b><br>{d}</div></div>',
                    unsafe_allow_html=True)

    section("Cara Menggunakan Dashboard")
    note("Pilih halaman lewat tombol navigasi di atas. Pada halaman <b>Model dan Evaluasi</b> dan "
         "<b>Prediksi</b> tersedia slider <b>Jumlah fitur yang dihapus</b> (0 sampai "
         f"{MAX_DROP}) untuk mencoba pengurangan fitur. Nilai slider tetap tersimpan saat berpindah halaman.")


# 2. EDA
elif page == "EDA":
    note(f"EDA dilakukan pada data mentah ({EDA['n_raw']:,} baris) sebelum pembersihan. "
         "Grafik distribusi memakai sampel acak 6.000 baris agar ringan.")
    tabs = st.tabs(["Statistik Deskriptif", "Distribusi Fitur", "Relasi terhadap Diabetes", "Korelasi"])

    with tabs[0]:
        section("Fitur Numerik")
        st.dataframe(EDA["desc"].style.format("{:.3f}"))
        l, r = st.columns(2)
        with l:
            section("Frekuensi gender")
            st.dataframe(EDA["freq_gender"].style.format({"Persen (%)": "{:.2f}"}))
        with r:
            section("Frekuensi smoking_history")
            st.dataframe(EDA["freq_smoking"].style.format({"Persen (%)": "{:.2f}"}))

    with tabs[1]:
        feat = st.selectbox("Pilih fitur numerik", NUM_COLS)
        smp = EDA["sample"].copy()
        smp["Kelas"] = smp["diabetes"].map({0: "Kelas 0", 1: "Kelas 1"})
        l, r = st.columns(2)
        fig = px.histogram(smp, x=feat, color="Kelas", barmode="overlay", opacity=.65, nbins=40,
                           marginal=None, color_discrete_sequence=[PRIMARY, ACCENT],
                           title=f"Histogram {feat} per kelas")
        show(fig, 380, l)
        fig = px.box(smp, x="Kelas", y=feat, color="Kelas", color_discrete_sequence=[PRIMARY, ACCENT],
                     title=f"Boxplot {feat} per kelas")
        fig.update_layout(showlegend=False)
        show(fig, 380, r)

    with tabs[2]:
        st.dataframe(EDA["rel"].style.format({"Rata-rata Kelas 0": "{:.3f}", "Rata-rata Kelas 1": "{:.3f}",
                                              "Selisih": "{:+.3f}", "Δ (%)": "{:+.2f}"}))
        d = EDA["rel"].reset_index().rename(columns={"index": "Fitur"})
        fig = px.bar(d, x="Fitur", y="Δ (%)", color_discrete_sequence=[ACCENT], text_auto=".1f",
                     title="Perubahan rata-rata Kelas 1 terhadap Kelas 0 (%)")
        show(fig, 360)

    with tabs[3]:
        corr = EDA["corr"]
        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                        title="Heatmap Korelasi Pearson (data bersih, termasuk target)")
        show(fig, 560)
        st.caption("Korelasi fitur terhadap diabetes:")
        st.dataframe(corr["diabetes"].drop("diabetes").sort_values(ascending=False).to_frame("Korelasi")
                     .style.format("{:.4f}"))


# 3. DATA PREPROCESSING
elif page == "Data Preprocessing":
    cl = META["cleaning"]
    c = st.columns(4)
    kpi(c[0], "Data Awal", f"{EDA['n_raw']:,}", "baris")
    kpi(c[1], "Data Bersih", f"{META['n_clean']:,}", "baris")
    kpi(c[2], "Baris Dibuang", f"{EDA['n_raw'] - META['n_clean']:,}", "duplikat + anomali")
    kpi(c[3], "Jumlah Fitur", f"{len(ALL_FEATURES)}", "tidak ada fitur baru")

    section("Rekap Pembersihan Data")
    st.dataframe(cl.astype(str), hide_index=True)
    note("Tidak ditemukan missing value (NaN) maupun nilai 0 tidak valid pada fitur klinis. Anomali yang "
         "ditangani: baris duplikat, gender 'Other', dan smoking_history 'No Info'.")

    fig = go.Figure(go.Waterfall(
        x=["Data awal", "Duplikat", "Gender 'Other'", "Smoking 'No Info'", "Data bersih"],
        measure=["absolute", "relative", "relative", "relative", "total"],
        y=[int(cl.iloc[0]["Sisa baris"]), -int(cl.iloc[1]["Baris dihapus"]), -int(cl.iloc[2]["Baris dihapus"]),
           -int(cl.iloc[3]["Baris dihapus"]), 0],
        connector={"line": {"color": "#CBD5E1"}}, decreasing={"marker": {"color": ACCENT}},
        increasing={"marker": {"color": PRIMARY}}, totals={"marker": {"color": PRIMARY}}))
    fig.update_layout(title="Jumlah baris pada tiap tahap pembersihan")
    show(fig, 360)

    section("Encoding Fitur Kategorikal")
    enc = pd.DataFrame([
        ["gender", "Female", 0], ["gender", "Male", 1],
        ["smoking_history", "never", 0], ["smoking_history", "former", 1], ["smoking_history", "not current", 2],
        ["smoking_history", "current", 3], ["smoking_history", "ever", 4]],
        columns=["Fitur", "Kategori", "Kode"])
    st.dataframe(enc, hide_index=True)
    st.caption("Setiap fitur kategorikal tetap satu kolom sehingga jumlah fitur tetap 8.")


# 4. STRATIFIED SPLIT
elif page == "Stratified Split":
    sp = META["split"]
    note("Data dibagi 80:20 secara stratified sehingga proporsi kelas pada data latih dan data uji "
         "sama dengan proporsi pada data keseluruhan.")
    st.dataframe(sp.style.format({"Kelas 0 (%)": "{:.2f}", "Kelas 1 (%)": "{:.2f}"}), hide_index=True)
    d = sp[sp["Subset"] != "Total"].melt(id_vars="Subset", value_vars=["Kelas 0", "Kelas 1"],
                                         var_name="Kelas", value_name="Sampel")
    fig = px.bar(d, x="Subset", y="Sampel", color="Kelas", barmode="group", text_auto=",",
                 color_discrete_sequence=[PRIMARY, ACCENT], title="Jumlah sampel per kelas")
    show(fig, 380)

# 5. SMOTE
elif page == "SMOTE":
    sm = META["smote"].rename_axis("Kelas").reset_index()
    note("SMOTE hanya diterapkan pada data latih. Data uji tetap berupa data asli agar evaluasi tidak bias.")
    l, r = st.columns([2, 3])
    with l:
        section("Jumlah Kelas pada Data Latih")
        st.dataframe(sm.assign(Kelas=sm["Kelas"].map({0: "Kelas 0", 1: "Kelas 1"})), hide_index=True)
    with r:
        m = sm.melt(id_vars="Kelas", var_name="Tahap", value_name="Jumlah")
        m["Kelas"] = m["Kelas"].map({0: "Kelas 0", 1: "Kelas 1"})
        fig = px.bar(m, x="Tahap", y="Jumlah", color="Kelas", barmode="group", text_auto=",",
                     color_discrete_sequence=[PRIMARY, ACCENT], title="Sebelum vs Sesudah SMOTE")
        show(fig, 360)

    section("Pengaruh SMOTE terhadap Kinerja (semua fitur, data uji)")
    no, wi = META["no_smote"], META["with_smote"]
    l, r = st.columns(2)
    with l:
        st.caption("Tanpa SMOTE"); st.dataframe(no.style.format("{:.4f}"))
    with r:
        st.caption("Dengan SMOTE"); st.dataframe(wi.style.format("{:.4f}"))
    st.caption("Perubahan (Dengan SMOTE dikurangi Tanpa SMOTE)")
    dl = (wi - no)[["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]]
    st.dataframe(dl.style.format("{:+.4f}"))
    g = pd.DataFrame({"Tanpa SMOTE": no["Recall"], "Dengan SMOTE": wi["Recall"]}).reset_index() \
        .melt(id_vars="index", var_name="Skenario", value_name="Recall")
    fig = px.bar(g, x="index", y="Recall", color="Skenario", barmode="group", text_auto=".3f",
                 color_discrete_sequence=["#A6BDDB", PRIMARY], labels={"index": "Model"},
                 title="Recall: tanpa SMOTE vs dengan SMOTE")
    show(fig, 360)

# 6. MODEL DAN EVALUASI
elif page == "Model dan Evaluasi":
    note(f"Hasil pada data uji dengan <b>{len(ALL_FEATURES) - n_drop} fitur</b> "
         f"(menghapus {n_drop} fitur dengan importance terendah). Model dilatih pada data hasil SMOTE.")
    tabs = st.tabs(["10-Fold CV", "Rekap Metrik", "Confusion Matrix", "Kurva ROC",
                    "Feature Importance", "Semua Fitur vs Fitur Terpilih"])

    with tabs[0]:
        if META.get("cv") is not None:
            st.caption("Skor 10-Fold Cross-Validation pada data latih (Mean ± Std. Dev.), semua fitur, "
                       "SMOTE di dalam tiap fold.")
            st.dataframe(META["cv"].set_index("Model"))
        else:
            st.info("Tabel CV tidak tersedia. Jalankan bagian 9 di notebook sebelum sel export.")

    with tabs[1]:
        tbl = pd.DataFrame({m: R[n_drop][m]["metrics"] for m in MODEL_NAMES}).T[METRICS]
        st.dataframe(tbl.style.format("{:.4f}").background_gradient(cmap="Blues", axis=0))
        long = tbl.reset_index().melt(id_vars="index", var_name="Metrik", value_name="Nilai")
        fig = px.bar(long, x="Metrik", y="Nilai", color="index", barmode="group",
                     color_discrete_map=MODEL_COLORS, labels={"index": "Model"})
        fig.update_yaxes(range=[0, 1.02])
        show(fig, 400)

    with tabs[2]:
        st.dataframe(pd.DataFrame({m: R[n_drop][m]["cm"] for m in MODEL_NAMES}).T[["TP", "TN", "FP", "FN"]])
        cols = st.columns(4)
        for col, m in zip(cols, MODEL_NAMES):
            cm = R[n_drop][m]["cm"]
            fig = px.imshow([[cm["TN"], cm["FP"]], [cm["FN"], cm["TP"]]], text_auto=True,
                            color_continuous_scale="Blues", x=["Pred 0", "Pred 1"], y=["Aktual 0", "Aktual 1"])
            fig.update_layout(title=m, coloraxis_showscale=False)
            show(fig, 300, col)

    with tabs[3]:
        fig = go.Figure()
        for m in MODEL_NAMES:
            roc, auc = R[n_drop][m]["roc"], R[n_drop][m]["metrics"]["ROC-AUC"]
            fig.add_trace(go.Scatter(x=roc["fpr"], y=roc["tpr"], mode="lines", name=f"{m} (AUC {auc:.4f})",
                                     line=dict(color=MODEL_COLORS[m], width=2.5)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random",
                                 line=dict(color="#9CA3AF", dash="dash")))
        fig.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
        show(fig, 460)

    with tabs[4]:
        model_sel = st.selectbox("Pilih model", MODEL_NAMES, key="fi_model")
        t = META["importance"][model_sel].reset_index()
        dropped = set(R[n_drop][model_sel]["dropped"])
        colors = ["#CBD5E1" if f in dropped else MODEL_COLORS[model_sel] for f in t["Fitur"]]
        fig = go.Figure()
        fig.add_trace(go.Bar(x=t["Fitur"], y=t["Importance"], name="Importance", marker_color=colors))
        fig.add_trace(go.Scatter(x=t["Fitur"], y=t["Kumulatif (%)"], name="Kumulatif (%)", yaxis="y2",
                                 mode="lines+markers", line=dict(color=ACCENT, dash="dash")))
        fig.update_layout(yaxis=dict(title="Importance"),
                          yaxis2=dict(title="Kumulatif (%)", overlaying="y", side="right", range=[0, 105]))
        show(fig, 420)
        st.caption("Batang abu-abu adalah fitur yang dibuang pada pengaturan slider saat ini.")
        st.dataframe(t.style.format({"Importance": "{:.4f}", "Kumulatif (%)": "{:.2f}"}), hide_index=True)

    with tabs[5]:
        if n_drop == 0:
            st.info("Geser slider ke nilai lebih dari 0 untuk melihat perbandingan dengan semua fitur.")
        else:
            cols_m, rows = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"], []
            for m in MODEL_NAMES:
                for c in cols_m:
                    a, b = R[0][m]["metrics"][c], R[n_drop][m]["metrics"][c]
                    rows.append({"Model": m, "Metrik": c, "Semua Fitur": a, "Fitur Terpilih": b, "Δ": b - a})
            st.dataframe(pd.DataFrame(rows).style.format(
                {"Semua Fitur": "{:.4f}", "Fitur Terpilih": "{:.4f}", "Δ": "{:+.4f}"}), hide_index=True, height=420)
            section("Fitur per Model")
            st.dataframe(pd.DataFrame([{"Model": m, "Fitur Dibuang": ", ".join(R[n_drop][m]["dropped"]),
                                        "Fitur Terpilih": ", ".join(R[n_drop][m]["selected"])}
                                       for m in MODEL_NAMES]), hide_index=True)

# 7. PREDIKSI
else:
    models = load_models(n_drop)
    note(f"Prediksi memakai model dengan <b>{len(ALL_FEATURES) - n_drop} fitur</b>. Isian untuk fitur yang "
         "dibuang tetap tampil tetapi tidak dipakai model.")
    fs = META["feat_stats"]
    SMOKE = {"never": 0, "former": 1, "not current": 2, "current": 3, "ever": 4}

    with st.form("form_prediksi"):
        a, b, c = st.columns(3)
        gender = a.selectbox("Gender", ["Female", "Male"])
        age = b.number_input("Usia (tahun)", float(fs.loc["age", "min"]), float(fs.loc["age", "max"]), 45.0, 1.0)
        smoking = c.selectbox("Riwayat merokok", list(SMOKE.keys()))
        a, b, c = st.columns(3)
        bmi = a.number_input("BMI", float(fs.loc["bmi", "min"]), float(fs.loc["bmi", "max"]), 27.0, 0.1)
        hba1c = b.number_input("Kadar HbA1c (%)", float(fs.loc["HbA1c_level", "min"]),
                               float(fs.loc["HbA1c_level", "max"]), 5.5, 0.1)
        glu = c.number_input("Glukosa darah (mg/dL)", float(fs.loc["blood_glucose_level", "min"]),
                             float(fs.loc["blood_glucose_level", "max"]), 130.0, 1.0)
        a, b, c = st.columns(3)
        hyp = a.selectbox("Hipertensi", ["Tidak", "Ya"])
        heart = b.selectbox("Penyakit jantung", ["Tidak", "Ya"])
        model_sel = c.selectbox("Model utama", MODEL_NAMES, index=MODEL_NAMES.index("LightGBM"))
        submitted = st.form_submit_button("Prediksi")

    if submitted:
        row = {"gender": 0 if gender == "Female" else 1, "age": age, "hypertension": int(hyp == "Ya"),
               "heart_disease": int(heart == "Ya"), "smoking_history": SMOKE[smoking], "bmi": bmi,
               "HbA1c_level": hba1c, "blood_glucose_level": glu}

        def predict(name):
            feats = models[name]["features"]
            x = pd.DataFrame([[row[f] for f in feats]], columns=feats)
            return float(models[name]["model"].predict_proba(x)[0, 1])

        p = predict(model_sel)
        pos = p >= 0.5
        st.markdown(f'<div class="result-card {"result-pos" if pos else "result-neg"}">'
                    f'<div class="title">Hasil {model_sel}</div>'
                    f'<div class="main">{"Berisiko Diabetes" if pos else "Tidak Berisiko Diabetes"}</div>'
                    f'<div>Probabilitas diabetes: {p*100:.1f}%</div></div>', unsafe_allow_html=True)

        section("Perbandingan Seluruh Model")
        probs = pd.DataFrame({"Model": MODEL_NAMES, "Probabilitas": [predict(m) for m in MODEL_NAMES]})
        probs["Prediksi"] = np.where(probs["Probabilitas"] >= 0.5, "Positif", "Negatif")
        fig = px.bar(probs, x="Probabilitas", y="Model", orientation="h", color="Model",
                     color_discrete_map=MODEL_COLORS, text=probs["Probabilitas"].map("{:.1%}".format))
        fig.add_vline(x=0.5, line_dash="dash", line_color="#9CA3AF")
        fig.update_layout(showlegend=False); fig.update_xaxes(range=[0, 1])
        show(fig, 280)
        st.dataframe(probs.style.format({"Probabilitas": "{:.2%}"}), hide_index=True)
        st.markdown('<div class="warn-box">Hasil ini berasal dari model pembelajaran mesin untuk keperluan '
                    'akademik dan bukan diagnosis medis. Konsultasikan dengan tenaga kesehatan.</div>',
                    unsafe_allow_html=True)

st.markdown(f'<div class="footer">{MATA_KULIAH} &middot; {INSTITUSI}</div>', unsafe_allow_html=True)