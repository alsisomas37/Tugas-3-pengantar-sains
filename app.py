import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm

# Konfigurasi halaman agar tampilan penuh dan modern
st.set_page_config(
    page_title="Tugas 3 STDA4101",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Kustomisasi Tema via Judul
st.title("📊 Analisis Regresi Linear")
st.caption("Aplikasi Dasbor Statis - Tugas 3 STDA4101 (Analisis Data Statistik)")
st.markdown("---")

# ==========================
# FUNGSI KUSTOM UNTUK MEMPERCANTIK TABEL OLS
# ==========================
def buat_tabel_ols_cantik(model_regresi):
    """Fungsi untuk mengambil parameter OLS dan mengubahnya menjadi DataFrame yang rapi"""
    df_hasil = pd.DataFrame({
        'Koefisien (Beta)': model_regresi.params,
        'Standard Error': model_regresi.bse,
        't-Statistic': model_regresi.tvalues,
        'P-Value (p>|t|)': model_regresi.pvalues,
        '[Conf. Interval 95% Bawah]': model_regresi.conf_int()[0],
        '[Conf. Interval 95% Atas]': model_regresi.conf_int()[1]
    })
    
    # Menambahkan kolom status signifikansi untuk mempermudah pembacaan
    df_hasil['Kesimpulan'] = df_hasil['P-Value (p>|t|)'].apply(
        lambda x: 'Signifikan' if x < 0.05 else 'Tidak Signifikan'
    )
    
    # Format angka desimal agar rapi
    return df_hasil.style.format({
        'Koefisien (Beta)': '{:.4f}',
        'Standard Error': '{:.4f}',
        't-Statistic': '{:.4f}',
        'P-Value (p>|t|)': '{:.4f}',
        '[Conf. Interval 95% Bawah]': '{:.4f}',
        '[Conf. Interval 95% Atas]': '{:.4f}'
    })

# ==========================
# BACA DATA
# ==========================
@st.cache_data
def load_data():
    return pd.read_excel("Data Tugas Tuton STDA4101-2025.2.xlsx")

try:
    df = load_data()
except Exception as e:
    st.error(f"❌ Gagal memuat file data. Pastikan file 'Data Tugas Tuton STDA4101-2025.2.xlsx' berada di folder yang sama.")
    st.stop()

# ==========================
# DATA & DESKRIPTIF
# ==========================
col_data, col_desc = st.columns([1, 1])

with col_data:
    st.subheader("📋 Pratinjau Data")
    st.dataframe(df, use_container_width=True, height=250)

with col_desc:
    st.subheader("📈 Statistik Deskriptif")
    st.dataframe(df[["Y", "AGE", "LDL", "HDL"]].describe(), use_container_width=True, height=250)

st.markdown("---")

# ==========================
# SCATTER PLOT DENGAN GARIS TREN
# ==========================
st.subheader("🔍 Scatter Plot & Garis Tren Linear")

col1, col2, col3 = st.columns(3)

plt.rcParams['figure.facecolor'] = 'none'
plt.rcParams['axes.facecolor'] = '#f8f9fa'

with col1:
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(df["AGE"], df["Y"], color='#1f77b4', alpha=0.7, edgecolors='w', s=50, label='Data')
    m, c = np.polyfit(df["AGE"], df["Y"], 1)
    ax.plot(df["AGE"], m*df["AGE"] + c, color='#d62728', linestyle='--', linewidth=2, label='Tren')
    ax.set_title("AGE vs Y", fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel("AGE")
    ax.set_ylabel("Y")
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

with col2:
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(df["LDL"], df["Y"], color='#ff7f0e', alpha=0.7, edgecolors='w', s=50, label='Data')
    m, c = np.polyfit(df["LDL"], df["Y"], 1)
    ax.plot(df["LDL"], m*df["LDL"] + c, color='#d62728', linestyle='--', linewidth=2, label='Tren')
    ax.set_title("LDL vs Y", fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel("LDL")
    ax.set_ylabel("Y")
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

with col3:
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(df["HDL"], df["Y"], color='#2ca02c', alpha=0.7, edgecolors='w', s=50, label='Data')
    m, c = np.polyfit(df["HDL"], df["Y"], 1)
    ax.plot(df["HDL"], m*df["HDL"] + c, color='#d62728', linestyle='--', linewidth=2, label='Tren')
    ax.set_title("HDL vs Y", fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel("HDL")
    ax.set_ylabel("Y")
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

st.markdown("---")

# ==========================
# REGRESI SIMULTAN
# ==========================
st.subheader("🧠 Regresi Linear Berganda (Simultan)")

X = df[["AGE", "LDL", "HDL"]]
Y = df["Y"]
X = sm.add_constant(X)
model = sm.OLS(Y, X).fit()

col_sim1, col_sim2 = st.columns([2, 1])

with col_sim1:
    st.markdown("**Tabel Parameter Model (Rapi & Cantik):**")
    # Menampilkan tabel hasil modifikasi fungsi kustom
    st.dataframe(buat_tabel_ols_cantik(model), use_container_width=True)

with col_sim2:
    st.markdown("**Informasi & Persamaan:**")
    
    # Menggunakan komponen metrik bawaan streamlit agar visual menarik
    st.metric(label="Koefisien Determinasi ($R^2$)", value=f"{model.rsquared:.4f}")
    st.metric(label="F-Statistic P-Value", value=f"{model.f_pvalue:.4e}")
    
    const = model.params["const"]
    age = model.params["AGE"]
    ldl = model.params["LDL"]
    hdl = model.params["HDL"]
    
    st.markdown("**Persamaan Regresi Berganda:**")
    st.latex(
        rf"Y = {const:.3f} + ({age:.3f})\text{{AGE}} + ({ldl:.3f})\text{{LDL}} + ({hdl:.3f})\text{{HDL}}"
    )

st.markdown("---")

# ==========================
# REGRESI PARSIAL (Menggunakan Tabs)
# ==========================
st.subheader("📐 Analisis Regresi Parsial")

tab_age, tab_ldl, tab_hdl = st.tabs(["Variable AGE", "Variable LDL", "Variable HDL"])

with tab_age:
    X_age = sm.add_constant(df["AGE"])
    model_age = sm.OLS(Y, X_age).fit()
    
    c_age1, c_age2 = st.columns([2, 1])
    with c_age1:
        st.markdown("**Tabel Parameter OLS Parsial (AGE):**")
        st.dataframe(buat_tabel_ols_cantik(model_age), use_container_width=True)
    with c_age2:
        st.metric(label="Partial $R^2$ (AGE)", value=f"{model_age.rsquared:.4f}")
        st.markdown("**Persamaan Model:**")
        st.latex(rf"Y = {model_age.params['const']:.3f} + ({model_age.params['AGE']:.3f})\text{{AGE}}")

with tab_ldl:
    X_ldl = sm.add_constant(df["LDL"])
    model_ldl = sm.OLS(Y, X_ldl).fit()
    
    c_ldl1, c_ldl2 = st.columns([2, 1])
    with c_ldl1:
        st.markdown("**Tabel Parameter OLS Parsial (LDL):**")
        st.dataframe(buat_tabel_ols_cantik(model_ldl), use_container_width=True)
    with c_ldl2:
        st.metric(label="Partial $R^2$ (LDL)", value=f"{model_ldl.rsquared:.4f}")
        st.markdown("**Persamaan Model:**")
        st.latex(rf"Y = {model_ldl.params['const']:.3f} + ({model_ldl.params['LDL']:.3f})\text{{LDL}}")

with tab_hdl:
    X_hdl = sm.add_constant(df["HDL"])
    model_hdl = sm.OLS(Y, X_hdl).fit()
    
    c_hdl1, c_hdl2 = st.columns([2, 1])
    with c_hdl1:
        st.markdown("**Tabel Parameter OLS Parsial (HDL):**")
        st.dataframe(buat_tabel_ols_cantik(model_hdl), use_container_width=True)
    with c_hdl2:
        st.metric(label="Partial $R^2$ (HDL)", value=f"{model_hdl.rsquared:.4f}")
        st.markdown("**Persamaan Model:**")
        st.latex(rf"Y = {model_hdl.params['const']:.3f} + ({model_hdl.params['HDL']:.3f})\text{{HDL}}")

st.markdown("---")

# ==========================
# KESIMPULAN
# ==========================
st.subheader("📌 Kesimpulan Hasil Uji Signifikansi ($\\alpha = 5\%$)" )

k_col1, k_col2, k_col3 = st.columns(3)

with k_col1:
    st.markdown("**Variabel AGE**")
    if model.pvalues["AGE"] < 0.05:
        st.success(f"Signifikan (p-val: {model.pvalues['AGE']:.4f}) \n\nAGE berpengaruh signifikan terhadap Y.")
    else:
        st.warning(f"Tidak Signifikan (p-val: {model.pvalues['AGE']:.4f}) \n\nAGE tidak berpengaruh signifikan terhadap Y.")

with k_col2:
    st.markdown("**Variabel LDL**")
    if model.pvalues["LDL"] < 0.05:
        st.success(f"Signifikan (p-val: {model.pvalues['LDL']:.4f}) \n\nLDL berpengaruh signifikan terhadap Y.")
    else:
        st.warning(f"Tidak Signifikan (p-val: {model.pvalues['LDL']:.4f}) \n\nLDL tidak berpengaruh signifikan terhadap Y.")

with k_col3:
    st.markdown("**Variabel HDL**")
    if model.pvalues["HDL"] < 0.05:
        st.success(f"Signifikan (p-val: {model.pvalues['HDL']:.4f}) \n\nHDL berpengaruh signifikan terhadap Y.")
    else:
        st.warning(f"Tidak Signifikan (p-val: {model.pvalues['HDL']:.4f}) \n\nHDL tidak berpengaruh signifikan terhadap Y.")

# Ringkasan Akhir Berwarna Biru
st.info(
    f"💡 **Ringkasan Kemampuan Prediksi:** Kombinasi variabel AGE, LDL, dan HDL secara bersama-sama (simultan) mampu menjelaskan variabel Y sebesar **{model.rsquared * 100:.2f}%** (Nilai $R^2 = {model.rsquared:.4f}$), sedangkan sisanya dijelaskan oleh faktor lain di luar model."
)