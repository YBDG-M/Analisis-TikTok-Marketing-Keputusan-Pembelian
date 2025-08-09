# app.py
import streamlit as st
import pandas as pd
import numpy as np
import cloudpickle
import plotly.express as px
from io import BytesIO

# ================================
# 1. Load Model
# ================================
@st.cache_resource
def load_model():
    with open("best_model_tuned.pkl", "rb") as f:
        model = cloudpickle.load(f)
    return model

model = load_model()

# ================================
# 2. Judul Aplikasi
# ================================
st.title("📊 Prediksi & Rekomendasi Konten TikTok per Brand + Insight Pembelian")
st.markdown("""
Aplikasi ini memprediksi performa konten TikTok, memberikan rekomendasi otomatis  
berdasarkan **Play Count**, **Collect Count**, dan **Digg Count** per brand (**Fore Coffee** & **Janji Jiwa**)  
serta menambahkan **insight potensi keputusan pembelian**.
""")

# ================================
# 3. Upload File
# ================================
uploaded_file = st.file_uploader("Unggah file CSV data TikTok", type=["csv"])

if uploaded_file is not None:
    try:
        df_new = pd.read_csv(uploaded_file)

        st.subheader("📄 Data Awal")
        st.dataframe(df_new.head())

        # ================================
        # 4. Buat Kolom Turunan
        # ================================
        if 'diggCount' in df_new.columns and 'authorMeta/fans' in df_new.columns:
            df_new['likes_per_follower'] = df_new['diggCount'] / (df_new['authorMeta/fans'] + 1)
        if 'commentCount' in df_new.columns and 'diggCount' in df_new.columns:
            df_new['comments_per_like'] = df_new['commentCount'] / (df_new['diggCount'] + 1)
        if 'playCount' in df_new.columns and 'authorMeta/fans' in df_new.columns:
            df_new['plays_per_follower'] = df_new['playCount'] / (df_new['authorMeta/fans'] + 1)

        # ================================
        # 5. Validasi Kolom
        # ================================
        expected_cols = model.feature_names_in_
        missing_cols = [col for col in expected_cols if col not in df_new.columns]

        if missing_cols:
            st.error(f"Kolom berikut hilang di dataset: {missing_cols}")
        else:
            # Prediksi
            preds_log = model.predict(df_new[expected_cols])
            preds = np.expm1(preds_log)

            pred_df = pd.DataFrame(preds, columns=["Pred_PlayCount", "Pred_CollectCount", "Pred_DiggCount"])
            result_df = pd.concat([df_new.reset_index(drop=True), pred_df], axis=1)

            st.subheader("📊 Hasil Prediksi")
            st.dataframe(result_df.head())

            # ================================
            # 6. Rekomendasi Otomatis per Brand
            # ================================
            st.subheader("💡 Rekomendasi Otomatis per Brand + Insight Pembelian")

            brands = ["Fore Coffee", "Janji Jiwa"]
            if "brand" not in result_df.columns:
                st.error("Kolom 'brand' tidak ditemukan. Pastikan dataset memiliki kolom brand.")
            else:
                for b in brands:
                    st.markdown(f"### ☕ {b}")
                    df_brand = result_df[result_df["brand"].str.lower() == b.lower()]
                    if df_brand.empty:
                        st.write("Tidak ada data untuk brand ini.")
                        continue

                    # ===== Potensi Views Tertinggi =====
                    top_play = df_brand.loc[df_brand["Pred_PlayCount"].idxmax()]
                    st.markdown(f"**Potensi penonton tertinggi:** `{top_play.get('authorMeta/name', 'N/A')}` "
                                f"di `{top_play.get('locationMeta/city', 'N/A')}`, **{top_play['Pred_PlayCount']:.0f} views**.")
                    if 'webVideoUrl' in top_play and pd.notna(top_play['webVideoUrl']):
                        st.markdown(f"[🔗 Lihat Video]({top_play['webVideoUrl']})")
                    st.info("Insight: Cocok untuk meningkatkan **brand awareness** karena jangkauan luas dapat membentuk persepsi positif di benak konsumen.")

                    # ===== Potensi Collect Tertinggi =====
                    top_collect = df_brand.loc[df_brand["Pred_CollectCount"].idxmax()]
                    st.markdown(f"**Potensi disimpan terbanyak:** `{top_collect.get('authorMeta/name', 'N/A')}` "
                                f"di `{top_collect.get('locationMeta/city', 'N/A')}`, **{top_collect['Pred_CollectCount']:.0f} simpanan**.")
                    if 'webVideoUrl' in top_collect and pd.notna(top_collect['webVideoUrl']):
                        st.markdown(f"[🔗 Lihat Video]({top_collect['webVideoUrl']})")
                    st.info("Insight: Tingginya jumlah simpanan menandakan minat yang kuat, berpotensi mendorong **pembelian di masa depan**.")

                    # ===== Potensi Likes Tertinggi =====
                    top_digg = df_brand.loc[df_brand["Pred_DiggCount"].idxmax()]
                    st.markdown(f"**Potensi disukai terbanyak:** `{top_digg.get('authorMeta/name', 'N/A')}` "
                                f"di `{top_digg.get('locationMeta/city', 'N/A')}`, **{top_digg['Pred_DiggCount']:.0f} likes**.")
                    if 'webVideoUrl' in top_digg and pd.notna(top_digg['webVideoUrl']):
                        st.markdown(f"[🔗 Lihat Video]({top_digg['webVideoUrl']})")
                    st.info("Insight: Engagement tinggi memberi efek **social proof**, meningkatkan keyakinan konsumen untuk membeli.")

            # ================================
            # 7. Visualisasi
            # ================================
            fig = px.scatter(
                result_df,
                x="Pred_PlayCount",
                y="Pred_DiggCount",
                color="Pred_CollectCount",
                hover_data=result_df.columns,
                title="Hubungan Prediksi Play Count vs Digg Count"
            )
            st.plotly_chart(fig, use_container_width=True)

            # ================================
            # 8. Download Hasil
            # ================================
            buffer = BytesIO()
            result_df.to_csv(buffer, index=False)
            buffer.seek(0)
            st.download_button(
                label="💾 Unduh Hasil Prediksi",
                data=buffer,
                file_name="hasil_prediksi_tiktok.csv",
                mime="text/csv"
            )

    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses file: {e}")
