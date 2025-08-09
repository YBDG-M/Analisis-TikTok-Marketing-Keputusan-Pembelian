import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# === Fungsi Mapping Nama Kolom ===
def map_columns(df):
    mapping = {
        'authorMeta/name': 'username',
        'webVideoUrl': 'webvideourl',
        'locationMeta/city': 'location',
        'locationMeta/locationName': 'locationname',
        'collectCount': 'collectcount',
        'diggCount': 'diggcount',
        'playCount': 'playcount',
        'brand': 'brand',
        'authorMeta/fans': 'fans'  # optional, kalau ada
    }
    df = df.rename(columns=mapping)
    return df

# === Fungsi Bersihkan Data ===
def clean_data(df):
    df = map_columns(df)

    # Hapus duplikat
    df = df.drop_duplicates()

    # Isi missing value
    df = df.fillna({
        'collectcount': 0,
        'diggcount': 0,
        'playcount': 0,
        'brand': 'Unknown',
        'location': 'Unknown',
        'webvideourl': '',
        'username': 'Unknown',
        'fans': np.nan
    })

    # Pastikan tipe numerik
    for col in ['collectcount', 'diggcount', 'playcount', 'fans']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df[['collectcount','diggcount','playcount']] = df[['collectcount','diggcount','playcount']].fillna(0)

    # Tangani outlier (capping 99 percentile)
    for col in ['collectcount', 'diggcount', 'playcount']:
        cap = df[col].quantile(0.99)
        df[col] = np.where(df[col] > cap, cap, df[col])

    return df

# === Fungsi Hitung Skor & Keputusan Pembelian ===
def compute_scores(df, w_collect=0.4, w_digg=0.35, w_play=0.25):
    # Pastikan kolom ada
    for col in ['collectcount','diggcount','playcount']:
        if col not in df.columns:
            df[col] = 0

    # Normalisasi min-max tiap kolom (jika semua nol, hasil tetap 0)
    def minmax(series):
        lo = series.min()
        hi = series.max()
        if hi - lo == 0:
            return series.apply(lambda _: 0.0)
        return (series - lo) / (hi - lo)

    df['n_collect'] = minmax(df['collectcount'])
    df['n_digg'] = minmax(df['diggcount'])
    df['n_play'] = minmax(df['playcount'])

    # Hitung skor gabungan dengan bobot
    df['performance_score'] = (
        df['n_collect'] * w_collect +
        df['n_digg'] * w_digg +
        df['n_play'] * w_play
    )

    return df

def aggregate_brand_location(df):
    agg = df.groupby(['brand','location'], as_index=False).agg({
        'performance_score': 'mean',
        'collectcount': 'sum',
        'diggcount': 'sum',
        'playcount': 'sum',
        'username': 'nunique'  # jumlah creator di lokasi
    }).rename(columns={'username':'unique_creators'})

    # urutkan
    agg = agg.sort_values(['brand','performance_score'], ascending=[True, False])
    return agg

def assign_priority(df_agg, high_thresh=None, low_thresh=None):
    # Jika threshold tidak diberikan, buat otomatis berdasarkan quantile
    if high_thresh is None:
        high_thresh = df_agg['performance_score'].quantile(0.75)
    if low_thresh is None:
        low_thresh = df_agg['performance_score'].quantile(0.40)

    def label(s):
        if s >= high_thresh:
            return 'High Priority'
        elif s >= low_thresh:
            return 'Medium Priority'
        else:
            return 'Low Priority'

    df_agg['priority'] = df_agg['performance_score'].apply(label)
    df_agg['priority_score_thresholds'] = f"high>={high_thresh:.3f}, low>={low_thresh:.3f}"
    return df_agg

# === Streamlit App ===
st.set_page_config(layout="wide", page_title="Analisis TikTok Marketing + Keputusan Pembelian")
st.title("📊 Analisis TikTok Marketing + Keputusan Pembelian")

uploaded_file = st.file_uploader("Upload file CSV", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df = clean_data(df)

    st.markdown("### Data Preview")
    st.dataframe(df.head(10))

    # Sidebar: bobot dan threshold manual
    st.sidebar.header("Pengaturan Keputusan Pembelian")
    w_collect = st.sidebar.slider("Bobot Collect (save)", 0.0, 1.0, 0.40, 0.05)
    w_digg = st.sidebar.slider("Bobot Digg (like)", 0.0, 1.0, 0.35, 0.05)
    w_play = st.sidebar.slider("Bobot Play (view)", 0.0, 1.0, 0.25, 0.05)
    # pastikan bobot berjumlah 1 (normalisasi otomatis jika perlu)
    total_w = w_collect + w_digg + w_play
    if abs(total_w - 1.0) > 1e-6:
        w_collect, w_digg, w_play = w_collect/total_w, w_digg/total_w, w_play/total_w
        st.sidebar.write(f"Bobot dinormalisasi ke: collect={w_collect:.2f}, digg={w_digg:.2f}, play={w_play:.2f}")

    # Hitung skor
    df = compute_scores(df, w_collect=w_collect, w_digg=w_digg, w_play=w_play)

    # Tampilkan top video berdasarkan skor
    st.header("🏆 Top Video berdasarkan Skor Performa")
    top_videos = df.sort_values('performance_score', ascending=False).head(10)[
        ['username','brand','location','collectcount','diggcount','playcount','performance_score','webvideourl']
    ]
    st.dataframe(top_videos)

    # Agregasi per brand-location
    st.header("📍 Rangkuman Per Brand & Lokasi")
    agg = aggregate_brand_location(df)

    # Pilihan brand untuk dilihat
    brands = agg['brand'].unique().tolist()
    chosen_brand = st.selectbox("Pilih brand untuk analisis lokasi", options=brands)

    if chosen_brand:
        agg_brand = agg[agg['brand'] == chosen_brand].copy()

        # Threshold manual (opsional)
        st.subheader(f"Keputusan Pembelian untuk {chosen_brand}")
        auto_thresh = st.checkbox("Gunakan threshold otomatis (quantile)", value=True)
        if not auto_thresh:
            high_thresh = st.number_input("High priority threshold (score >=)", min_value=0.0, max_value=1.0, value=0.7, step=0.01)
            low_thresh = st.number_input("Low/Medium threshold (score >=)", min_value=0.0, max_value=1.0, value=0.4, step=0.01)
        else:
            high_thresh = None
            low_thresh = None

        agg_brand = assign_priority(agg_brand, high_thresh=high_thresh, low_thresh=low_thresh)

        # Visualisasi ranking lokasi
        fig_rank = px.bar(agg_brand.sort_values('performance_score', ascending=False),
                          x='location', y='performance_score', color='priority',
                          title=f"Ranking Lokasi berdasarkan Skor ({chosen_brand})")
        st.plotly_chart(fig_rank, use_container_width=True)

        # Tampilkan tabel rekomendasi
        st.markdown("#### Tabel Rekomendasi Lokasi (ringkasan)")
        display_cols = ['brand','location','performance_score','priority','collectcount','diggcount','playcount','unique_creators']
        st.dataframe(agg_brand[display_cols].reset_index(drop=True))

        # Rekomendasi tindakan bisnis otomatis (deterministic rules)
        st.markdown("#### 💡 Rekomendasi Tindakan Bisnis Otomatis")
        recs = []
        for _, row in agg_brand.iterrows():
            if row['priority'] == 'High Priority':
                action = ("Invest/Buy Inventory & Jalankan Campaign lokal; "
                          "Kolaborasi dengan creator setempat; "
                          "Gunakan video dengan performa terbaik sebagai creative template.")
            elif row['priority'] == 'Medium Priority':
                action = ("Testing kecil-kecilan promosi; optimasi konten; "
                          "Jalankan A/B konten; pantau 2-4 minggu.")
            else:
                action = ("Low priority — tidak perlu investasi besar; "
                          "pantau perkembangan atau coba konten organik.")
            recs.append((row['brand'], row['location'], row['priority'], row['performance_score'], action))

        rec_df = pd.DataFrame(recs, columns=['brand','location','priority','score','recommended_action'])
        st.table(rec_df)

        # Tampilkan contoh video terbaik per lokasi (link)
        st.markdown("#### 🎯 Contoh Video Rujukan untuk Setiap Lokasi (video tertinggi per lokasi)")
        example_rows = []
        for _, loc_row in agg_brand.iterrows():
            # cari video tertinggi di lokasi itu
            mask = (df['brand'] == loc_row['brand']) & (df['location'] == loc_row['location'])
            sub = df[mask]
            if not sub.empty:
                top_idx = sub['performance_score'].idxmax()
                top_row = sub.loc[top_idx]
                example_rows.append({
                    'brand': loc_row['brand'],
                    'location': loc_row['location'],
                    'example_username': top_row.get('username',''),
                    'example_score': top_row['performance_score'],
                    'example_video': top_row.get('webvideourl','')
                })
        examples_df = pd.DataFrame(example_rows)
        if not examples_df.empty:
            for _, r in examples_df.iterrows():
                url = r['example_video'] if pd.notna(r['example_video']) and r['example_video'] != '' else None
                st.markdown(f"- **{r['location']}** — creator: *{r['example_username']}* (score: {r['example_score']:.3f}) " + (f"→ [Lihat Video]({url})" if url else "→ (no url)"))
        else:
            st.info("Tidak ada contoh video untuk ditampilkan.")

        # Unduh rekomendasi
        csv = agg_brand[display_cols + ['priority']].to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Unduh Rekomendasi Lokasi (CSV)", csv, "rekomendasi_lokasi.csv", "text/csv")

    # tambahan visualisasi global
    st.header("📈 Visualisasi Global Top Brand")
    top_brand = agg.groupby('brand', as_index=False).agg({'performance_score':'mean'}).sort_values('performance_score', ascending=False).head(10)
    fig_top_brand = px.bar(top_brand, x='brand', y='performance_score', title="Top Brand berdasarkan Skor Rata-rata")
    st.plotly_chart(fig_top_brand)

else:
    st.info("⬆️ Silakan upload file CSV terlebih dahulu.")
