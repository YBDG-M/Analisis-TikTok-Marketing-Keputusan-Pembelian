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
        'authorMeta/fans': 'fans'
    }
    df = df.rename(columns=mapping)
    return df

# === Fungsi Bersihkan Data ===
def clean_data(df):
    df = map_columns(df)
    df = df.drop_duplicates()
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
    for col in ['collectcount', 'diggcount', 'playcount', 'fans']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df[['collectcount','diggcount','playcount']] = df[['collectcount','diggcount','playcount']].fillna(0)
    for col in ['collectcount', 'diggcount', 'playcount']:
        cap = df[col].quantile(0.99)
        df[col] = np.where(df[col] > cap, cap, df[col])
    return df

# === Fungsi Hitung Skor ===
def compute_scores(df, w_collect=0.4, w_digg=0.35, w_play=0.25):
    for col in ['collectcount','diggcount','playcount']:
        if col not in df.columns:
            df[col] = 0
    def minmax(series):
        lo = series.min()
        hi = series.max()
        if hi - lo == 0:
            return series.apply(lambda _: 0.0)
        return (series - lo) / (hi - lo)
    df['n_collect'] = minmax(df['collectcount'])
    df['n_digg'] = minmax(df['diggcount'])
    df['n_play'] = minmax(df['playcount'])
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
        'username': 'nunique'
    }).rename(columns={'username':'unique_creators'})
    agg = agg.sort_values(['brand','performance_score'], ascending=[True, False])
    return agg

def assign_priority(df_agg, high_thresh=None, low_thresh=None):
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

    # Sidebar pengaturan bobot
    st.sidebar.header("Pengaturan Keputusan Pembelian")
    w_collect = st.sidebar.slider("Bobot Collect (save)", 0.0, 1.0, 0.40, 0.05)
    w_digg = st.sidebar.slider("Bobot Digg (like)", 0.0, 1.0, 0.35, 0.05)
    w_play = st.sidebar.slider("Bobot Play (view)", 0.0, 1.0, 0.25, 0.05)
    total_w = w_collect + w_digg + w_play
    if abs(total_w - 1.0) > 1e-6:
        w_collect, w_digg, w_play = w_collect/total_w, w_digg/total_w, w_play/total_w
        st.sidebar.write(f"Bobot dinormalisasi ke: collect={w_collect:.2f}, digg={w_digg:.2f}, play={w_play:.2f}")

    # Hitung skor gabungan
    df = compute_scores(df, w_collect=w_collect, w_digg=w_digg, w_play=w_play)

    # Sidebar filter brand & lokasi
    st.sidebar.header("Filter Data Analisis")
    brand_filter = st.sidebar.selectbox("Pilih Brand", options=["All"] + sorted(df['brand'].unique()))
    if brand_filter != "All":
        df = df[df['brand'] == brand_filter]
    location_filter = st.sidebar.selectbox("Pilih Lokasi", options=["All"] + sorted(df['location'].unique()))
    if location_filter != "All":
        df = df[df['location'] == location_filter]

    # === ANALISIS PER METRIK ===
    st.header("🔍 Analisis Metrik Spesifik")

    # PlayCount Tinggi
    st.subheader("📢 PlayCount Tinggi → Awareness (Jangkauan Luas)")
    top_play = df.sort_values('playcount', ascending=False).head(10)
    st.dataframe(top_play[['username','brand','location','playcount','webvideourl']])
    fig_play = px.bar(top_play, x='username', y='playcount', color='brand',
                      title="Top 10 Video berdasarkan PlayCount (Awareness)")
    st.plotly_chart(fig_play, use_container_width=True)

    # CollectCount Tinggi
    st.subheader("💰 CollectCount Tinggi → Indikasi Minat Beli / Niat Kembali")
    top_collect = df.sort_values('collectcount', ascending=False).head(10)
    st.dataframe(top_collect[['username','brand','location','collectcount','webvideourl']])
    fig_collect = px.bar(top_collect, x='username', y='collectcount', color='brand',
                         title="Top 10 Video berdasarkan CollectCount (Minat Beli)")
    st.plotly_chart(fig_collect, use_container_width=True)

    # DiggCount Tinggi
    st.subheader("🤝 DiggCount Tinggi → Engagement & Social Proof")
    top_digg = df.sort_values('diggcount', ascending=False).head(10)
    st.dataframe(top_digg[['username','brand','location','diggcount','webvideourl']])
    fig_digg = px.bar(top_digg, x='username', y='diggcount', color='brand',
                      title="Top 10 Video berdasarkan DiggCount (Engagement)")
    st.plotly_chart(fig_digg, use_container_width=True)

    # === ANALISIS GABUNGAN ===
    st.header("🏆 Top Video berdasarkan Skor Performa")
    top_videos = df.sort_values('performance_score', ascending=False).head(10)[
        ['username','brand','location','collectcount','diggcount','playcount','performance_score','webvideourl']
    ]
    st.dataframe(top_videos)

    agg = aggregate_brand_location(df)
    brands = agg['brand'].unique().tolist()
    chosen_brand = st.selectbox("Pilih brand untuk analisis lokasi", options=brands)
    if chosen_brand:
        agg_brand = agg[agg['brand'] == chosen_brand].copy()
        auto_thresh = st.checkbox("Gunakan threshold otomatis (quantile)", value=True)
        if not auto_thresh:
            high_thresh = st.number_input("High priority threshold (score >=)", min_value=0.0, max_value=1.0, value=0.7, step=0.01)
            low_thresh = st.number_input("Low/Medium threshold (score >=)", min_value=0.0, max_value=1.0, value=0.4, step=0.01)
        else:
            high_thresh = None
            low_thresh = None
        agg_brand = assign_priority(agg_brand, high_thresh=high_thresh, low_thresh=low_thresh)

        fig_rank = px.bar(agg_brand.sort_values('performance_score', ascending=False),
                          x='location', y='performance_score', color='priority',
                          title=f"Ranking Lokasi berdasarkan Skor ({chosen_brand})")
        st.plotly_chart(fig_rank, use_container_width=True)

        display_cols = ['brand','location','performance_score','priority','collectcount','diggcount','playcount','unique_creators']
        st.dataframe(agg_brand[display_cols].reset_index(drop=True))

        st.markdown("#### 💡 Rekomendasi Tindakan Bisnis Otomatis")
        recs = []
        for _, row in agg_brand.iterrows():
            if row['priority'] == 'High Priority':
                action = "Invest/Buy Inventory & Jalankan Campaign lokal; Kolaborasi dengan creator setempat; Gunakan video performa terbaik."
            elif row['priority'] == 'Medium Priority':
                action = "Testing kecil promosi; optimasi konten; A/B testing; pantau 2-4 minggu."
            else:
                action = "Low priority — tidak perlu investasi besar; pantau perkembangan atau coba konten organik."
            recs.append((row['brand'], row['location'], row['priority'], row['performance_score'], action))
        rec_df = pd.DataFrame(recs, columns=['brand','location','priority','score','recommended_action'])
        st.table(rec_df)

        st.markdown("#### 🎯 Video Rujukan per Lokasi")
        example_rows = []
        for _, loc_row in agg_brand.iterrows():
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

        csv = agg_brand[display_cols + ['priority']].to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Unduh Rekomendasi Lokasi (CSV)", csv, "rekomendasi_lokasi.csv", "text/csv")

    st.header("📈 Visualisasi Global Top Brand")
    top_brand = agg.groupby('brand', as_index=False).agg({'performance_score':'mean'}).sort_values('performance_score', ascending=False).head(10)
    fig_top_brand = px.bar(top_brand, x='brand', y='performance_score', title="Top Brand berdasarkan Skor Rata-rata")
    st.plotly_chart(fig_top_brand)

else:
    st.info("⬆️ Silakan upload file CSV terlebih dahulu.")
