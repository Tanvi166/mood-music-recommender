import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler


# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Mood-Based Music Recommender System",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -----------------------------
# Custom CSS for modern dark UI
# -----------------------------
st.markdown(
    """
    <style>
    .stApp {
        background: radial-gradient(circle at top left, #1f2937 0, #0f172a 32%, #020617 100%);
        color: #f8fafc;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #111827 0%, #020617 100%);
        border-right: 1px solid rgba(148, 163, 184, 0.22);
    }

    .hero {
        padding: 2rem 2.2rem;
        border-radius: 18px;
        background: linear-gradient(135deg, #ec4899 0%, #8b5cf6 45%, #06b6d4 100%);
        box-shadow: 0 20px 60px rgba(8, 13, 30, 0.45);
        margin-bottom: 1.4rem;
    }

    .hero h1 {
        font-size: 2.55rem;
        line-height: 1.1;
        margin: 0;
        color: white;
        letter-spacing: 0;
    }

    .hero p {
        margin: 0.7rem 0 0;
        color: rgba(255, 255, 255, 0.92);
        font-size: 1.05rem;
    }

    .section-title {
        font-size: 1.2rem;
        font-weight: 700;
        margin: 0.6rem 0 0.8rem;
        color: #e2e8f0;
    }

    .metric-card {
        padding: 1.1rem;
        border-radius: 14px;
        background: rgba(15, 23, 42, 0.82);
        border: 1px solid rgba(148, 163, 184, 0.20);
        box-shadow: 0 12px 30px rgba(2, 6, 23, 0.30);
        min-height: 116px;
    }

    .metric-label {
        color: #94a3b8;
        font-size: 0.9rem;
        margin-bottom: 0.35rem;
    }

    .metric-value {
        color: white;
        font-size: 1.35rem;
        font-weight: 800;
    }

    .recommendation-card {
        padding: 1rem 1.15rem;
        margin-bottom: 0.8rem;
        border-radius: 14px;
        background: linear-gradient(145deg, rgba(30, 41, 59, 0.94), rgba(15, 23, 42, 0.94));
        border: 1px solid rgba(148, 163, 184, 0.22);
        box-shadow: 0 12px 26px rgba(2, 6, 23, 0.22);
    }

    .song-name {
        font-size: 1.08rem;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 0.2rem;
    }

    .song-meta {
        color: #cbd5e1;
        font-size: 0.92rem;
        margin-bottom: 0.6rem;
    }

    .tag {
        display: inline-block;
        padding: 0.2rem 0.55rem;
        border-radius: 999px;
        margin-right: 0.35rem;
        margin-bottom: 0.2rem;
        background: rgba(14, 165, 233, 0.14);
        color: #bae6fd;
        border: 1px solid rgba(125, 211, 252, 0.25);
        font-size: 0.78rem;
    }

    .success-pop {
        animation: popIn 0.7s ease-out;
        padding: 0.9rem 1rem;
        border-radius: 14px;
        background: rgba(34, 197, 94, 0.13);
        border: 1px solid rgba(74, 222, 128, 0.38);
        color: #dcfce7;
        font-weight: 700;
        margin-bottom: 1rem;
    }

    @keyframes popIn {
        0% { transform: scale(0.96); opacity: 0; }
        60% { transform: scale(1.02); opacity: 1; }
        100% { transform: scale(1); opacity: 1; }
    }

    .footer {
        margin-top: 2rem;
        padding: 1rem 0;
        text-align: center;
        color: #94a3b8;
        border-top: 1px solid rgba(148, 163, 184, 0.2);
    }

    div[data-testid="stProgress"] > div > div > div > div {
        background: linear-gradient(90deg, #22c55e, #06b6d4, #8b5cf6);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# In-code song dataset
# -----------------------------
@st.cache_data
def load_song_data():
    songs = [
        {"song_name": "Sunshine Drive", "artist": "Luna Ray", "mood": "Happy", "energy_level": "High", "genre": "Pop"},
        {"song_name": "Weekend Glow", "artist": "The Neon Club", "mood": "Happy", "energy_level": "Medium", "genre": "Pop"},
        {"song_name": "Smile Again", "artist": "Mira Stone", "mood": "Happy", "energy_level": "Low", "genre": "Lo-fi"},
        {"song_name": "Bright City", "artist": "Nova Beats", "mood": "Happy", "energy_level": "High", "genre": "EDM"},
        {"song_name": "Good Day Anthem", "artist": "Skyline Crew", "mood": "Happy", "energy_level": "High", "genre": "Hip-Hop"},
        {"song_name": "Rain on Glass", "artist": "Evan Blue", "mood": "Sad", "energy_level": "Low", "genre": "Lo-fi"},
        {"song_name": "Empty Platform", "artist": "Noah Vale", "mood": "Sad", "energy_level": "Low", "genre": "Pop"},
        {"song_name": "Fading Letters", "artist": "Iris Lane", "mood": "Sad", "energy_level": "Medium", "genre": "Rock"},
        {"song_name": "Midnight Apology", "artist": "Velvet Echo", "mood": "Sad", "energy_level": "Medium", "genre": "Hip-Hop"},
        {"song_name": "Blue Window", "artist": "Soft Static", "mood": "Sad", "energy_level": "Low", "genre": "Lo-fi"},
        {"song_name": "Ocean Breathing", "artist": "Ari Bloom", "mood": "Calm", "energy_level": "Low", "genre": "Lo-fi"},
        {"song_name": "Quiet Stars", "artist": "Mellow Atlas", "mood": "Calm", "energy_level": "Low", "genre": "Pop"},
        {"song_name": "Tea and Moonlight", "artist": "Nia Fern", "mood": "Calm", "energy_level": "Medium", "genre": "Lo-fi"},
        {"song_name": "Gentle Waves", "artist": "Cloud Harbor", "mood": "Calm", "energy_level": "Low", "genre": "EDM"},
        {"song_name": "Slow Horizon", "artist": "Paper North", "mood": "Calm", "energy_level": "Medium", "genre": "Rock"},
        {"song_name": "Thunder Pulse", "artist": "Axel Riot", "mood": "Energetic", "energy_level": "High", "genre": "Rock"},
        {"song_name": "Bass Launch", "artist": "DJ Orbit", "mood": "Energetic", "energy_level": "High", "genre": "EDM"},
        {"song_name": "Run the Night", "artist": "Metro Kings", "mood": "Energetic", "energy_level": "High", "genre": "Hip-Hop"},
        {"song_name": "Electric Feet", "artist": "Zara Volt", "mood": "Energetic", "energy_level": "Medium", "genre": "Pop"},
        {"song_name": "Fireline", "artist": "The Voltage", "mood": "Energetic", "energy_level": "High", "genre": "Rock"},
        {"song_name": "Rose Lights", "artist": "Sia Moon", "mood": "Romantic", "energy_level": "Medium", "genre": "Pop"},
        {"song_name": "Hold You Close", "artist": "Arman Grey", "mood": "Romantic", "energy_level": "Low", "genre": "Lo-fi"},
        {"song_name": "Velvet Promise", "artist": "June Valley", "mood": "Romantic", "energy_level": "Medium", "genre": "Rock"},
        {"song_name": "Late Night Text", "artist": "Kairo Muse", "mood": "Romantic", "energy_level": "Medium", "genre": "Hip-Hop"},
        {"song_name": "Dancing Hearts", "artist": "Elena Vox", "mood": "Romantic", "energy_level": "High", "genre": "EDM"},
        {"song_name": "Golden Hour Call", "artist": "Maya Sol", "mood": "Romantic", "energy_level": "Low", "genre": "Pop"},
        {"song_name": "Afterparty Bloom", "artist": "Pulse Avenue", "mood": "Happy", "energy_level": "Medium", "genre": "EDM"},
        {"song_name": "Study Lantern", "artist": "Lo Cloud", "mood": "Calm", "energy_level": "Medium", "genre": "Hip-Hop"},
        {"song_name": "Breakout", "artist": "Crimson Field", "mood": "Energetic", "energy_level": "Medium", "genre": "Rock"},
        {"song_name": "Soft Goodbye", "artist": "Amelia Coast", "mood": "Sad", "energy_level": "Medium", "genre": "Pop"},
    ]
    return pd.DataFrame(songs)


# -----------------------------
# Machine learning preparation
# -----------------------------
def train_model(song_df):
    model_df = song_df.copy()
    encoders = {}

    for column in ["mood", "energy_level", "genre"]:
        encoder = LabelEncoder()
        model_df[f"{column}_encoded"] = encoder.fit_transform(model_df[column])
        encoders[column] = encoder

    feature_columns = ["mood_encoded", "energy_level_encoded", "genre_encoded"]
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(model_df[feature_columns])

    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    model_df["cluster"] = kmeans.fit_predict(scaled_features)

    return model_df, encoders, scaler, kmeans, feature_columns


def encode_user_input(mood, energy_level, genre, encoders):
    return np.array(
        [
            encoders["mood"].transform([mood])[0],
            encoders["energy_level"].transform([energy_level])[0],
            encoders["genre"].transform([genre])[0],
        ]
    ).reshape(1, -1)


def recommend_songs(mood, energy_level, genre, model_df, encoders, scaler, kmeans, top_n=5):
    user_input = encode_user_input(mood, energy_level, genre, encoders)
    user_scaled = scaler.transform(user_input)
    selected_cluster = kmeans.predict(user_scaled)[0]

    cluster_songs = model_df[model_df["cluster"] == selected_cluster].copy()
    all_scaled = scaler.transform(cluster_songs[["mood_encoded", "energy_level_encoded", "genre_encoded"]])

    distances = np.linalg.norm(all_scaled - user_scaled, axis=1)
    cluster_songs["distance"] = distances
    max_distance = max(distances.max(), 0.01)
    cluster_songs["confidence"] = ((1 - (cluster_songs["distance"] / (max_distance + 0.01))) * 35 + 65).round(0)

    exact_bonus = (
        (cluster_songs["mood"] == mood).astype(int) * 3
        + (cluster_songs["energy_level"] == energy_level).astype(int) * 2
        + (cluster_songs["genre"] == genre).astype(int) * 2
    )
    cluster_songs["ranking_score"] = cluster_songs["confidence"] + exact_bonus

    recommendations = cluster_songs.sort_values(
        by=["ranking_score", "confidence"],
        ascending=False,
    ).head(top_n)

    return recommendations, selected_cluster


def mood_meter_value(mood, energy_level):
    mood_base = {
        "Sad": 25,
        "Calm": 45,
        "Romantic": 62,
        "Happy": 78,
        "Energetic": 90,
    }
    energy_boost = {"Low": -8, "Medium": 0, "High": 8}
    return int(np.clip(mood_base[mood] + energy_boost[energy_level], 5, 100))


def vibe_text(mood, energy_level, genre):
    vibes = {
        "Happy": "sunny, feel-good, and ready for a bright playlist",
        "Sad": "soft, reflective, and emotionally honest",
        "Calm": "peaceful, focused, and easy on the mind",
        "Energetic": "bold, fast-moving, and built for momentum",
        "Romantic": "warm, dreamy, and full of late-night glow",
    }
    return f"{energy_level} energy {genre} with a {vibes[mood]} vibe"


# -----------------------------
# App layout
# -----------------------------
songs_df = load_song_data()
model_df, label_encoders, feature_scaler, kmeans_model, features = train_model(songs_df)

st.markdown(
    """
    <div class="hero">
        <h1>🎧 Mood-Based Music Recommender System</h1>
        <p>Pick your mood, energy, and genre. Machine learning finds the closest songs using K-Means clustering.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("🎵 Tune Your Vibe")
    st.caption("Choose your current listening preference.")

    selected_mood = st.selectbox(
        "Mood",
        ["Happy", "Sad", "Calm", "Energetic", "Romantic"],
        index=0,
    )
    selected_energy = st.selectbox(
        "Energy Level",
        ["Low", "Medium", "High"],
        index=1,
    )
    selected_genre = st.selectbox(
        "Genre",
        ["Pop", "Lo-fi", "Rock", "Hip-Hop", "EDM"],
        index=0,
    )

    recommend_button = st.button("✨ Recommend Songs", use_container_width=True)

    st.divider()
    st.subheader("ML Workflow")
    st.markdown(
        """
        **Dataset** → song details are collected with mood, energy, and genre.

        **Preprocessing** → text categories are converted into numbers using LabelEncoder.

        **K-Means** → songs are grouped into similar clusters without predefined labels.

        **Recommendation** → your selected vibe is matched with the nearest cluster.
        """
    )

    st.divider()
    st.subheader("About Project")
    st.info(
        "This Streamlit app demonstrates an unsupervised machine learning "
        "music recommendation system for a college project. It avoids external "
        "APIs and works fully with local project data."
    )

recommendations, chosen_cluster = recommend_songs(
    selected_mood,
    selected_energy,
    selected_genre,
    model_df,
    label_encoders,
    feature_scaler,
    kmeans_model,
)

if recommend_button:
    st.balloons()
    st.markdown(
        "<div class='success-pop'>✅ Your personalized playlist is ready. Enjoy the vibe!</div>",
        unsafe_allow_html=True,
    )

mood_score = mood_meter_value(selected_mood, selected_energy)
average_confidence = int(recommendations["confidence"].mean())

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Mood Meter</div>
            <div class="metric-value">{mood_score}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.progress(mood_score)

with col2:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Recommendation Confidence</div>
            <div class="metric-value">{average_confidence}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.progress(average_confidence)

with col3:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Currently Recommended Vibe</div>
            <div class="metric-value">{vibe_text(selected_mood, selected_energy, selected_genre).title()}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

left_col, right_col = st.columns([1.05, 1])

with left_col:
    st.markdown("<div class='section-title'>🎶 Top 5 Recommendations</div>", unsafe_allow_html=True)

    for rank, (_, song) in enumerate(recommendations.iterrows(), start=1):
        st.markdown(
            f"""
            <div class="recommendation-card">
                <div class="song-name">{rank}. {song["song_name"]}</div>
                <div class="song-meta">by {song["artist"]}</div>
                <span class="tag">Mood: {song["mood"]}</span>
                <span class="tag">Energy: {song["energy_level"]}</span>
                <span class="tag">Genre: {song["genre"]}</span>
                <span class="tag">Confidence: {int(song["confidence"])}%</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

with right_col:
    st.markdown("<div class='section-title'>📊 K-Means Song Clusters</div>", unsafe_allow_html=True)

    chart_df = model_df.copy()
    chart_df["hover_info"] = (
        chart_df["song_name"]
        + " by "
        + chart_df["artist"]
        + "<br>Mood: "
        + chart_df["mood"]
        + "<br>Energy: "
        + chart_df["energy_level"]
        + "<br>Genre: "
        + chart_df["genre"]
    )

    fig = px.scatter(
        chart_df,
        x="mood_encoded",
        y="genre_encoded",
        color="cluster",
        size="energy_level_encoded",
        hover_name="song_name",
        hover_data={
            "artist": True,
            "mood": True,
            "energy_level": True,
            "genre": True,
            "mood_encoded": False,
            "genre_encoded": False,
            "energy_level_encoded": False,
            "cluster": True,
        },
        color_continuous_scale="Turbo",
        title="Song Similarity Clusters",
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.35)",
        font=dict(color="#e2e8f0"),
        title_font=dict(size=18),
        margin=dict(l=20, r=20, t=55, b=20),
        coloraxis_colorbar=dict(title="Cluster"),
    )
    fig.update_traces(marker=dict(line=dict(width=1, color="rgba(255,255,255,0.55)")))
    st.plotly_chart(fig, use_container_width=True)

with st.expander("📚 View Complete In-Code Song Dataset"):
    st.dataframe(
        songs_df,
        use_container_width=True,
        hide_index=True,
    )

st.markdown(
    "<div class='footer'>Developed using Streamlit & Machine Learning</div>",
    unsafe_allow_html=True,
)
