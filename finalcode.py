# ✅ FINALIZED CODE: MindPrint - Decode Personalities, Design Futures (Full Version)
import streamlit as st
import pandas as pd
import numpy as np
import os
import time
from datetime import datetime, timedelta
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

# === Setup ===
st.set_page_config(page_title="Finding You in U", layout="centered")
st.title("🎮 MindPrint: Finding You in 'U'")

questions = {
    "Q1": "I enjoy social gatherings and talking to people.",
    "Q2": "I often plan things out in detail before doing them.",
    "Q3": "I get stressed out easily in unfamiliar situations.",
    "Q4": "I often find myself daydreaming.",
    "Q5": "I prefer staying at home rather than going out.",
    "Q6": "I often seek adventure and excitement.",
    "Q7": "I am sensitive to other people’s feelings.",
    "Q8": "I work well under pressure.",
    "Q9": "I enjoy artistic and creative activities.",
    "Q10": "I prefer sticking to routines rather than changes."
}

q_keys = list(questions.keys())

if 'state' not in st.session_state:
    st.session_state.state = {
        'started': False,
        'name': "",
        'q_index': 0,
        'responses': [None] * len(questions),
        'start_time': time.time(),
        'quiz_done': False,
        'completed_time': None,
        'submitted': False,
        'method': "",
        'results': {},
    }

ss = st.session_state.state

cluster_names = [
    "🌛 Dreamer Adventurer", "🌈 Creative Explorer",
    "📚 Introverted Planner", "🏄️ Calm Empath", "⚖️ Balanced Achiever"
]
cluster_map = {i: cluster_names[i % len(cluster_names)] for i in range(5)}

color_palette = {
    "🌈 Creative Explorer": "#e67e22",
    "📚 Introverted Planner": "#2980b9",
    "🌛 Dreamer Adventurer": "#8e44ad",
    "🏄️ Calm Empath": "#16a085",
    "⚖️ Balanced Achiever": "#2c3e50"
}

career_tips = {
    "🌈 Creative Explorer": "Great for Design, Innovation, Branding",
    "📚 Introverted Planner": "Perfect for Operations, QA, Analysis",
    "🌛 Dreamer Adventurer": "Ideal for Marketing, Social Media, Storytelling",
    "🏄️ Calm Empath": "Great for HR, Counseling, Customer Relations",
    "⚖️ Balanced Achiever": "Suited for Leadership, Project Management"
}

# === Helper Functions ===
def save_submission(name, personality, responses, method):
    timestamp = datetime.now().isoformat()
    data = {"Name": name, "Personality": personality, "Timestamp": timestamp, "Method": method}
    data.update({f"A{i+1}": v for i, v in enumerate(responses)})
    df_new = pd.DataFrame([data])

    if os.path.exists("submissions.csv"):
        df_old = pd.read_csv("submissions.csv")
        df_all = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_all = df_new

    df_all.to_csv("submissions.csv", index=False)
    df_all[[f"A{i+1}" for i in range(10)]].to_csv("clustering_data.csv", index=False)

def reset_session():
    for key in list(st.session_state.state.keys()):
        st.session_state.state[key] = None
    st.rerun()

def show_personality_card(personality):
    st.markdown(f"""
        <div style='background-color:{color_palette.get(personality, '#7f8c8d')};
             padding:15px;border-radius:10px;color:white;text-align:center;font-size:20px;'>
        <strong>{personality}</strong><br>
        {career_tips.get(personality, '')}
        </div>
    """, unsafe_allow_html=True)

def radar_chart(responses, name="Your Profile"):
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=responses + [responses[0]],
        theta=[f"A{i+1}" for i in range(10)] + ["A1"],
        fill='toself',
        name=name
    ))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=False)
    st.plotly_chart(fig)

# === CSV Upload + Training Visualization ===
st.sidebar.header("📂 Upload CSV to Train Clustering Models")
uploaded_file = st.sidebar.file_uploader("Upload CSV with A1 to A10 columns", type="csv")
best_model_name = None
best_model_score = -1
model_scores = {}

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    if all(f"A{i+1}" in df.columns for i in range(10)):
        df_clean = df[[f"A{i+1}" for i in range(10)]].dropna()
        X_scaled = StandardScaler().fit_transform(df_clean)

        st.sidebar.markdown("### Clustering Model Results:")
        clustering_models = {
            "KMeans": KMeans(n_clusters=5, random_state=42, n_init='auto'),
            "Agglomerative": AgglomerativeClustering(n_clusters=5),
            "DBSCAN": DBSCAN(eps=0.5, min_samples=3),
            "Spectral": SpectralClustering(n_clusters=5, random_state=42, assign_labels='discretize'),
            "GMM": GaussianMixture(n_components=5, random_state=42)
        }

        for model_name, model in clustering_models.items():
            try:
                if model_name == "GMM":
                    labels = model.fit(X_scaled).predict(X_scaled)
                else:
                    labels = model.fit_predict(X_scaled)

                if len(set(labels)) > 1:
                    score = silhouette_score(X_scaled, labels)
                    model_scores[model_name] = score
                    st.sidebar.markdown(f"**{model_name}** - Silhouette Score: `{score:.3f}`")

                    fig, ax = plt.subplots()
                    sns.countplot(x=labels, ax=ax, palette="viridis")
                    ax.set_title(f"{model_name} Cluster Distribution")
                    st.sidebar.pyplot(fig)

                    if model_name == "KMeans":
                        pca = PCA(n_components=2)
                        reduced_data = pca.fit_transform(X_scaled)
                        fig2, ax2 = plt.subplots()
                        scatter = ax2.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, cmap='viridis', alpha=0.6)
                        ax2.set_title("2D PCA Plot - KMeans Clusters")
                        ax2.set_xlabel("PCA Component 1")
                        ax2.set_ylabel("PCA Component 2")
                        st.sidebar.pyplot(fig2)

                    if score > best_model_score:
                        best_model_score = score
                        best_model_name = model_name
            except Exception as e:
                st.sidebar.warning(f"{model_name} failed: {str(e)}")

        if best_model_name:
            st.sidebar.success(f"🏆 Best Performing Model: **{best_model_name}** (Score: {best_model_score:.3f})")

        if st.sidebar.button("💾 Save Cleaned Training Data"):
            df_clean.to_csv("trained_data.csv", index=False)
            st.sidebar.success("Training data saved as 'trained_data.csv'")

        if model_scores:
            st.sidebar.markdown("### 📊 Model Score Comparison")
            score_fig, ax = plt.subplots()
            ax.bar(model_scores.keys(), model_scores.values(), color='skyblue')
            ax.set_ylabel("Silhouette Score")
            ax.set_title("Model Performance")
            st.sidebar.pyplot(score_fig)
# === Leaderboard Reset Option ===
with st.sidebar.expander("🧹 Manage Leaderboard"):
    if st.button("Clear Leaderboard"):
        if os.path.exists("submissions.csv"):
            os.remove("submissions.csv")
        if os.path.exists("clustering_data.csv"):
            os.remove("clustering_data.csv")
        st.success("Leaderboard cleared successfully!")
        st.experimental_rerun()
# === Quiz Flow ===
if not ss['started']:
    name_input = st.text_input("👤 Enter Your Name to Begin:")
    if st.button("Start Quiz"):
        if not name_input.strip():
            st.warning("🚫 Please enter your name to start.")
        else:
            ss['name'] = name_input.strip()
            ss['started'] = True
            ss['start_time'] = time.time()
            st.rerun()

elif not ss['quiz_done']:
    q_idx = ss['q_index']
    time_left = 30 - int(time.time() - ss['start_time'])
    if time_left <= 0:
        ss['q_index'] += 1
        ss['start_time'] = time.time()
        if ss['q_index'] >= len(q_keys):
            ss['quiz_done'] = True
            ss['completed_time'] = time.time()
        st.rerun()
    else:
        st.progress(int((q_idx + 1) / len(q_keys) * 100))
        st.info(f"⏰ Time Left: {time_left}s")
        st.subheader(f"Q{q_idx + 1}: {questions[q_keys[q_idx]]}")
        response = st.slider("Your answer:", 1, 5, 3, key=q_keys[q_idx])
        ss['responses'][q_idx] = response
        if st.button("Next"):
            if ss['responses'][q_idx] is not None:
                ss['q_index'] += 1
                ss['start_time'] = time.time()
                if ss['q_index'] >= len(q_keys):
                    ss['quiz_done'] = True
                    ss['completed_time'] = time.time()
                st.rerun()
            else:
                st.warning("⚠️ Please answer before continuing.")

# === Results ===
else:
    st.subheader("✅ All questions answered!")
    method_choice = st.radio("🔍 Prediction Method:", ["Logic-Based", "KMeans"], horizontal=True)
    ss['method'] = method_choice
    user_df = pd.DataFrame([ss['responses']], columns=[f"A{i+1}" for i in range(10)])
    personality = None

    with st.expander("ℹ️ What does each model do?"):
        st.markdown("""
        - **Logic-Based:** Uses fixed trait-based scores.
        - **KMeans:** Finds clusters of similar people automatically.
        """)

    if method_choice == "Logic-Based":
        scores = {
            "🌈 Creative Explorer": np.mean([user_df["A4"][0], user_df["A9"][0]]),
            "📚 Introverted Planner": np.mean([user_df["A2"][0], user_df["A5"][0], user_df["A10"][0]]),
            "🌛 Dreamer Adventurer": np.mean([user_df["A1"][0], user_df["A6"][0]]),
            "🏄️ Calm Empath": np.mean([user_df["A3"][0], user_df["A7"][0]]),
            "⚖️ Balanced Achiever": user_df["A8"][0]
        }
        personality = max(scores, key=scores.get)
        show_personality_card(personality)
        radar_chart(ss['responses'])

    else:
        if os.path.exists("clustering_data.csv"):
            df = pd.read_csv("clustering_data.csv")
            df_clean = df.dropna()
            X_scaled = StandardScaler().fit_transform(df_clean)
            kmeans = KMeans(n_clusters=5, random_state=42, n_init='auto')
            labels = kmeans.fit_predict(X_scaled)
            label = kmeans.predict(StandardScaler().fit(df_clean).transform(user_df))[0]
            personality = cluster_map[label % len(cluster_names)]
            show_personality_card(personality)
            radar_chart(ss['responses'])
            st.markdown("#### KMeans Cluster Distribution")
            fig, ax = plt.subplots()
            sns.countplot(x=labels, ax=ax, palette="viridis")
            ax.set_title("Cluster Distribution")
            st.pyplot(fig)

            # PCA Visualization
            pca = PCA(n_components=2)
            reduced = pca.fit_transform(X_scaled)
            user_p = pca.transform(StandardScaler().fit_transform(user_df))
            plt.figure(figsize=(6, 4))
            plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap='viridis', alpha=0.6, label="Clusters")
            plt.scatter(user_p[0][0], user_p[0][1], c='red', s=120, label="You", marker="X")
            plt.title("2D PCA Visualization")
            plt.legend()
            st.pyplot(plt)

    if personality and not ss['submitted']:
        save_submission(ss['name'], personality, ss['responses'], method_choice)
        ss['submitted'] = True

    if st.button("🔄 Reset for Next Player"):
        reset_session()

# === Leaderboard ===
st.header("🏆 Leaderboard (Last 1 Hour)")
if os.path.exists("submissions.csv"):
    df = pd.read_csv("submissions.csv")
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df = df[df['Timestamp'] >= datetime.now() - timedelta(hours=1)]
    if not df.empty:
        df = df.drop_duplicates(subset='Name', keep='first')
        df['ID'] = range(1, len(df) + 1)
        df['Current Player'] = df['Name'].apply(lambda x: "🟢 You" if x == ss['name'] else "")
        st.dataframe(
            df[['ID', 'Name', 'Personality', 'Method', 'Current Player'] + [f"A{i+1}" for i in range(10)]].style
            .applymap(lambda v: "background-color: #dff0d8" if v == "🟢 You" else "", subset=["Current Player"])
        )
        st.download_button("📄 Download Leaderboard", df.to_csv(index=False), "leaderboard.csv")
    else:
        st.info("No recent submissions found.")
else:
    st.info("No submission data available yet.")
