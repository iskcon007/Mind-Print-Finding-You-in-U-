# ✅ FINALIZED CODE: MindPrint - Decode Personalities, Design Futures
# (Best-Model Integrated + Data Sufficiency Guardrails)

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

# =========================
# App Setup & Constants
# =========================
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
cols_A = [f"A{i+1}" for i in range(10)]

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

# persist best model name chosen in sidebar training
if "best_model_name" not in st.session_state:
    st.session_state["best_model_name"] = "KMeans"

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

# =========================
# Data Sufficiency Helper
# =========================
def kmeans_data_requirements(n_rows, k, n_features):
    """
    Returns (ok_to_run, hard_min, soft_target, message)

    - hard_min: minimum rows before we block K-Means (e.g., 2*k or 10 total)
    - soft_target: recommended rows for stability (e.g., 10*k, 5*features, or 30)

    Rules of thumb:
      • Absolute minimum: n_rows >= k (algorithmic requirement)
      • Safer minimum:    n_rows >= 2*k (we enforce this as hard_min)
      • Stable target:    n_rows >= max(10*k, 5*features, 30)
    """
    hard_min = max(2 * k, 10)
    soft_target = max(10 * k, 5 * n_features, 30)
    if n_rows < hard_min:
        msg = (f"📉 K-Means needs more data. You have **{n_rows}** rows; "
               f"need at least **{hard_min}** for k={k}. "
               f"Upload more plays or a bigger CSV for reliable clusters.")
        return False, hard_min, soft_target, msg
    msg = (f"ℹ️ You have **{n_rows}** rows. For k={k}, we recommend **≥ {soft_target}** rows "
           f"(more rows → more stable clusters).")
    return True, hard_min, soft_target, msg

# =========================
# Helpers
# =========================
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
    st.plotly_chart(fig, use_container_width=True)

# =========================
# Sidebar: CSV Training + Model Selection
# =========================
st.sidebar.header("📂 Upload CSV to Train Clustering Models")
uploaded_file = st.sidebar.file_uploader("Upload CSV with A1 to A10 columns", type="csv")
best_model_name = None
best_model_score = -1.0
model_scores = {}

if uploaded_file:
    df_train = pd.read_csv(uploaded_file)
    if all(c in df_train.columns for c in cols_A):
        df_clean = df_train[cols_A].copy()
        df_clean = df_clean.fillna(df_clean.median(numeric_only=True))
        n_rows_train = len(df_clean)

        # --- Data sufficiency check for training/evaluation ---
        ok, hard_min, soft_target, req_msg = kmeans_data_requirements(
            n_rows_train, k=5, n_features=df_clean.shape[1]
        )
        if not ok:
            st.sidebar.error(req_msg)
            st.stop()  # stop script now; user needs to upload more data

        if n_rows_train < soft_target:
            st.sidebar.warning(req_msg)
        else:
            st.sidebar.info(req_msg)

        # --- Evaluate models only if sufficient rows ---
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
                    if model_name == "KMeans":
                        try:
                            model.set_params(n_init="auto")
                        except Exception:
                            model.set_params(n_init=10)
                    labels = model.fit_predict(X_scaled)

                # silhouette valid only if >1 cluster label present
                if len(set(labels)) > 1:
                    score = silhouette_score(X_scaled, labels)
                    model_scores[model_name] = score
                    st.sidebar.markdown(f"**{model_name}** — Silhouette: `{score:.3f}`")

                    # Distribution
                    fig, ax = plt.subplots()
                    sns.countplot(x=labels, ax=ax, palette="viridis")
                    ax.set_title(f"{model_name} Cluster Distribution")
                    st.sidebar.pyplot(fig)

                    # PCA preview for KMeans
                    if model_name == "KMeans":
                        pca = PCA(n_components=2, random_state=42)
                        reduced = pca.fit_transform(X_scaled)
                        fig2, ax2 = plt.subplots()
                        ax2.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap='viridis', alpha=0.6)
                        ax2.set_title("2D PCA - KMeans Clusters")
                        ax2.set_xlabel("PCA 1"); ax2.set_ylabel("PCA 2")
                        st.sidebar.pyplot(fig2)

                    if score > best_model_score:
                        best_model_score = score
                        best_model_name = model_name
            except Exception as e:
                st.sidebar.warning(f"{model_name} failed: {str(e)}")
    else:
        st.sidebar.error("CSV must contain columns A1..A10.")

    # Persist best model choice
    if best_model_name:
        st.sidebar.success(f"🏆 Best Model: **{best_model_name}** (Silhouette {best_model_score:.3f})")
        st.session_state["best_model_name"] = best_model_name
    else:
        st.session_state["best_model_name"] = st.session_state.get("best_model_name", "KMeans")

# Save the cleaned data (optional)
if uploaded_file and 'df_clean' in locals() and st.sidebar.button("💾 Save Cleaned Training Data"):
    df_clean.to_csv("trained_data.csv", index=False)
    st.sidebar.success("Training data saved as 'trained_data.csv'")

# Compare model scores bar
if model_scores:
    st.sidebar.markdown("### 📊 Model Score Comparison")
    fig_scores, ax_scores = plt.subplots()
    ax_scores.bar(model_scores.keys(), model_scores.values())
    ax_scores.set_ylabel("Silhouette Score")
    ax_scores.set_title("Model Performance")
    st.sidebar.pyplot(fig_scores)

# Sidebar: Leaderboard maintenance
with st.sidebar.expander("🧹 Manage Leaderboard"):
    if st.button("Clear Leaderboard"):
        if os.path.exists("submissions.csv"):
            os.remove("submissions.csv")
        if os.path.exists("clustering_data.csv"):
            os.remove("clustering_data.csv")
        st.success("Leaderboard cleared successfully!")
        st.experimental_rerun()

# =========================
# Quiz Flow
# =========================
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

# =========================
# Results
# =========================
else:
    st.subheader("✅ All questions answered!")
    method_choice = st.radio("🔍 Prediction Method:",
                             ["Logic-Based", "Data-Driven (Best Model)"],
                             horizontal=True)
    ss['method'] = method_choice
    user_df = pd.DataFrame([ss['responses']], columns=cols_A)
    personality = None

    with st.expander("ℹ️ What does each model do?"):
        st.markdown("""
        - **Logic-Based:** Uses fixed trait-based scores.
        - **Data-Driven (Best Model):** Uses the best clustering model selected in the sidebar (KMeans/GMM/Agglomerative/Spectral/DBSCAN).
        """)

    # ---------- Logic-Based ----------
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

    # ---------- Data-Driven (Best Model) ----------
    else:
        best_name = st.session_state.get("best_model_name", "KMeans")

        if os.path.exists("clustering_data.csv"):
            df_hist = pd.read_csv("clustering_data.csv")
            base = df_hist[cols_A].copy()
            base = base.fillna(base.median(numeric_only=True))
            n_rows = len(base)

            # --- Data sufficiency guard (for runtime prediction/visualization) ---
            planned_k = 5
            ok, hard_min, soft_target, req_msg = kmeans_data_requirements(
                n_rows, k=planned_k, n_features=base.shape[1]
            )
            if not ok:
                st.error(req_msg + " Showing Logic-Based result instead.")
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
                st.stop()
            elif n_rows < soft_target:
                st.warning(req_msg)
            else:
                st.info(req_msg)

            # proceed with modeling
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(base)

            user_num = user_df[cols_A].copy()
            user_num = user_num.fillna(base.median(numeric_only=True))
            user_scaled = scaler.transform(user_num)

            # choose valid k
            k = max(2, min(5, n_rows))

            # Visualization helper
            def visualize(labels, title_suffix=""):
                st.markdown("#### Cluster Distribution")
                fig, ax = plt.subplots()
                sns.countplot(x=labels, ax=ax, palette="viridis")
                ax.set_title(f"Cluster Distribution {title_suffix} (n={n_rows})")
                st.pyplot(fig)

                pca = PCA(n_components=2, random_state=42)
                reduced = pca.fit_transform(X_scaled)
                user_p = pca.transform(user_scaled)
                plt.figure(figsize=(6, 4))
                plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, alpha=0.6)
                plt.scatter(user_p[0][0], user_p[0][1], c='red', s=120, label="You", marker="X")
                plt.title(f"2D PCA Visualization {title_suffix}")
                plt.legend()
                st.pyplot(plt)

            # Fit + predict depending on best model name
            name = (best_name or "KMeans").strip().lower()

            if name == "kmeans":
                try:
                    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
                except TypeError:
                    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(X_scaled)
                user_label = int(kmeans.predict(user_scaled)[0])
                visualize(labels, f"(KMeans, k={k})")

            elif name == "gmm":
                gm = GaussianMixture(n_components=k, random_state=42)
                gm.fit(X_scaled)
                labels = gm.predict(X_scaled)
                user_label = int(gm.predict(user_scaled)[0])
                visualize(labels, f"(GMM, k={k})")

            elif name == "agglomerative":
                X_stack = np.vstack([X_scaled, user_scaled])
                agg = AgglomerativeClustering(n_clusters=k)
                labels_all = agg.fit_predict(X_stack)
                labels, user_label = labels_all[:-1], int(labels_all[-1])
                visualize(labels, f"(Agglomerative, k={k})")

            elif name == "spectral":
                X_stack = np.vstack([X_scaled, user_scaled])
                spec = SpectralClustering(n_clusters=k, random_state=42, assign_labels="discretize")
                labels_all = spec.fit_predict(X_stack)
                labels, user_label = labels_all[:-1], int(labels_all[-1])
                visualize(labels, f"(Spectral, k={k})")

            elif name == "dbscan":
                X_stack = np.vstack([X_scaled, user_scaled])
                db = DBSCAN(eps=0.5, min_samples=3)
                labels_all = db.fit_predict(X_stack)
                labels, user_label = labels_all[:-1], int(labels_all[-1])
                visualize(labels, "(DBSCAN)")

            else:  # fallback to KMeans
                try:
                    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
                except TypeError:
                    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(X_scaled)
                user_label = int(kmeans.predict(user_scaled)[0])
                visualize(labels, f"(KMeans, k={k})")

            personality = cluster_map[user_label % len(cluster_names)]
            show_personality_card(personality)
            radar_chart(ss['responses'])
        else:
            st.info("No training data found yet. Run a few plays or upload CSV in the sidebar.")

    # Save submission once per result
    if personality and not ss['submitted']:
        save_submission(ss['name'], personality, ss['responses'], method_choice)
        ss['submitted'] = True

    if st.button("🔄 Reset for Next Player"):
        reset_session()

# =========================
# Leaderboard
# =========================
st.header("🏆 Leaderboard (Last 1 Hour)")
if os.path.exists("submissions.csv"):
    dfl = pd.read_csv("submissions.csv")
    dfl['Timestamp'] = pd.to_datetime(dfl['Timestamp'])
    dfl = dfl[dfl['Timestamp'] >= datetime.now() - timedelta(hours=1)]
    if not dfl.empty:
        dfl = dfl.drop_duplicates(subset='Name', keep='first')
        dfl['ID'] = range(1, len(dfl) + 1)
        dfl['Current Player'] = dfl['Name'].apply(lambda x: "🟢 You" if x == ss['name'] else "")
        st.dataframe(
            dfl[['ID', 'Name', 'Personality', 'Method', 'Current Player'] + cols_A].style
            .applymap(lambda v: "background-color: #dff0d8" if v == "🟢 You" else "", subset=["Current Player"])
        )
        st.download_button("📄 Download Leaderboard", dfl.to_csv(index=False), "leaderboard.csv")
    else:
        st.info("No recent submissions found.")
else:
    st.info("No submission data available yet.")
