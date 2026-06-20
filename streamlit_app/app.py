import json
import pickle
from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.express as px

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# --- Custom Styling (Premium Dark Mode) ---
st.set_page_config(page_title="ShopSense ML", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background: radial-gradient(circle at 10% 20%, rgb(18, 18, 22) 0%, rgb(10, 10, 12) 90%);
        color: #E2E8F0;
    }
    
    /* Header styling */
    h1, h2, h3 {
        color: #FFFFFF !important;
        font-weight: 800 !important;
        letter-spacing: -0.5px;
    }
    
    .hero-title {
        background: linear-gradient(90deg, #00C6FF 0%, #0072FF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem !important;
        font-weight: 800 !important;
        margin-bottom: 0px !important;
    }
    
    .hero-subtitle {
        color: #94A3B8;
        font-size: 1.1rem;
        margin-top: -10px;
        margin-bottom: 30px;
    }

    /* Cards */
    div[data-testid="stMetricValue"] {
        font-size: 2rem !important;
        font-weight: 800 !important;
        color: #00C6FF !important;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 1rem !important;
        font-weight: 600 !important;
        color: #94A3B8 !important;
    }
    
    /* Dataframes */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid #334155;
    }
    
    hr {
        border-color: #334155;
        opacity: 0.5;
    }
</style>
""", unsafe_allow_html=True)


# --- Data Loading ---
@st.cache_data
def load_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))

@st.cache_data
def load_pickle(path: Path):
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)

def metric_table(metrics_payload):
    rows = []
    for model, values in metrics_payload.get("metrics", {}).items():
        row = {"Model": model.replace("_", " ").title()}
        row.update({k.upper(): round(v, 4) for k,v in values.items()})
        rows.append(row)
    return pd.DataFrame(rows)

# --- App Layout ---
st.markdown('<h1 class="hero-title">ShopSense Machine Learning</h1>', unsafe_allow_html=True)
st.markdown('<p class="hero-subtitle">Production Dashboard: Retrieval → Ranking → MMR Reranking</p>', unsafe_allow_html=True)

metrics_payload = load_json(PROJECT_ROOT / "reports" / "metrics.json")
segment_payload = load_json(PROJECT_ROOT / "reports" / "segment_metrics.json")
manifest = load_json(PROJECT_ROOT / "artifacts" / "model_manifest.json")

train_df = None
if (PROJECT_ROOT / "data" / "processed" / "train.pkl").exists():
    train_df = pd.read_pickle(PROJECT_ROOT / "data" / "processed" / "train.pkl")
    
reverse_user_mapping = load_pickle(PROJECT_ROOT / "artifacts" / "reverse_user_mapping.pkl")
reverse_item_mapping = load_pickle(PROJECT_ROOT / "artifacts" / "reverse_item_mapping.pkl")

# --- Top Level Metrics ---
if manifest and metrics_payload:
    best_model_name = manifest.get('active_model', 'Hybrid MMR')
    
    # Extract best metrics safely
    best_metrics = metrics_payload.get('metrics', {}).get(best_model_name, {})
    ndcg = best_metrics.get('ndcg@10', 0.0)
    map_score = best_metrics.get('map@10', 0.0)
    recall = best_metrics.get('recall@10', 0.0)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Production Model", best_model_name.upper())
    col2.metric("NDCG@10", f"{ndcg:.4f}")
    col3.metric("MAP@10", f"{map_score:.4f}")
    col4.metric("Recall@10", f"{recall:.4f}")
    st.markdown("<hr>", unsafe_allow_html=True)
else:
    st.warning("⏳ Waiting for evaluate_all.py to finish generating real metrics...")


# --- Sidebar ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1162/1162499.png", width=50) # Tiny generic icon
    st.header("Simulator Controls")
    
    available_users = []
    if train_df is not None and reverse_user_mapping:
        user_indices = train_df["user_idx"].drop_duplicates().head(500).tolist()
        available_users = [str(reverse_user_mapping.get(int(idx), idx)) for idx in user_indices]

    selected_user = st.selectbox("Impersonate User ID", available_users or ["Artifacts not ready"])
    category_filter = st.selectbox("Cold-Start Context", ["None", "Outerwear", "Tops", "Shoes", "Accessories"])
    
    st.markdown("---")
    st.markdown("**System Status**")
    if manifest:
        st.success("✅ Models Synced")
        st.success(f"🤖 Best CF: {manifest.get('best_cf_model', 'ALS').upper()}")
    else:
        st.error("❌ Models Missing")


# --- Main Content Columns ---
col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("🛍️ User Purchase History")
    if train_df is None or not reverse_user_mapping or selected_user == "Artifacts not ready":
        st.info("Processed data required to show history.")
    else:
        user_mapping = {str(v): int(k) for k, v in reverse_user_mapping.items()}
        user_idx = user_mapping.get(selected_user)
        if user_idx is not None:
            history = train_df[train_df["user_idx"] == user_idx].tail(10).copy()
            history["Actual Item ID"] = history["item_idx"].map(lambda item: reverse_item_mapping.get(int(item), item))
            st.dataframe(history[["t_dat", "Actual Item ID", "price"]], use_container_width=True, hide_index=True)

with col_right:
    st.subheader("⚡ Live API Response Preview")
    if manifest and selected_user != "Artifacts not ready":
        with st.container(border=True):
            st.markdown(f"**Endpoint:** `GET /api/v1/recommendations/{selected_user}?k=10`")
            st.markdown(f"**Active Model:** `{manifest.get('active_model')}`")
            st.markdown(f"**Cache Policy:** `Redis TTL 24h`")
            
            # Mocking the JSON response preview
            preview = {
                "status": "success",
                "latency_ms": 42.5,
                "cached": False,
                "items": [
                    {"rank": 1, "item_id": "012345", "score": 0.98},
                    {"rank": 2, "item_id": "067890", "score": 0.85}
                ]
            }
            st.json(preview, expanded=False)
    else:
        st.info("API preview requires training artifacts.")


st.markdown("<hr>", unsafe_allow_html=True)


# --- Evaluation Reports ---
st.subheader("📊 Offline A/B Simulator Results")

if metrics_payload:
    df_metrics = metric_table(metrics_payload)
    st.dataframe(
        df_metrics.style.background_gradient(cmap='Blues', subset=['NDCG@10', 'MAP@10', 'RECALL@10']), 
        use_container_width=True,
        hide_index=True
    )
else:
    st.info("Evaluation metrics missing. Run evaluate_all.py")


st.markdown("<br>", unsafe_allow_html=True)
st.subheader("🎯 Model Performance by User Segment")

if segment_payload and segment_payload.get("metrics"):
    selected_metric = st.selectbox("Select Metric to Visualize", ["ndcg@10", "recall@10", "map@10"])
    
    rows = []
    for segment, model_metrics in segment_payload["metrics"].items():
        for model, values in model_metrics.items():
            rows.append({"Segment": segment.title(), "Model": model.replace("_", " ").title(), "Score": values.get(selected_metric, 0.0)})
            
    segment_df = pd.DataFrame(rows)
    
    if not segment_df.empty:
        # Premium Plotly Chart
        fig = px.bar(
            segment_df, 
            x="Segment", 
            y="Score", 
            color="Model", 
            barmode="group",
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#E2E8F0",
            legend_title_text='Algorithm',
            margin=dict(l=0, r=0, t=30, b=0)
        )
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Segment report missing. Run evaluate_all.py")
