from pathlib import Path
import joblib
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import datetime

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="Support Ticket Classifier Dashboard",
    page_icon="🎫",
    layout="wide"
)

# ---------------- PATHS ----------------
MODEL_DIR = Path("models")
CATEGORY_MODEL_PATH = MODEL_DIR / "ticket_category_model.joblib"
PRIORITY_MODEL_PATH = MODEL_DIR / "ticket_priority_model.joblib"

# ---------------- LOAD MODELS ----------------
@st.cache_resource
def load_models():
    return joblib.load(CATEGORY_MODEL_PATH), joblib.load(PRIORITY_MODEL_PATH)

# ---------------- UTILS ----------------
def confidence_table(model, text):
    probs = model.predict_proba([text])[0]
    labels = model.classes_
    df = pd.DataFrame({"Label": labels, "Confidence": probs})
    return df.sort_values("Confidence", ascending=False)

def priority_color(p):
    p = p.lower()
    if p == "high":
        return "🔴 High"
    elif p == "medium":
        return "🟠 Medium"
    return "🟢 Low"

# ---------------- ADVANCED GRAPHS ----------------
def plot_heatmap(df, title):
    fig = go.Figure(data=go.Heatmap(
        z=[df["Confidence"]],
        x=df["Label"],
        y=[title],
        colorscale="Blues",
        text=[[f"{v:.2%}" for v in df["Confidence"]]],
        texttemplate="%{text}"
    ))
    fig.update_layout(template="plotly_dark", height=250, title=title)
    return fig

def plot_gauge(value, title):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value * 100,
        title={'text': title},
        gauge={'axis': {'range': [0, 100]}}
    ))
    fig.update_layout(template="plotly_dark", height=250)
    return fig

def plot_priority_donut(df):
    fig = px.pie(df, names="priority", hole=0.6, title="Priority Distribution")
    fig.update_layout(template="plotly_dark")
    return fig

def plot_category_trend(df):
    trend = df.groupby(["time", "category"]).size().reset_index(name="count")
    fig = px.line(trend, x="time", y="count", color="category",
                  markers=True, title="Category Trend Over Time")
    fig.update_layout(template="plotly_dark")
    return fig

# ---------------- AI INSIGHT ----------------
def generate_insight(text, category, priority):
    t = text.lower()

    if "error" in t or "fail" in t:
        reason = "System failure detected"
    elif "payment" in t or "invoice" in t:
        reason = "Billing issue identified"
    elif "login" in t:
        reason = "Authentication issue"
    else:
        reason = "General support request"

    return f"""
    **Prediction Insight**
    - Category → **{category}**
    - Priority → **{priority}**
    - Reason → {reason}
    """

# ---------------- CHECK ----------------
if not CATEGORY_MODEL_PATH.exists():
    st.error("❌ Train model first using train.py")
    st.stop()

category_model, priority_model = load_models()

# ---------------- SESSION ----------------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.title("⚙️ Control Panel")

    samples = [
        "Server is down for all users",
        "Payment deducted twice",
        "Login not working after update",
        "Need API documentation"
    ]

    sample = st.selectbox("Sample Tickets", samples)

    if st.button("Use Sample"):
        st.session_state["input"] = sample

    st.markdown("---")
    st.metric("Tickets Analyzed", len(st.session_state.history))

# ---------------- MAIN ----------------
st.title("🎫 Support Ticket Classifier Dashboard")
st.caption("Real-time ticket classification & prioritization system")

col1, col2 = st.columns([1.3, 1])

# ================= LEFT PANEL =================
with col1:
    st.subheader("📝 Ticket Input")

    text = st.text_area(
        "Enter support ticket",
        value=st.session_state.get("input", ""),
        height=180
    )

    if st.button("🚀 Analyze Ticket", use_container_width=True):

        if not text.strip():
            st.warning("Please enter ticket text")
        else:
            # Predictions
            category = category_model.predict([text])[0]
            priority = priority_model.predict([text])[0]

            # Confidence
            cat_df = confidence_table(category_model, text)
            pri_df = confidence_table(priority_model, text)

            # Save history
            st.session_state.history.append({
                "time": datetime.datetime.now().strftime("%H:%M:%S"),
                "text": text,
                "category": category,
                "priority": priority
            })

            # KPI
            st.subheader("📌 Results")
            c1, c2 = st.columns(2)
            c1.metric("Category", category)
            c2.metric("Priority", priority_color(priority))

            # AI Insight
            st.subheader("🤖 AI Insight")
            st.info(generate_insight(text, category, priority))

            # ADVANCED ANALYTICS
            st.subheader("📊 Advanced Analytics")

            colA, colB = st.columns(2)

            with colA:
                st.plotly_chart(plot_heatmap(cat_df, "Category Confidence"), use_container_width=True)
                st.plotly_chart(plot_gauge(cat_df.iloc[0]["Confidence"], "Top Category Confidence"), use_container_width=True)

            with colB:
                st.plotly_chart(plot_heatmap(pri_df, "Priority Confidence"), use_container_width=True)
                st.plotly_chart(plot_gauge(pri_df.iloc[0]["Confidence"], "Top Priority Confidence"), use_container_width=True)

# ================= RIGHT PANEL =================
with col2:
    st.subheader("📜 Recent Tickets")

    if st.session_state.history:
        df_hist = pd.DataFrame(st.session_state.history[::-1])

        st.dataframe(df_hist, use_container_width=True)

        st.subheader("📈 Insights Dashboard")

        st.plotly_chart(plot_priority_donut(df_hist), use_container_width=True)
        st.plotly_chart(plot_category_trend(df_hist), use_container_width=True)

    else:
        st.info("No tickets analyzed yet")

    st.subheader("📘 Guidance")
    st.markdown("""
    - Include clear issue description  
    - Mention urgency & impact  
    - Provide system context  
    """)