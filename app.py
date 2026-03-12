import streamlit as st
import pandas as pd
import pickle
from textblob import TextBlob
import matplotlib.pyplot as plt
from preprocess import clean_text

# =========================
# Load ML Model
# =========================

model = pickle.load(open("models/ticket_model.pkl", "rb"))
vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))

# =========================
# Page Title
# =========================

st.title("🚀 AI Support Ticket Intelligence Dashboard")

# =========================
# Sample Tickets
# =========================

tickets = [
"Unable to login to my account",
"Password reset link not working",
"Refund not received for cancelled order",
"Billing amount is incorrect",
"Server crashing when uploading files",
"Application loading very slowly",
"Payment gateway not working",
"Subscription charged twice",
"Account locked after multiple login attempts",
"Technical error while submitting form",
"Website showing 500 error",
"Cannot update profile details",
"Refund not processed",
"Invoice download failing",
"Database connection error",
"App crashes when clicking submit",
"Billing page not loading",
"Unable to cancel subscription",
"File upload failing",
"Customer support not responding"
]

# =========================
# Prediction Function
# =========================

results = []

for ticket in tickets:

    cleaned = clean_text(ticket)
    vec = vectorizer.transform([cleaned])

    category = model.predict(vec)[0]

    # Priority Logic
    if category == "Technical":
        priority = "High"
    elif category == "Billing":
        priority = "Medium"
    else:
        priority = "Low"

    # Sentiment Analysis
    polarity = TextBlob(ticket).sentiment.polarity

    if polarity < 0:
        sentiment = "Negative"
    elif polarity == 0:
        sentiment = "Neutral"
    else:
        sentiment = "Positive"

    results.append([ticket, category, priority, sentiment])

# =========================
# DataFrame
# =========================

df = pd.DataFrame(results, columns=[
"Ticket",
"Category",
"Priority",
"Sentiment"
])

# ===============================
# Dashboard Metrics
# ===============================

st.markdown("## 📊 Ticket Analytics Overview")

total_tickets = len(df)

category_counts = df["Category"].value_counts()
priority_counts = df["Priority"].value_counts()
sentiment_counts = df["Sentiment"].value_counts()

col1, col2, col3, col4 = st.columns(4)

# Total Tickets
col1.metric("Total Tickets", total_tickets)

# Categories
with col2:
    st.markdown("### 📂 Categories")
    for cat, count in category_counts.items():
        st.write(f"{cat}: {count}")

# Priority
with col3:
    st.markdown("### ⚡ Priority")
    st.write(f"🔴 High: {priority_counts.get('High',0)}")
    st.write(f"🟡 Medium: {priority_counts.get('Medium',0)}")
    st.write(f"🔵 Low: {priority_counts.get('Low',0)}")

# Sentiment
with col4:
    st.markdown("### 💬 Sentiment")
    st.write(f"😡 Negative: {sentiment_counts.get('Negative',0)}")
    st.write(f"😐 Neutral: {sentiment_counts.get('Neutral',0)}")
    st.write(f"😊 Positive: {sentiment_counts.get('Positive',0)}")

# =========================
# Ticket Table
# =========================

st.markdown("## 📋 Ticket Analysis Table")
st.dataframe(df)

# =========================
# Priority Distribution Chart
# =========================

st.markdown("## ⚡ Priority Distribution")

priority_order = ["High", "Medium", "Low"]
priority_values = [priority_counts.get(p,0) for p in priority_order]

fig1, ax1 = plt.subplots()

colors = ["red", "yellow", "blue"]

ax1.bar(priority_order, priority_values, color=colors)

ax1.set_xlabel("Priority")
ax1.set_ylabel("Number of Tickets")
ax1.set_title("Priority Distribution")

st.pyplot(fig1)

# =========================
# Sentiment Pie Chart
# =========================

st.markdown("## 💬 Sentiment Analysis")

sentiment_values = sentiment_counts.values
sentiment_labels = sentiment_counts.index

fig2, ax2 = plt.subplots()

ax2.pie(
sentiment_values,
labels=sentiment_labels,
autopct="%1.1f%%"
)

ax2.set_title("Customer Sentiment")

st.pyplot(fig2)

# =========================
# Download CSV
# =========================

csv = df.to_csv(index=False)

st.download_button(
"📥 Download Report",
csv,
"ticket_analysis.csv",
"text/csv"
)