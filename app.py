import streamlit as st
import plotly.graph_objects as go
from transformers.pipelines import pipeline
import praw

# --- KONFIG ---
st.set_page_config(page_title="Reddit AI", layout="wide")
st.title("AI Reddit Sentiment")
st.markdown("**FinBERT analyserer live kommentarer fra *r/WallStreetBets***")

# --- AI-MODEL ---
@st.cache_resource
def load_ai():
    with st.spinner("Henter AI-model... (kun første gang)"):
        return pipeline("text-classification", model="yiyanghkust/finbert-tone")

ai = load_ai()

# --- REDDIT KLIENT ---
@st.cache_resource
def get_reddit_client():
    return praw.Reddit(
        client_id=st.secrets["reddit"]["client_id"],
        client_secret=st.secrets["reddit"]["client_secret"],
        user_agent=st.secrets["reddit"]["user_agent"],
    )

# --- HENT KOMMENTARER FRA OFFICIEL REDDIT API ---
@st.cache_data(ttl=180)
def get_reddit_sentiment(symbol: str):
    reddit = get_reddit_client()
    subreddit = reddit.subreddit("wallstreetbets")
    comments = []

    try:
        # Find nye posts der nævner symbolet
        for submission in subreddit.search(f"${symbol}", sort="new", limit=5):
            submission.comments.replace_more(limit=0)
            for c in submission.comments.list():
                text = getattr(c, "body", "")
                if len(text) > 15:
                    comments.append(text)
                    if len(comments) >= 30:
                        break
            if len(comments) >= 30:
                break

        if not comments:
            return 0.0, "Ingen kommentarer fundet lige nu"

        scores, analyzed = [], []
        for text in comments[:7]:
            try:
                result = ai(text)[0]
                label, score = result["label"], result["score"]
                if label == "Positive":
                    scores.append(score)
                elif label == "Negative":
                    scores.append(-score)
                analyzed.append((text, label, score))
            except Exception:
                continue

        avg = sum(scores) / len(scores) if scores else 0.0
        return avg, analyzed

    except Exception as e:
        return 0.0, f"Reddit fejl: {str(e)[:120]}"

# --- 3 AKTIER ---
stocks = ["GME", "TSLA", "NVDA"]
names = ["GameStop", "Tesla", "Nvidia"]

# --- DASHBOARD ---
for name, symbol in zip(names, stocks):
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader(f"{name} (${symbol})")

        with st.spinner("Henter Reddit..."):
            sentiment, analysis = get_reddit_sentiment(symbol)

        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=sentiment * 100,
                title={"text": "AI Sentiment"},
                gauge={
                    "axis": {"range": [-100, 100]},
                    "bar": {
                        "color": "lime" if sentiment > 0.1 else "red" if sentiment < -0.1 else "gray"
                    },
                },
            )
        )
        st.plotly_chart(fig, width="stretch", key=f"gauge_{symbol}")

    with col2:
        with st.expander("AI-analyse af Reddit-kommentarer"):
            if isinstance(analysis, str):
                st.info(analysis)
            else:
                for text, label, score in analysis:
                    emoji = (
                        "Bullish" if label == "Positive"
                        else "Bearish" if label == "Negative"
                        else "Neutral"
                    )
                    st.write(f"{emoji} **{label}** ({score:.2f})")
                    st.caption(text[:120] + "..." if len(text) > 120 else text)

# --- STATUS ---
st.success("LIVE Reddit + AI kører 🚀")
st.info("Data gemmes i cache i 3 minutter (ttl=180).")
