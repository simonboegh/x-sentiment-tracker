import streamlit as st
import plotly.graph_objects as go
from transformers.pipelines import pipeline
import praw

# --- KONFIG ---
st.set_page_config(page_title="Reddit AI", layout="wide")
st.title("AI Reddit Sentiment")
st.markdown("**FinBERT analyserer live kommentarer fra *r/WallStreetBets***")

# --- AI-MODEL (FinBERT) ---
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

def score_to_text(score_100: int) -> str:
    """Omsætter -100..100 til en kort tekst."""
    if score_100 >= 40:
        return "meget bullish"
    elif score_100 >= 15:
        return "bullish"
    elif score_100 > -15:
        return "næsten neutral / blandet"
    elif score_100 > -40:
        return "bearish"
    else:
        return "meget bearish"

# --- HENT KOMMENTARER VIA OFFICIEL REDDIT API ---
@st.cache_data(ttl=180)  # cache 3 minutter
def get_reddit_sentiment(symbol: str):
    reddit = get_reddit_client()
    subreddit = reddit.subreddit("wallstreetbets")
    comments = []

    blocked_phrases = [
        "User Report",
        "Total Submissions",
        "First Seen In WSB",
        "Report generated",
        "moderator of this subreddit",
    ]

    try:
        # 1) Hent rå-kommentarer
        for submission in subreddit.search(f"${symbol}", sort="new", limit=5):
            submission.comments.replace_more(limit=0)
            for c in submission.comments.list():
                text = getattr(c, "body", "")
                if len(text) < 20 or len(text) > 800:
                    continue
                if any(bad in text for bad in blocked_phrases):
                    continue
                comments.append(text)
                if len(comments) >= 40:
                    break
            if len(comments) >= 40:
                break

        if not comments:
            return 0, "Ingen kommentarer fundet lige nu", 0, 0, 0, 0

        analyzed = []

        # 2) Kør FinBERT
        for text in comments[:15]:
            try:
                result = ai(text)[0]
                label = result["label"].lower()  # "positive", "negative", "neutral"
                conf = result["score"]

                if label == "positive":
                    sentiment_word = "Bullish"
                elif label == "negative":
                    sentiment_word = "Bearish"
                else:
                    sentiment_word = "Neutral"

                analyzed.append((text, sentiment_word, conf))
            except Exception:
                continue

        if not analyzed:
            return 0, "Kunne ikke analysere kommentarer lige nu", 0, 0, 0, 0

        # 3) Tæl bullish / bearish / neutral
        n_bull = sum(1 for _, s, _ in analyzed if s == "Bullish")
        n_bear = sum(1 for _, s, _ in analyzed if s == "Bearish")
        n_neutral = sum(1 for _, s, _ in analyzed if s == "Neutral")
        n_total = n_bull + n_bear + n_neutral

        if n_bull + n_bear > 0:
            score_100 = round(100 * (n_bull - n_bear) / (n_bull + n_bear))
        else:
            score_100 = 0

        analyzed_sorted = sorted(analyzed, key=lambda x: x[2], reverse=True)

        # Returnér også counts
        return score_100, analyzed_sorted[:8], n_total, n_bull, n_bear, n_neutral

    except Exception as e:
        return 0, f"Reddit fejl: {str(e)[:120]}", 0, 0, 0, 0

# --- AKTIER I DASHBOARD ---
stocks = ["GME", "TSLA", "NVDA"]
names = ["GameStop", "Tesla", "Nvidia"]

# --- DASHBOARD ---
for name, symbol in zip(names, stocks):
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader(f"{name} (${symbol})")

        with st.spinner("Henter Reddit..."):
            score_100, analysis, n_total, n_bull, n_bear, n_neutral = get_reddit_sentiment(symbol)

        sentiment_text = score_to_text(score_100)
        st.markdown(
            f"**WSB er {sentiment_text} på `{symbol}` lige nu.**  \n"
            f"Sentimentscoren er **{score_100}**, hvor "
            "**-100** betyder kun bearish kommentarer, **0** betyder lige mange bullish og bearish, "
            "og **+100** betyder kun bullish."
        )

        # Vis hvor mange kommentarer der ligger bag
        st.caption(
            f"Analyseret **{n_total}** kommentarer: "
            f"🐂 **{n_bull} bullish**, 🐻 **{n_bear} bearish**, 😶 **{n_neutral} neutrale**."
        )

        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=score_100,
                title={"text": "Netto-bullish sentiment (-100 til +100)"},
                gauge={
                    "axis": {"range": [-100, 100]},
                    "bar": {
                        "color": "lime"
                        if score_100 > 10
                        else "red"
                        if score_100 < -10
                        else "gray"
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
                for text, sentiment_word, conf in analysis:
                    if sentiment_word == "Bullish":
                        emoji = "🐂"
                    elif sentiment_word == "Bearish":
                        emoji = "🐻"
                    else:
                        emoji = "😶"

                    st.write(f"{emoji} **{sentiment_word}** (model-sikkerhed: {conf:.2f})")
                    st.caption(text[:220] + "..." if len(text) > 220 else text)
                    st.markdown("---")

# --- STATUS ---
st.success("LIVE Reddit + AI kører 🚀")
st.info("Scoren opdateres automatisk ca. hver 3. minut (via cache).")
