import streamlit as st
import plotly.graph_objects as go
from transformers.pipelines import pipeline
import praw

# ------------------- PARAMETRE -------------------

# Hvor mange WSB-tråde vi max kigger i pr. aktie
MAX_SUBMISSIONS = 25

# Hvor mange relevante kommentarer vi max analyserer pr. aktie
MAX_COMMENTS = 200

# ------------------- KONFIG & TITEL -------------------

st.set_page_config(page_title="Reddit AI", layout="wide")
st.title("AI Reddit Sentiment")
st.markdown("**FinBERT analyserer live kommentarer fra *r/WallStreetBets***")

# ------------------- AI-MODEL (FinBERT) -------------------

@st.cache_resource
def load_ai():
    with st.spinner("Henter AI-model... (kun første gang)"):
        return pipeline("text-classification", model="yiyanghkust/finbert-tone")

ai = load_ai()

# ------------------- REDDIT KLIENT -------------------

@st.cache_resource
def get_reddit_client():
    return praw.Reddit(
        client_id=st.secrets["reddit"]["client_id"],
        client_secret=st.secrets["reddit"]["client_secret"],
        user_agent=st.secrets["reddit"]["user_agent"],
    )

def score_to_text(score_100: int) -> str:
    """Omsætter -100..100 til kort tekst."""
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

# ------------------- HENT & ANALYSER KOMMENTARER -------------------

@st.cache_data(ttl=600)  # cache 10 minutter
def get_reddit_sentiment(symbol: str):
    reddit = get_reddit_client()
    subreddit = reddit.subreddit("wallstreetbets")

    comments = []
    posts_used = 0

    # Tekststumper vi vil filtrere væk (rapporter, auto-bots osv.)
    blocked_phrases = [
        "User Report",
        "Total Submissions",
        "First Seen In WSB",
        "Report generated",
        "moderator of this subreddit",
    ]

    try:
        # 1) Find op til MAX_SUBMISSIONS tråde der nævner symbolet
        query = f'"${symbol}" OR "{symbol}"'
        for submission in subreddit.search(query, sort="new", limit=MAX_SUBMISSIONS):
            posts_used += 1
            submission.comments.replace_more(limit=0)
            for c in submission.comments.list():
                text = getattr(c, "body", "")

                # Grundfiltrering
                if len(text) < 20 or len(text) > 800:
                    continue
                if any(bad in text for bad in blocked_phrases):
                    continue

                # Kræv at kommentaren faktisk nævner symbolet (case-insensitive)
                t_upper = text.upper()
                if symbol.upper() not in t_upper and f"${symbol.upper()}" not in t_upper:
                    continue

                comments.append(text)
                if len(comments) >= MAX_COMMENTS:
                    break
            if len(comments) >= MAX_COMMENTS:
                break

        raw_comments_count = len(comments)

        if raw_comments_count == 0:
            # score, fejltekst, bull_ex, bear_ex, counts..., posts_used, raw_comments_count
            return 0, "Ingen relevante kommentarer fundet lige nu", None, None, 0, 0, 0, 0, posts_used, raw_comments_count

        analyzed = []

        # 2) Kør FinBERT på ALLE relevante kommentarer
        for text in comments:
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
            return 0, "Kunne ikke analysere kommentarer lige nu", None, None, 0, 0, 0, 0, posts_used, raw_comments_count

        # 3) Tæl bullish / bearish / neutral
        n_bull = sum(1 for _, s, _ in analyzed if s == "Bullish")
        n_bear = sum(1 for _, s, _ in analyzed if s == "Bearish")
        n_neutral = sum(1 for _, s, _ in analyzed if s == "Neutral")
        n_total = n_bull + n_bear + n_neutral

        if n_bull + n_bear > 0:
            # Netto-bullish i procent: -100..100
            score_100 = round(100 * (n_bull - n_bear) / (n_bull + n_bear))
        else:
            score_100 = 0

        # 4) Find bedste bullish og bedste bearish eksempel (højeste sikkerhed)
        bull_candidates = [item for item in analyzed if item[1] == "Bullish"]
        bear_candidates = [item for item in analyzed if item[1] == "Bearish"]

        bull_example = max(bull_candidates, key=lambda x: x[2]) if bull_candidates else None
        bear_example = max(bear_candidates, key=lambda x: x[2]) if bear_candidates else None

        # score, ingen fejltekst, bull_ex, bear_ex, counts..., posts_used, raw_comments_count
        return (
            score_100,
            None,
            bull_example,
            bear_example,
            n_total,
            n_bull,
            n_bear,
            n_neutral,
            posts_used,
            raw_comments_count,
        )

    except Exception as e:
        return 0, f"Reddit fejl: {str(e)[:120]}", None, None, 0, 0, 0, 0, posts_used, len(comments)

# ------------------- AKTIER I DASHBOARD -------------------

stocks = ["TSLA", "PLTR", "SPY"]
names = ["Tesla", "Palantir", "S&P 500 (SPY)"]

# ------------------- DASHBOARD -------------------

for name, symbol in zip(names, stocks):
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader(f"{name} (${symbol})")

        with st.spinner("Henter Reddit..."):
            (
                score_100,
                error_msg,
                bull_ex,
                bear_ex,
                n_total,
                n_bull,
                n_bear,
                n_neutral,
                posts_used,
                raw_comments_count,
            ) = get_reddit_sentiment(symbol)

        sentiment_text = score_to_text(score_100)

        st.markdown(
            f"**WSB er {sentiment_text} på `{symbol}` lige nu.**  \n"
            f"Sentimentscoren er **{score_100}**, hvor "
            "**-100** betyder kun bearish kommentarer, **0** betyder lige mange bullish og bearish, "
            "og **+100** betyder kun bullish."
        )

        # Gør datagrundlaget tydeligt
        st.caption(
            f"Baseret på **{n_total} analyserede kommentarer** "
            f"(ud af {raw_comments_count} relevante) fra **{posts_used} WSB-tråde**, "
            f"der nævner `{symbol}`."
        )
        st.caption(
            f"Fordeling: 🐂 **{n_bull} bullish**, 🐻 **{n_bear} bearish**, 😶 **{n_neutral} neutrale**."
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
        with st.expander("Eksempler på kommentarer (AI-udvalgt)"):
            if error_msg:
                st.info(error_msg)
            else:
                if bull_ex:
                    text, _, conf = bull_ex
                    st.subheader("🐂 Bullish eksempel")
                    st.caption(f"Model-sikkerhed: {conf:.2f}")
                    st.write(text)
                else:
                    st.info("Ingen tydeligt bullish kommentar fundet lige nu.")

                st.markdown("---")

                if bear_ex:
                    text, _, conf = bear_ex
                    st.subheader("🐻 Bearish eksempel")
                    st.caption(f"Model-sikkerhed: {conf:.2f}")
                    st.write(text)
                else:
                    st.info("Ingen tydeligt bearish kommentar fundet lige nu.")

# ------------------- STATUS -------------------

st.success("LIVE Reddit + AI kører 🚀")
st.info(
    f"Scoren bygger på op til {MAX_COMMENTS} relevante kommentarer pr. aktie "
    f"fra maksimalt {MAX_SUBMISSIONS} nye WSB-tråde. Data caches i 10 minutter."
)
