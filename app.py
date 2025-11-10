import streamlit as st
import plotly.graph_objects as go
from transformers.pipelines import pipeline
import praw
from datetime import datetime, timezone

# ------------------- PARAMETRE -------------------

MAX_COMMENTS = 200        # max kommentarer vi analyserer pr. aktie
MAX_POSTS_SCAN = 400      # hvor mange af de nyeste WSB-opslag vi tjekker titlen på

# Udvidede keywords pr. aktie (uppercased) – bruges på TITLER
COMPANY_KEYWORDS = {
    "TSLA": [
        "TSLA", "$TSLA",
        "TESLA",
        "ELON", "MUSK", "ELON MUSK",
        "TSLAQ",
    ],
    "PLTR": [
        "PLTR", "$PLTR",
        "PALANTIR",
        "KARP", "ALEX KARP",
    ],
    "SPY": [
        "SPY", "$SPY",
        "SP500", "SP 500",
        "S&P500", "S&P 500",
        "SPX",
    ],
}

# ------------------- KONFIG & TITEL -------------------

st.set_page_config(page_title="Reddit AI", layout="wide")
st.title("AI Reddit Sentiment")
st.markdown("**FinBERT analyserer live kommentarer fra *r/WallStreetBets***")

# Manuelt refresh af cache
if st.button("🔄 Opdater data nu"):
    st.cache_data.clear()
    st.experimental_rerun()

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

@st.cache_data(ttl=300)  # cache 5 minutter
def get_reddit_sentiment(symbol: str):
    reddit = get_reddit_client()
    subreddit = reddit.subreddit("wallstreetbets")

    sym_up = symbol.upper()
    keywords = COMPANY_KEYWORDS.get(sym_up, [sym_up, f"${sym_up}"])

    comments = []
    posts_used_ids = set()
    fetch_time = datetime.now(timezone.utc)

    blocked_phrases = [
        "User Report",
        "Total Submissions",
        "First Seen In WSB",
        "Report generated",
        "moderator of this subreddit",
    ]

    try:
        # 1) Gå igennem de nyeste WSB-opslag (ikke søgeindeks)
        for submission in subreddit.new(limit=MAX_POSTS_SCAN):
            title_up = submission.title.upper()
            # Kun tråde hvor titlen matcher et keyword
            if not any(kw in title_up for kw in keywords):
                continue

            posts_used_ids.add(submission.id)

            # Hent alle kommentarer i den tråd
            submission.comments.replace_more(limit=0)
            for c in submission.comments.list():
                try:
                    text = c.body
                except Exception:
                    continue

                # Filtrér tydeligt junk
                if len(text) < 10 or len(text) > 800:
                    continue
                if any(bad in text for bad in blocked_phrases):
                    continue

                # Her kræver vi IKKE keywords i selve kommentaren –
                # tråden er allerede om aktien, så vi tager samtalen med.
                comments.append(text)

                if len(comments) >= MAX_COMMENTS:
                    break
            if len(comments) >= MAX_COMMENTS:
                break

        raw_comments_count = len(comments)
        posts_used = len(posts_used_ids)

        if raw_comments_count == 0:
            return (
                0, "Ingen kommentarer fundet i nylige WSB-opslag om denne aktie",
                None, None, 0, 0, 0, 0, posts_used, raw_comments_count, fetch_time
            )

        analyzed = []

        # 2) Kør FinBERT på ALLE kommentarer i de valgte tråde
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
            return (
                0, "Kunne ikke analysere kommentarer lige nu",
                None, None, 0, 0, 0, 0, posts_used, raw_comments_count, fetch_time
            )

        # 3) Tæl bullish / bearish / neutral
        n_bull = sum(1 for _, s, _ in analyzed if s == "Bullish")
        n_bear = sum(1 for _, s, _ in analyzed if s == "Bearish")
        n_neutral = sum(1 for _, s, _ in analyzed if s == "Neutral")
        n_total = n_bull + n_bear + n_neutral

        if n_bull + n_bear > 0:
            score_100 = round(100 * (n_bull - n_bear) / (n_bull + n_bear))
        else:
            score_100 = 0

        # 4) Find bedste bullish og bedste bearish eksempel
        bull_candidates = [item for item in analyzed if item[1] == "Bullish"]
        bear_candidates = [item for item in analyzed if item[1] == "Bearish"]

        bull_example = max(bull_candidates, key=lambda x: x[2]) if bull_candidates else None
        bear_example = max(bear_candidates, key=lambda x: x[2]) if bear_candidates else None

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
            fetch_time,
        )

    except Exception as e:
        return (
            0, f"Reddit fejl: {str(e)[:120]}",
            None, None, 0, 0, 0, 0, len(posts_used_ids), len(comments), fetch_time
        )

# ------------------- AKTIER I DASHBOARD -------------------

stocks = ["TSLA", "PLTR", "SPY"]
names = ["Tesla", "Palantir", "S&P 500 (SPY)"]

# Hent data til alle aktier med progress bar
results = {}
progress = st.progress(0, text="Indlæser data fra Reddit...")

for i, symbol in enumerate(stocks):
    results[symbol] = get_reddit_sentiment(symbol)
    progress.progress((i + 1) / len(stocks), text=f"Indlæser {symbol} ({i+1}/{len(stocks)})")

progress.empty()

# ------------------- RAD 1: 3 GAUGES PÅ STRIBE -------------------

st.subheader("WallStreetBets-sentiment (nyeste tråde om aktien)")

cols = st.columns(3)

for col, (name, symbol) in zip(cols, zip(names, stocks)):
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
        fetch_time,
    ) = results[symbol]

    with col:
        st.markdown(f"### {name} (`{symbol}`)")

        if error_msg:
            st.info(error_msg)
        else:
            sentiment_text = score_to_text(score_100)
            st.markdown(
                f"**WSB er {sentiment_text} på `{symbol}` lige nu.**  \n"
                f"Score: **{score_100}** (−100 bearish, 0 neutral, +100 bullish)."
            )

            fig = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=score_100,
                    title={"text": "Netto-bullish sentiment"},
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

            last_updated = fetch_time.strftime("%Y-%m-%d %H:%M UTC")
            st.caption(
                f"Sidst opdateret: **{last_updated}** · "
                f"{n_total} analyserede kommentarer (ud af {raw_comments_count}) "
                f"fra **{posts_used} nylige WSB-opslag**, hvor titlen nævner aktien "
                f"(tjekket blandt de {MAX_POSTS_SCAN} nyeste opslag)."
            )
            st.caption(
                f"Fordeling: 🐂 {n_bull} bullish · 🐻 {n_bear} bearish · 😶 {n_neutral} neutrale."
            )

# ------------------- RAD 2: EKSEMPLER PÅ KOMMENTARER -------------------

st.subheader("Eksempler på kommentarer (AI-udvalgt)")

for name, symbol in zip(names, stocks):
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
        fetch_time,
    ) = results[symbol]

    with st.expander(f"{name} (`{symbol}`)"):
        if error_msg:
            st.info(error_msg)
            continue

        if bull_ex:
            text, _, conf = bull_ex
            st.markdown("#### 🐂 Bullish eksempel")
            st.caption(f"Model-sikkerhed: {conf:.2f}")
            st.write(text)
        else:
            st.info("Ingen tydeligt bullish kommentar fundet lige nu.")
        st.markdown("---")
        if bear_ex:
            text, _, conf = bear_ex
            st.markdown("#### 🐻 Bearish eksempel")
            st.caption(f"Model-sikkerhed: {conf:.2f}")
            st.write(text)
        else:
            st.info("Ingen tydeligt bearish kommentar fundet lige nu.")

# ------------------- STATUS -------------------

st.success("LIVE Reddit + AI kører 🚀")
st.info(
    f"Scoren bygger på kommentarer i nylige WSB-opslag, hvor titlen matcher dine nøgleord "
    f"(op til {MAX_COMMENTS} kommentarer pr. aktie, fundet blandt de {MAX_POSTS_SCAN} nyeste opslag).\n"
    "Data caches i 5 minutter – brug knappen **'Opdater data nu'** for at tvinge en frisk hentning."
)
