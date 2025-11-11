import streamlit as st
import plotly.graph_objects as go
from transformers.pipelines import pipeline
import praw
from datetime import datetime, timezone
import requests

# ------------------- PARAMETRE -------------------

MAX_COMMENTS = 200        # max kommentarer vi analyserer pr. aktie (Reddit)
MAX_POSTS_SCAN = 400      # hvor mange af de nyeste WSB-opslag vi tjekker titlen på
NEWS_API_URL = "https://newsapi.org/v2/everything"

# Udvidede keywords pr. aktie (uppercased) – bruges både til Reddit TITLER og til nyhedssøgning
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

OM_METODEN_TEKST = """
**Kort fortalt**

- Appen måler **stemningen omkring udvalgte aktier** to steder:
  1. I **Reddit-kommentarer** fra *r/WallStreetBets*.
  2. I **klassiske finansnyheder** (via et nyheds-API).
- Begge steder bruger jeg den samme sproglige AI-model, **FinBERT**, som er trænet på
  finansielt sprog (earnings, nyheder, analytikernoter osv.).
- For hver tekst (kommentar eller artikel) vurderer modellen, om tonen er  
  **positiv (bullish), negativ (bearish) eller neutral**.
- Derefter laves en samlet **sentimentscore fra -100 til +100** for hver aktie:

> **Score = -100** → kun bearish  
> **Score = 0** → lige mange bullish og bearish  
> **Score = +100** → kun bullish

Neutrale tekster tæller med i fordelingen, men påvirker ikke selve scoren.

---

**Kilde 1: Reddit / r/WallStreetBets**

- For hver aktie (fx TSLA) kigger jeg på de **nyeste opslag i r/WallStreetBets**, hvor
  titlen matcher et sæt nøgleord, fx:

  - `TSLA`, `TESLA`, `ELON`, `ELON MUSK` osv. for Tesla
  - `PLTR`, `PALANTIR`, `KARP` osv. for Palantir
  - `SPY`, `SP500`, `S&P500`, `SPX` osv. for S&P 500 (SPY)

- For hvert sådant opslag hentes kommentarerne, renses let (fjern spam, meget korte eller
  ekstremt lange tekster), og FinBERT vurderer dem som bullish/bearish/neutral.
- På den måde får du et billede af **retail/WSB-stemningen** i de nyeste tråde om aktien.

---

**Kilde 2: Klassiske finansnyheder**

- For den samme aktie hentes nyheder via et nyheds-API (NewsAPI), hvor der søges på
  de **samme nøgleord** (fx `TSLA OR TESLA OR ELON MUSK`).
- For hver artikel analyseres **titel + kort beskrivelse** med FinBERT, som igen
  klassificerer tonen som bullish/bearish/neutral.
- Dermed får du en separat sentimentscore for **klassiske medier / finansnyheder**.

---

**Vigtige begrænsninger**

- FinBERT er trænet på **seriøst finanssprog**, ikke på memes, slang og ironi fra
  *r/WallStreetBets*.  
  Derfor kan modellen nogle gange misforstå sarkasme, interne jokes eller emojis.
- Nyheds-API’et leverer artikler fra mange forskellige kilder (ikke alle er lige dybt
  finansielle), så nogle artikler er mere “markedstunge” end andre.
- Resultaterne skal derfor ses som et **groft stemningsbillede**, ikke som en præcis
  sandhed om markedet eller som investeringsrådgivning.

Kort sagt: Appen forsøger at oversætte både Reddit-snak og nyhedsflow til simple
tal, der viser om stemningen mest hælder til bullish eller bearish på tværs af
sociale medier og traditionelle medier.
"""

# ------------------- KONFIG & TITEL -------------------

st.set_page_config(page_title="Reddit + News AI Sentiment", layout="wide")
st.title("AI Sentiment: WallStreetBets vs. Finansnyheder")
st.markdown("**FinBERT analyserer både *r/WallStreetBets*-kommentarer og klassiske finansnyheder.**")

# Manuelt refresh af cache
if st.button("🔄 Opdater data nu"):
    st.cache_data.clear()
    st.rerun()  # ny Streamlit-metode

with st.expander("Hvordan virker AI-sentimentet?"):
    st.markdown(OM_METODEN_TEKST)

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

# ------------------- HENT & ANALYSER KOMMENTARER (REDDIT) -------------------

@st.cache_data(ttl=300)  # cache 5 minutter
def get_reddit_sentiment(symbol: str):
    reddit = get_reddit_client()
    subreddit = reddit.subreddit("wallstreetbets")

    sym_up = symbol.upper()
    keywords = COMPANY_KEYWORDS.get(sym_up, [sym_up, f"${sym_up}"])

    comments = []          # liste af (text, title)
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

                # Vi kræver ikke keywords i kommentaren – tråden handler om aktien
                comments.append((text, submission.title))

                if len(comments) >= MAX_COMMENTS:
                    break
            if len(comments) >= MAX_COMMENTS:
                break

        raw_comments_count = len(comments)
        posts_used = len(posts_used_ids)

        if raw_comments_count == 0:
            return (
                0,
                "Ingen kommentarer fundet i nylige WSB-opslag om denne aktie",
                None,
                None,
                0,
                0,
                0,
                0,
                posts_used,
                raw_comments_count,
                fetch_time,
            )

        analyzed = []  # (text, title, sentiment_word, conf)

        # 2) Kør FinBERT på ALLE kommentarer i de valgte tråde
        for text, title in comments:
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

                analyzed.append((text, title, sentiment_word, conf))
            except Exception:
                continue

        if not analyzed:
            return (
                0,
                "Kunne ikke analysere kommentarer lige nu",
                None,
                None,
                0,
                0,
                0,
                0,
                posts_used,
                raw_comments_count,
                fetch_time,
            )

        # 3) Tæl bullish / bearish / neutral
        n_bull = sum(1 for _, _, s, _ in analyzed if s == "Bullish")
        n_bear = sum(1 for _, _, s, _ in analyzed if s == "Bearish")
        n_neutral = sum(1 for _, _, s, _ in analyzed if s == "Neutral")
        n_total = n_bull + n_bear + n_neutral

        if n_bull + n_bear > 0:
            score_100 = round(100 * (n_bull - n_bear) / (n_bull + n_bear))
        else:
            score_100 = 0

        # 4) Find bedste bullish og bedste bearish eksempel
        bull_candidates = [item for item in analyzed if item[2] == "Bullish"]
        bear_candidates = [item for item in analyzed if item[2] == "Bearish"]

        bull_example = max(bull_candidates, key=lambda x: x[3]) if bull_candidates else None
        bear_example = max(bear_candidates, key=lambda x: x[3]) if bear_candidates else None

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
            0,
            f"Reddit fejl: {str(e)[:120]}",
            None,
            None,
            0,
            0,
            0,
            0,
            len(posts_used_ids),
            len(comments),
            fetch_time,
        )

# ------------------- HENT & ANALYSER NYHEDER -------------------

@st.cache_data(ttl=600)  # cache 10 minutter
def get_news_sentiment(symbol: str):
    """Bruger FinBERT til at måle sentiment i finansnyheder om en given aktie."""
    fetch_time = datetime.now(timezone.utc)
    sym_up = symbol.upper()
    keywords = COMPANY_KEYWORDS.get(sym_up, [sym_up])

    # Brug de samme keywords som til Reddit, men uden '$', til nyhedssøgningen
    cleaned_keywords = []
    for kw in keywords:
        kw_clean = kw.replace("$", "").strip()
        if kw_clean:
            cleaned_keywords.append(kw_clean)

    # Byg query som "TSLA OR TESLA OR ELON MUSK"
    if cleaned_keywords:
        q = " OR ".join(sorted(set(cleaned_keywords)))
    else:
        q = symbol

    try:
        # 1) Hent nyheder fra API
        params = {
            "q": q,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": 30,
            "apiKey": st.secrets["news"]["api_key"],
        }
        r = requests.get(NEWS_API_URL, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        articles = data.get("articles", [])

        if not articles:
            return (
                0,
                f"Ingen nyheder fundet for {symbol} lige nu",
                None,
                None,
                0,
                0,
                0,
                0,
                0,
                fetch_time,
            )

        analyzed = []  # (headline, url, sentiment_word, conf)

        # 2) Kør FinBERT på title + description
        for art in articles:
            title = art.get("title") or ""
            desc = art.get("description") or ""
            url = art.get("url") or ""

            text = f"{title}. {desc}".strip()
            if len(text) < 20:
                continue

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

                analyzed.append((title, url, sentiment_word, conf))
            except Exception:
                continue

        if not analyzed:
            return (
                0,
                "Kunne ikke analysere nyhedsartikler lige nu",
                None,
                None,
                0,
                0,
                0,
                0,
                len(articles),
                fetch_time,
            )

        # 3) Tæl bullish / bearish / neutral
        n_bull = sum(1 for _, _, s, _ in analyzed if s == "Bullish")
        n_bear = sum(1 for _, _, s, _ in analyzed if s == "Bearish")
        n_neutral = sum(1 for _, _, s, _ in analyzed if s == "Neutral")
        n_total = n_bull + n_bear + n_neutral

        if n_bull + n_bear > 0:
            score_100 = round(100 * (n_bull - n_bear) / (n_bull + n_bear))
        else:
            score_100 = 0

        # 4) Find bedste bullish og bearish artikel
        bull_candidates = [item for item in analyzed if item[2] == "Bullish"]
        bear_candidates = [item for item in analyzed if item[2] == "Bearish"]

        bull_example = max(bull_candidates, key=lambda x: x[3]) if bull_candidates else None
        bear_example = max(bear_candidates, key=lambda x: x[3]) if bear_candidates else None

        return (
            score_100,
            None,
            bull_example,
            bear_example,
            n_total,
            n_bull,
            n_bear,
            n_neutral,
            len(articles),
            fetch_time,
        )

    except Exception as e:
        return (
            0,
            f"Nyheds-API fejl: {str(e)[:120]}",
            None,
            None,
            0,
            0,
            0,
            0,
            0,
            fetch_time,
        )

# ------------------- AKTIER I DASHBOARD -------------------

stocks = ["TSLA", "PLTR", "SPY"]
names = ["Tesla", "Palantir", "S&P 500 (SPY)"]

# Hent Reddit-data til alle aktier med progress bar
results_reddit = {}
progress = st.progress(0, text="Indlæser Reddit-data...")

for i, symbol in enumerate(stocks):
    results_reddit[symbol] = get_reddit_sentiment(symbol)
    progress.progress((i + 1) / len(stocks), text=f"Indlæser Reddit for {symbol} ({i+1}/{len(stocks)})")

progress.empty()

# Hent nyhedsdata til alle aktier
results_news = {}
for symbol in stocks:
    results_news[symbol] = get_news_sentiment(symbol)

# ------------------- RAD 1: REDDIT-SENTIMENT -------------------

st.subheader("📊 WallStreetBets-sentiment (nyeste tråde om aktien)")

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
    ) = results_reddit[symbol]

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
                    title={"text": "WSB-sentiment"},
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
            st.plotly_chart(fig, width="stretch", key=f"gauge_reddit_{symbol}")

            last_updated = fetch_time.strftime("%Y-%m-%d %H:%M UTC")
            st.caption(
                f"Sidst opdateret: **{last_updated}** · "
                f"{n_total} analyserede kommentarer (ud af {raw_comments_count}) "
                f"fra **{posts_used} nylige WSB-opslag**."
            )
            st.caption(
                f"Fordeling: 🐂 {n_bull} bullish · 🐻 {n_bear} bearish · 😶 {n_neutral} neutrale."
            )

# ------------------- RAD 2: NYHEDS-SENTIMENT -------------------

st.subheader("📰 Finansnyheder-sentiment (klassiske medier)")

cols_news = st.columns(3)

for col, (name, symbol) in zip(cols_news, zip(names, stocks)):
    (
        news_score,
        news_error,
        news_bull_ex,
        news_bear_ex,
        news_n_total,
        news_n_bull,
        news_n_bear,
        news_n_neutral,
        news_n_articles,
        news_fetch_time,
    ) = results_news[symbol]

    with col:
        st.markdown(f"### {name} (`{symbol}`)")

        if news_error:
            st.info(news_error)
            continue

        sentiment_text = score_to_text(news_score)
        st.markdown(
            f"**Nyhedsflowet er {sentiment_text} på `{symbol}` lige nu.**  \n"
            f"Score: **{news_score}** (−100 bearish, 0 neutral, +100 bullish)."
        )

        fig_news = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=news_score,
                title={"text": "Nyheds-sentiment"},
                gauge={
                    "axis": {"range": [-100, 100]},
                    "bar": {
                        "color": "lime"
                        if news_score > 10
                        else "red"
                        if news_score < -10
                        else "gray"
                    },
                },
            )
        )
        st.plotly_chart(fig_news, width="stretch", key=f"gauge_news_{symbol}")

        last_updated_news = news_fetch_time.strftime("%Y-%m-%d %H:%M UTC")
        st.caption(
            f"Sidst opdateret: **{last_updated_news}** · "
            f"{news_n_total} analyserede artikler (ud af {news_n_articles} hentet). "
            f"Fordeling: 🐂 {news_n_bull} bullish · 🐻 {news_n_bear} bearish · 😶 {news_n_neutral} neutrale."
        )

# ------------------- RAD 3: EKSEMPLER FRA REDDIT -------------------

st.subheader("💬 Eksempler på WSB-kommentarer (AI-udvalgt)")

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
    ) = results_reddit[symbol]

    with st.expander(f"{name} (`{symbol}`) – Reddit-kommentarer"):
        if error_msg:
            st.info(error_msg)
            continue

        if bull_ex:
            text, title, _, conf = bull_ex
            st.markdown("#### 🐂 Bullish kommentar")
            st.caption(f"Fra opslaget: *{title}*")
            st.caption(f"Model-sikkerhed: {conf:.2f}")
            st.write(text)
        else:
            st.info("Ingen tydeligt bullish kommentar fundet lige nu.")
        st.markdown("---")
        if bear_ex:
            text, title, _, conf = bear_ex
            st.markdown("#### 🐻 Bearish kommentar")
            st.caption(f"Fra opslaget: *{title}*")
            st.caption(f"Model-sikkerhed: {conf:.2f}")
            st.write(text)
        else:
            st.info("Ingen tydeligt bearish kommentar fundet lige nu.")

# ------------------- RAD 4: EKSEMPLER FRA NYHEDER -------------------

st.subheader("📑 Eksempler på nyhedsartikler (AI-udvalgt)")

for name, symbol in zip(names, stocks):
    (
        news_score,
        news_error,
        news_bull_ex,
        news_bear_ex,
        news_n_total,
        news_n_bull,
        news_n_bear,
        news_n_neutral,
        news_n_articles,
        news_fetch_time,
    ) = results_news[symbol]

    with st.expander(f"{name} (`{symbol}`) – nyheder"):
        if news_error:
            st.info(news_error)
            continue

        if news_bull_ex:
            title, url, _, conf = news_bull_ex
            st.markdown("#### 🐂 Bullish nyhed")
            st.caption(f"Model-sikkerhed: {conf:.2f}")
            st.write(title)
            if url:
                st.markdown(f"[Læs artikel]({url})")
        else:
            st.info("Ingen tydeligt bullish artikel fundet lige nu.")
        st.markdown("---")
        if news_bear_ex:
            title, url, _, conf = news_bear_ex
            st.markdown("#### 🐻 Bearish nyhed")
            st.caption(f"Model-sikkerhed: {conf:.2f}")
            st.write(title)
            if url:
                st.markdown(f"[Læs artikel]({url})")
        else:
            st.info("Ingen tydeligt bearish artikel fundet lige nu.")
