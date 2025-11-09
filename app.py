import time
import streamlit as st
import plotly.graph_objects as go
from transformers.pipelines import pipeline
from pmaw import PushshiftAPI

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

# --- HENT REDDIT KOMMENTARER ---
@st.cache_data(ttl=180)  # Opdater hvert 3. minut
def get_reddit_sentiment(symbol: str):
    api = PushshiftAPI()
    try:
        comments = api.search_comments(
            q=f"${symbol}",
            subreddit="wallstreetbets",
            limit=30,
            filter=["body"],
        )
        comments = [c["body"] for c in comments if len(c["body"]) > 15]

        if not comments:
            # Returnér en tekstbesked når der ikke er noget at analysere
            return 0.0, "Ingen kommentarer fundet lige nu"

        scores = []
        analyzed = []

        for text in comments[:7]:  # 7 kommentarer
            try:
                result = ai(text)[0]
                label = result["label"]
                score = result["score"]

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
        # Returnér en tekstbesked hvis Reddit / Pushshift fejler
        return 0.0, f"Reddit fejl: {str(e)[:80]}"

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

        # Gauge
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=sentiment * 100,
                title={"text": "AI Sentiment"},
                gauge={
                    "axis": {"range": [-100, 100]},
                    "bar": {
                        "color": "lime"
                        if sentiment > 0.1
                        else "red"
                        if sentiment < -0.1
                        else "gray"
                    },
                },
            )
        )
        st.plotly_chart(fig, use_container_width=True, key=f"gauge_{symbol}")

    with col2:
        with st.expander("AI-analyse af Reddit-kommentarer"):
            # Hvis analysis er en tekststreng → vis info-boks
            if isinstance(analysis, str):
                st.info(analysis)
            else:
                # Ellers forventer vi en liste af (text, label, score)
                for text, label, score in analysis:
                    emoji = (
                        "Bullish"
                        if label == "Positive"
                        else "Bearish"
                        if label == "Negative"
                        else "Neutral"
                    )
                    st.write(f"{emoji} **{label}** ({score:.2f})")
                    st.caption(text[:120] + "..." if len(text) > 120 else text)

# --- STATUS ---
st.success("**LIVE Reddit + AI kører 🚀**")
st.info("Data gemmes i cache i 3 minutter (ttl=180).")

# Jeg vil *stærkt* anbefale at fjerne den her blok i første omgang:
# time.sleep(180)
# st.rerun()
# På Streamlit Cloud kan den slags sleep + rerun give timeouts.
