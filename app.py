import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from transformers import pipeline
import snscrape.modules.twitter as sntwitter
import time

# --- KONFIG ---
st.set_page_config(page_title="LIVE X Sentiment", layout="wide")
st.title("LIVE X-Sentiment Tracker – GRATIS & LAV RAM!")

# --- LILLE, HURTIG MODEL (300 MB) ---
@st.cache_resource
def load_model():
    return pipeline(
        "sentiment-analysis",
        model="mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis",
        device=-1  # CPU
    )

model = load_model()

# --- HENT TWEETS (GRATIS) ---
def get_live_tweets(symbol, max_results=15):
    query = f"${symbol} lang:en -filter:replies"
    tweets = []
    try:
        for i, tweet in enumerate(sntwitter.TwitterSearchScraper(query).get_items()):
            if i >= max_results:
                break
            tweets.append(tweet.rawContent)
        if tweets:
            st.success(f"**{symbol}: {len(tweets)} tweets**")
        return tweets
    except:
        st.warning(f"**{symbol}: Ingen tweets**")
        return []

# --- SENTIMENT (hurtig) ---
def get_sentiment(tweets):
    if not tweets:
        return 0.0
    scores = []
    for t in tweets[:5]:
        try:
            result = model(t[:512])[0]  # Max 512 tegn
            label = result['label']
            score = result['score']
            if label in ['positive', 'POS']:
                scores.append(score)
            elif label in ['negative', 'NEG']:
                scores.append(-score)
        except:
            continue
    return sum(scores)/len(scores) if scores else 0.0

# --- PRIS ---
def get_price(symbol):
    try:
        return round(yf.Ticker(symbol).history(period="1d")['Close'].iloc[-1], 2)
    except:
        return None

# --- AKTIER ---
stocks = ["GME", "TSLA", "AMC"]
names = ["GameStop", "Tesla", "AMC"]

# --- DASHBOARD ---
cols = st.columns(3)

for i, (name, symbol) in enumerate(zip(names, stocks)):
    with cols[i]:
        st.subheader(f"{name} (${symbol})")
        
        tweets = get_live_tweets(symbol)
        score = get_sentiment(tweets)
        price = get_price(symbol)
        
        # Gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score * 100,
            title={'text': "Sentiment"},
            gauge={
                'axis': {'range': [-100, 100]},
                'bar': {'color': "lime" if score > 0.1 else "red" if score < -0.1 else "gray"}
            }
        ))
        st.plotly_chart(fig, use_container_width=True)
        
        st.metric("Pris", f"${price}" if price else "N/A")
        
        with st.expander("Tweets"):
            for t in tweets[:3]:
                st.caption(t[:100] + "...")

# --- STATUS ---
st.success("Kører på gratis Streamlit! Opdaterer hvert 3. min.")
time.sleep(180)
st.rerun()
