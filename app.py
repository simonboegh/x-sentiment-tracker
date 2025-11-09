import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from transformers import pipeline
import snscrape.modules.twitter as sntwitter
import time

# --- KONFIG ---
st.set_page_config(page_title="X Sentiment", layout="wide")
st.title("LIVE X-Sentiment – GRATIS & STABIL!")

# --- ULTRA-LET MODEL (500 MB) ---
@st.cache_resource
def load_model():
    return pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        tokenizer="cardiffnlp/twitter-roberta-base-sentiment-latest",
        device=-1
    )

model = load_model()

# --- HENT TWEETS (max 10 – spar RAM) ---
def get_live_tweets(symbol):
    query = f"${symbol} lang:en -filter:replies"
    tweets = []
    try:
        for i, tweet in enumerate(sntwitter.TwitterSearchScraper(query).get_items(), 1):
            if i > 10: break
            tweets.append(tweet.rawContent[:500])  # Max 500 tegn
        return tweets
    except:
        return []

# --- SENTIMENT (kun 3 tweets – hurtigt!) ---
def get_sentiment(tweets):
    if not tweets:
        return 0.0
    scores = []
    for t in tweets[:3]:
        try:
            result = model(t)[0]
            label = result['label']
            score = result['score']
            if label == 'LABEL_2':  # positiv
                scores.append(score)
            elif label == 'LABEL_0':  # negativ
                scores.append(-score)
            # LABEL_1 = neutral → 0
        except:
            continue
    return sum(scores)/len(scores) if scores else 0.0

# --- PRIS ---
def get_price(symbol):
    try:
        return round(yf.Ticker(symbol).history(period="1d")['Close'].iloc[-1], 2)
    except:
        return "N/A"

# --- AKTIER ---
stocks = ["GME", "TSLA", "AMC"]
names = ["GameStop", "Tesla", "AMC"]

# --- DASHBOARD ---
for name, symbol in zip(names, stocks):
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader(f"{name}")
        st.metric("Pris", f"${get_price(symbol)}")
        
        tweets = get_live_tweets(symbol)
        score = get_sentiment(tweets)
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score * 100,
            title={'text': "Sentiment"},
            gauge={'axis': {'range': [-100, 100]},
                   'bar': {'color': "lime" if score > 0.1 else "red" if score < -0.1 else "gray"}}
        ))
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        with st.expander(f"Tweets om ${symbol}"):
            if tweets:
                for t in tweets:
                    st.caption(t[:200] + "...")
            else:
                st.caption("Henter tweets...")

# --- STATUS ---
st.success("Kører stabilt på gratis Streamlit!")
st.info("Opdaterer hvert 3. minut.")
time.sleep(180)
st.rerun()
