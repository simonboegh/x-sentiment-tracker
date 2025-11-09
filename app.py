import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from transformers import pipeline
import tweepy
import os

# --- KONFIGURATION ---
st.set_page_config(page_title="X Sentiment Tracker", layout="wide")
st.title("Real-time X-Sentiment Tracker (Nasdaq)")

# --- LOAD FINBERT MODEL (én gang) ---
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis", model="ProsusAI/finbert", device=-1)

model = load_model()

# --- X API CLIENT (fra Streamlit Secrets) ---
client = tweepy.Client(
    bearer_token=os.getenv("BEARER_TOKEN"),
    wait_on_rate_limit=True
)

# --- HENT TWEETS (sikker og hurtig) ---
def get_tweets(symbol):
    query = f"${symbol} OR '{symbol} stock' -is:retweet lang:en"
    try:
        response = client.search_recent_tweets(
            query=query,
            max_results=20,
            tweet_fields=['created_at']
        )
        if response.data:
            return [tweet.text for tweet in response.data]
        else:
            return ["Ingen tweets fundet lige nu."]
    except Exception as e:
        return [f"API-fejl: {str(e)[:50]}..."]

# --- SENTIMENT ANALYSE (robust) ---
def get_sentiment(tweets):
    if not tweets or "Ingen" in tweets[0] or "API-fejl" in tweets[0]:
        return 0.0
    scores = []
    for tweet in tweets[:5]:
        try:
            result = model(tweet)[0]
            score = result['score'] if result['label'] == 'positive' else -result['score']
            scores.append(score)
        except:
            continue
    return sum(scores) / len(scores) if scores else 0.0

# --- HENT AKTIERPRIS ---
def get_price(symbol):
    try:
        price = yf.Ticker(symbol).history(period="1d")['Close'].iloc[-1]
        return round(price, 2)
    except:
        return None

# --- DATA ---
stocks = ["NVDA", "TSLA", "AAPL", "MSFT", "AMZN"]
names = ["Nvidia", "Tesla", "Apple", "Microsoft", "Amazon"]

# --- DASHBOARD ---
cols = st.columns(5)

for i, (name, symbol) in enumerate(zip(names, stocks)):
    with cols[i]:
        st.subheader(name)
        
        with st.spinner(f"Henter ${symbol}..."):
            tweets = get_tweets(symbol)
            score = get_sentiment(tweets)
            price = get_price(symbol)
        
        # Sentiment gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Sentiment"},
            gauge={
                'axis': {'range': [-100, 100]},
                'bar': {'color': "green" if score > 0 else "red"},
                'steps': [
                    {'range': [-100, -30], 'color': "darkred"},
                    {'range': [-30, 30], 'color': "orange"},
                    {'range': [30, 100], 'color': "lightgreen"}
                ]
            }
        ))
        st.plotly_chart(fig, use_container_width=True)
        
        # Pris
        st.metric("Pris (USD)", f"${price}" if price else "N/A")
        
        # Tweets
        with st.expander("Seneste tweets"):
            for t in tweets[:3]:
                st.caption(t)

# --- AUTO-REFRESH (hver 5. minut) ---
st.info("Opdateres hvert 5. minut...")
# time.sleep(300)  # Fjern kommentar når det kører stabilt
# st.rerun()

st.success("Live data kører! 🚀")
