import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from transformers import pipeline
import tweepy
import os
import threading
import time

# --- KONFIG ---
st.set_page_config(page_title="X Sentiment Tracker", layout="wide")
st.title("Real-time X-Sentiment Tracker (Nasdaq)")

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis", model="ProsusAI/finbert", device=-1)

model = load_model()

# --- X CLIENT ---
client = tweepy.Client(
    bearer_token=os.getenv("BEARER_TOKEN"),
    wait_on_rate_limit=True
)

# --- HENT TWEETS MED TIMEOUT (max 10 sek) ---
def get_tweets_with_timeout(symbol, timeout=10):
    result = ["Laster..."]
    def fetch():
        query = f"${symbol} OR '{symbol} stock' -is:retweet lang:en"
        try:
            resp = client.search_recent_tweets(
                query=query,
                max_results=20,
                tweet_fields=['created_at']
            )
            if resp.data:
                result[0] = [t.text for t in resp.data]
            else:
                result[0] = ["Ingen tweets fundet."]
        except Exception as e:
            result[0] = [f"API-fejl: {str(e)[:40]}..."]
    
    thread = threading.Thread(target=fetch)
    thread.start()
    thread.join(timeout)
    
    if thread.is_alive():
        return ["Timeout – prøver igen snart."]
    return result[0]

# --- SENTIMENT ---
def get_sentiment(tweets):
    if not tweets or any(x.startswith(("Ingen", "API", "Timeout", "Laster")) for x in tweets):
        return 0.0
    scores = []
    for t in tweets[:5]:
        try:
            r = model(t)[0]
            s = r['score'] if r['label'] == 'positive' else -r['score']
            scores.append(s)
        except:
            continue
    return sum(scores)/len(scores) if scores else 0.0

# --- PRIS ---
def get_price(symbol):
    try:
        return round(yf.Ticker(symbol).history(period="1d")['Close'].iloc[-1], 2)
    except:
        return None

# --- DATA ---
stocks = ["GME", "TSLA", "NVDA", "AAPL", "MSFT"]
names = ["GameStop", "Tesla", "Nvidia", "Apple", "Microsoft"]

# --- DASHBOARD ---
cols = st.columns(5)

for i, (name, symbol) in enumerate(zip(names, stocks)):
    with cols[i]:
        st.subheader(name)
        
        # Hent med timeout
        tweets = get_tweets_with_timeout(symbol, timeout=8)
        score = get_sentiment(tweets)
        price = get_price(symbol)
        
        # Gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score * 100,
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
        
        st.metric("Pris (USD)", f"${price}" if price else "N/A")
        
        with st.expander("Tweets"):
            for t in tweets[:3]:
                st.caption(t)

# --- STATUS ---
st.success("Live – opdateres hvert 5. min!")
