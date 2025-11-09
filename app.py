import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from transformers import pipeline
import tweepy
import time

st.set_page_config(page_title="X Sentiment Tracker", layout="wide")
st.title("Real-time X-Sentiment Tracker (Nasdaq)")

@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis", model="ProsusAI/finbert")
model = load_model()

client = tweepy.Client(bearer_token=st.secrets["BEARER_TOKEN"], wait_on_rate_limit=True)

def get_tweets(symbol):
    query = f"${symbol} OR '{symbol} stock' -is:retweet lang:en"
    try:
        resp = client.search_recent_tweets(query=query, max_results=20)
        return [t.text for t in resp.data] if resp.data else []
    except Exception as e:
        return [f"Fejl: {str(e)}"]

def get_sentiment(tweets):
    if not tweets: return 0
    scores = []
    for t in tweets[:10]:
        try:
            r = model(t)[0]
            s = r['score'] if r['label'] == 'positive' else -r['score']
            scores.append(s)
        except: pass
    return sum(scores)/len(scores) if scores else 0

def get_price(symbol):
    try:
        return round(yf.Ticker(symbol).history(period="1d")['Close'].iloc[-1], 2)
    except: return None

stocks = ["NVDA", "TSLA", "AAPL", "MSFT", "AMZN"]
names = ["Nvidia", "Tesla", "Apple", "Microsoft", "Amazon"]

cols = st.columns(5)
for i, (name, sym) in enumerate(zip(names, stocks)):
    with cols[i]:
        st.subheader(name)
        tweets = get_tweets(sym)
        score = get_sentiment(tweets)
        price = get_price(sym)
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score*100,
            title={'text': "Sentiment"},
            gauge={'axis': {'range': [-100,100]},
                   'bar': {'color': "green" if score>0 else "red"}}
        ))
        st.plotly_chart(fig, use_container_width=True)
        st.metric("Pris", f"${price}" if price else "N/A")
        with st.expander("Seneste tweets"):
            for t in tweets[:3]: st.caption(t)

# Auto-refresh
time.sleep(300)
st.rerun()
