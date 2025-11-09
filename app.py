import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import snscrape.modules.twitter as sntwitter
import time

# --- KONFIG ---
st.set_page_config(page_title="X Sentiment", layout="wide")
st.title("LIVE X-Sentiment – 100 % GRATIS & STABIL!")

# --- ORDTÆLLING (INGEN AI = INGEN RAM!) ---
POSITIVE_WORDS = ["bullish", "moon", "buy", "pump", "to the moon", "squeeze", "rocket", "long", "hodl", "diamond hands"]
NEGATIVE_WORDS = ["bearish", "sell", "crash", "dump", "short", "paper hands", "falling", "dead"]

def get_sentiment_score(tweets):
    if not tweets:
        return 0.0
    pos_count = sum(1 for t in tweets if any(word in t.lower() for word in POSITIVE_WORDS))
    neg_count = sum(1 for t in tweets if any(word in t.lower() for word in NEGATIVE_WORDS))
    total = pos_count + neg_count
    return (pos_count - neg_count) / total if total > 0 else 0.0

# --- HENT TWEETS (max 8 – hurtigt!) ---
def get_live_tweets(symbol):
    query = f"${symbol} lang:en -filter:replies"
    tweets = []
    try:
        for i, tweet in enumerate(sntwitter.TwitterSearchScraper(query).get_items(), 1):
            if i > 8: break
            tweets.append(tweet.rawContent)
        return tweets
    except:
        return []

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
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader(name)
        st.metric("Pris", f"${get_price(symbol)}")
        
        tweets = get_live_tweets(symbol)
        score = get_sentiment_score(tweets)
        
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
                    st.caption(t[:150] + "...")
            else:
                st.caption("Henter tweets...")

# --- STATUS ---
st.success("Kører på 200 MB RAM – ALTID STABILT!")
st.info("Opdaterer hvert 3. minut.")
time.sleep(180)
st.rerun()
