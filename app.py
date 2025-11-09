import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from transformers import pipeline
import snscrape.modules.twitter as sntwitter
import time

# --- KONFIG ---
st.set_page_config(page_title="LIVE X Sentiment", layout="wide")
st.title("LIVE X-Sentiment Tracker (Nasdaq) – GRATIS & UDEN API!")

# --- FINBERT MODEL ---
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis", model="ProsusAI/finbert")

model = load_model()

# --- HENT TWEETS MED SNSCRAPE (GRATIS!) ---
def get_live_tweets(symbol, max_results=20):
    query = f"${symbol} OR '{symbol} stock' lang:en -filter:replies"
    tweets = []
    try:
        for i, tweet in enumerate(sntwitter.TwitterSearchScraper(query).get_items()):
            if i >= max_results:
                break
            tweets.append(tweet.rawContent)
        if tweets:
            st.success(f"**{symbol}: Fundet {len(tweets)} LIVE tweets!**")
        else:
            st.info(f"**{symbol}: Ingen tweets lige nu.**")
        return tweets
    except Exception as e:
        st.error(f"**Fejl:** {str(e)[:50]}")
        return []

# --- SENTIMENT ---
def get_sentiment(tweets):
    if not tweets:
        return 0.0
    scores = []
    for t in tweets[:10]:
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

# --- AKTIER MED HØJ X-AKTIVITET ---
stocks = ["GME", "TSLA", "AMC"]
names = ["GameStop", "Tesla", "AMC"]

# --- DASHBOARD ---
cols = st.columns(3)

for i, (name, symbol) in enumerate(zip(names, stocks)):
    with cols[i]:
        st.subheader(f"{name} (${symbol})")
        
        tweets = get_live_tweets(symbol, max_results=30)
        score = get_sentiment(tweets)
        price = get_price(symbol)
        
        # Gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score * 100,
            title={'text': "LIVE Sentiment"},
            gauge={
                'axis': {'range': [-100, 100]},
                'bar': {'color': "lime" if score > 0.2 else "red" if score < -0.2 else "gray"},
                'steps': [
                    {'range': [-100, -30], 'color': "darkred"},
                    {'range': [-30, 30], 'color': "orange"},
                    {'range': [30, 100], 'color': "lightgreen"}
                ]
            }
        ))
        st.plotly_chart(fig, use_container_width=True)
        
        st.metric("Pris (USD)", f"${price}" if price else "N/A")
        
        with st.expander("LIVE Tweets"):
            for t in tweets[:5]:
                st.caption(t)

# --- AUTO-REFRESH ---
st.info("Opdaterer automatisk hvert 3. minut!")
time.sleep(180)
st.rerun()
