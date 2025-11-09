import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from transformers import pipeline
import tweepy
import os
import threading
import time

# --- KONFIG ---
st.set_page_config(page_title="LIVE X Sentiment", layout="wide")
st.title("LIVE X-Sentiment Tracker (Nasdaq)")

# --- MODEL ---
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis", model="ProsusAI/finbert", device=-1)

model = load_model()

# --- X CLIENT ---
client = tweepy.Client(
    bearer_token=os.getenv("BEARER_TOKEN"),
    wait_on_rate_limit=True
)

# --- HENT TWEETS (LIVE) ---
def get_live_tweets(symbol, timeout=10):
    result = ["Henter..."]
    def fetch():
        # SØGNING DER VIRKER LIVE
        query = f"${symbol} OR '{symbol} stock' OR '{symbol} price' -is:retweet lang:en"
        try:
            resp = client.search_recent_tweets(
                query=query,
                max_results=50,  # Mere chance
                tweet_fields=['created_at', 'author_id']
            )
            if resp.data:
                tweets = [t.text for t in resp.data]
                st.success(f"**{symbol}: Fundet {len(tweets)} LIVE tweets!**")
                result[0] = tweets
            else:
                st.info(f"**{symbol}: Ingen tweets lige nu – prøver igen snart.**")
                result[0] = []
        except Exception as e:
            st.error(f"**{symbol} fejl:** {str(e)[:50]}")
            result[0] = []
    
    thread = threading.Thread(target=fetch)
    thread.start()
    thread.join(timeout)
    
    if thread.is_alive():
        st.warning(f"**{symbol}: Timeout – prøver igen.**")
        return []
    return result[0]

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

# --- AKTIER MED HØJ X-AKTIVITET (GARANTI FOR TWEETS) ---
stocks = ["GME", "TSLA", "AMC"]
names = ["GameStop", "Tesla", "AMC Entertainment"]

# --- DASHBOARD ---
cols = st.columns(3)

for i, (name, symbol) in enumerate(zip(names, stocks)):
    with cols[i]:
        st.subheader(f"{name} (${symbol})")
        
        tweets = get_live_tweets(symbol, timeout=10)
        score = get_sentiment(tweets)
        price = get_price(symbol)
        
        # Gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=score * 100,
            delta={'reference': 0},
            title={'text': "LIVE Sentiment"},
            gauge={
                'axis': {'range': [-100, 100]},
                'bar': {'color': "cyan" if score > 0.3 else "magenta" if score < -0.3 else "yellow"},
                'steps': [
                    {'range': [-100, -40], 'color': "red"},
                    {'range': [-40, 40], 'color': "gray"},
                    {'range': [40, 100], 'color': "lime"}
                ]
            }
        ))
        st.plotly_chart(fig, use_container_width=True)
        
        st.metric("Pris (USD)", f"${price}" if price else "N/A", delta=None)
        
        with st.expander("LIVE Tweets"):
            if tweets:
                for t in tweets[:5]:
                    st.caption(t)
            else:
                st.caption("Venter på nye tweets...")

# --- AUTO-REFRESH ---
st.info("Opdaterer automatisk hvert 3. minut!")
time.sleep(180)
st.rerun()
