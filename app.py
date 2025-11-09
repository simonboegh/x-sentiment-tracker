import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from transformers import pipeline
import time

# --- KONFIG ---
st.set_page_config(page_title="AI News Sentiment", layout="wide")
st.title("AI Financial News Sentiment")
st.markdown("*FinBERT analyserer live nyheder fra Yahoo Finance*")

# --- AI-MODEL (FinBERT – ÆGTE AI!) ---
@st.cache_resource
def load_ai():
    return pipeline("text-classification", model="yiyanghkust/finbert-tone")

ai = load_ai()

# --- HENT NYHEDER + ANALYSÉR MED AI ---
def analyze_news(symbol):
    try:
        ticker = yf.Ticker(symbol)
        news = ticker.news[:5]  # 5 nyheder
        sentiments = []
        for item in news:
            title = item['title']
            result = ai(title)[0]
            label = result['label']
            score = result['score']
            if label == 'Positive':
                sentiments.append(score)
            elif label == 'Negative':
                sentiments.append(-score)
            # Neutral → 0
        return sum(sentiments)/len(sentiments) if sentiments else 0.0, news
    except:
        return 0.0, []

# --- AKTIER ---
stocks = ["GME", "TSLA", "NVDA"]
names = ["GameStop", "Tesla", "Nvidia"]

# --- DASHBOARD ---
for name, symbol in zip(names, stocks):
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader(f"{name} (${symbol})")
        sentiment, news = analyze_news(symbol)
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=sentiment * 100,
            title={'text': "AI Sentiment"},
            gauge={
                'axis': {'range': [-100, 100]},
                'bar': {'color': "lime" if sentiment > 0.1 else "red" if sentiment < -0.1 else "gray"}
            }
        ))
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        with st.expander(f"AI-analyse af nyheder"):
            if news:
                for item in news:
                    title = item['title']
                    result = ai(title)[0]
                    label = result['label']
                    emoji = "🟢" if label == "Positive" else "🔴" if label == "Negative" else "⚪"
                    st.write(f"{emoji} **{label}**: {title[:80]}...")
            else:
                st.write("Henter nyheder...")

# --- STATUS ---
st.success("**ÆGTE AI I PRAKSIS** – FinBERT analyserer live nyheder!")
st.info("Opdaterer hvert 3. minut.")
time.sleep(180)
st.rerun()
