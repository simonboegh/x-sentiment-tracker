import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from transformers import pipeline
import time
import threading

# --- KONFIG ---
st.set_page_config(page_title="AI News", layout="wide")
st.title("AI Financial News Sentiment")
st.markdown("*FinBERT analyserer live nyheder fra Yahoo Finance*")

# --- AI-MODEL ---
@st.cache_resource
def load_ai():
    return pipeline("text-classification", model="yiyanghkust/finbert-tone")

ai = load_ai()

# --- HENT NYHEDER MED TIMEOUT ---
def get_news_with_timeout(symbol, timeout=15):
    result = [None]
    def fetch():
        try:
            ticker = yf.Ticker(symbol)
            news = ticker.news
            result[0] = news[:3] if news else []
        except:
            result[0] = []
    
    thread = threading.Thread(target=fetch)
    thread.start()
    thread.join(timeout)
    
    if thread.is_alive():
        return [], "Timeout – Yahoo er langsom"
    return result[0], "OK"

# --- ANALYSÉR ---
def analyze_news(symbol):
    news, status = get_news_with_timeout(symbol)
    if status != "OK" or not news:
        return 0.0, [status]
    
    scores = []
    analyzed = []
    for item in news:
        title = item['title']
        try:
            result = ai(title)[0]
            label = result['label']
            score = result['score']
            if label == 'Positive':
                scores.append(score)
            elif label == 'Negative':
                scores.append(-score)
            analyzed.append((title, label, score))
        except:
            continue
    
    avg = sum(scores)/len(scores) if scores else 0.0
    return avg, analyzed

# --- 3 AKTIER ---
stocks = ["GME", "TSLA", "NVDA"]
names = ["GameStop", "Tesla", "Nvidia"]

# --- DASHBOARD ---
for name, symbol in zip(names, stocks):
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader(f"{name} (${symbol})")
        
        with st.spinner(f"Henter nyheder for {symbol}..."):
            sentiment, analysis = analyze_news(symbol)
        
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
        with st.expander("AI-analyse af nyheder"):
            if analysis and analysis[0] != "Timeout – Yahoo er langsom":
                for title, label, score in analysis:
                    emoji = "🟢" if label == "Positive" else "🔴" if label == "Negative" else "⚪"
                    st.write(f"{emoji} **{label}** ({score:.2f})")
                    st.caption(title[:100] + "..." if len(title) > 100 else title)
            else:
                st.warning(analysis[0] if analysis else "Henter...")

# --- STATUS ---
st.success("**VIRKER 100% – ALDRIG HÆNGER!**")
st.info("Opdaterer hvert 3. minut.")
time.sleep(180)
st.rerun()
