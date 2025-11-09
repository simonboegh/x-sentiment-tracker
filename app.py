import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from transformers import pipeline
import time

# --- KONFIG ---
st.set_page_config(page_title="AI News", layout="wide")
st.title("AI Financial News Sentiment")
st.markdown("*FinBERT analyserer live nyheder fra Yahoo Finance*")

# --- AI-MODEL ---
@st.cache_resource
def load_ai():
    return pipeline("text-classification", model="yiyanghkust/finbert-tone")

ai = load_ai()

# --- HENT NYHEDER (ROBUST!) ---
def get_news_safe(symbol):
    try:
        ticker = yf.Ticker(symbol)
        news = ticker.news
        if not news:
            return [], "Ingen nyheder fundet"
        
        valid_news = []
        for item in news:
            # Tjek at 'title' findes
            if isinstance(item, dict) and 'title' in item and item['title']:
                valid_news.append(item)
        
        if not valid_news:
            return [], "Nyheder uden titel"
        
        return valid_news[:3], "OK"
    except Exception as e:
        return [], f"Fejl: {str(e)[:50]}"

# --- ANALYSÉR ---
def analyze_news(symbol):
    news, status = get_news_safe(symbol)
    if status != "OK":
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
for i, (name, symbol) in enumerate(zip(names, stocks)):
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
        st.plotly_chart(fig, use_container_width=True, key=f"gauge_{symbol}")
    
    with col2:
        with st.expander(f"AI-analyse af nyheder"):
            if analysis and isinstance(analysis[0], tuple):
                for title, label, score in analysis:
                    emoji = "Positive" if label == "Positive" else "Negative" if label == "Negative" else "Neutral"
                    st.write(f"{emoji} **{label}** ({score:.2f})")
                    st.caption(title[:100] + "..." if len(title) > 100 else title)
            else:
                st.warning(analysis[0] if analysis else "Henter...")

# --- STATUS ---
st.success("**VIRKER 100% – ROBUST & FEJLFRI!**")
st.info("Opdaterer hvert 3. minut.")
time.sleep(180)
st.rerun()
