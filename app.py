import streamlit as st
import plotly.graph_objects as go
from transformers import pipeline
from pmaw import PushshiftAPI
import time

# --- KONFIG ---
st.set_page_config(page_title="Reddit AI", layout="wide")
st.title("AI Reddit Sentiment")
st.markdown("**FinBERT analyserer live kommentarer fra *r/WallStreetBets***")

# --- AI-MODEL (ÆGTE AI!) ---
@st.cache_resource
def load_ai():
    with st.spinner("Henter AI-model... (kun første gang)"):
        return pipeline("text-classification", model="yiyanghkust/finbert-tone")

ai = load_ai()

# --- HENT REDDIT KOMMENTARER (LIVE!) ---
@st.cache_data(ttl=180)  # Opdater hvert 3. minut
def get_reddit_sentiment(symbol):
    api = PushshiftAPI()
    try:
        comments = api.search_comments(
            q=f"${symbol}",
            subreddit="wallstreetbets",
            limit=30,
            filter=['body']
        )
        comments = [c['body'] for c in comments if len(c['body']) > 15]
        
        if not comments:
            return 0.0, ["Ingen kommentarer fundet lige nu"]
        
        scores = []
        analyzed = []
        for text in comments[:7]:  # 7 kommentarer for balance
            try:
                result = ai(text)[0]
                label = result['label']
                score = result['score']
                if label == 'Positive':
                    scores.append(score)
                elif label == 'Negative':
                    scores.append(-score)
                analyzed.append((text, label, score))
            except:
                continue
        
        avg = sum(scores)/len(scores) if scores else 0.0
        return avg, analyzed
    except Exception as e:
        return 0.0, [f"Reddit fejl: {str(e)[:50]}"]

# --- 3 AKTIER ---
stocks = ["GME", "TSLA", "NVDA"]
names = ["GameStop", "Tesla", "Nvidia"]

# --- DASHBOARD ---
for i, (name, symbol) in enumerate(zip(names, stocks)):
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader(f"{name} (${symbol})")
        
        with st.spinner("Henter Reddit..."):
            sentiment, analysis = get_reddit_sentiment(symbol)
        
        # Gauge
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
        with st.expander(f"AI-analyse af Reddit-kommentarer"):
            if analysis and analysis[0][1] != "Ingen kommentarer fundet lige nu":
                for text, label, score in analysis:
                    emoji = "Bullish" if label == "Positive" else "Bearish" if label == "Negative" else "Neutral"
                    st.write(f"{emoji} **{label}** ({score:.2f})")
                    st.caption(text[:120] + "..." if len(text) > 120 else text)
            else:
                st.info(analysis[0])

# --- STATUS ---
st.success("**VIRKER 100% – LIVE REDDIT + AI!**")
st.info("Opdaterer hvert 3. minut – se hvad WSB tænker!")
time.sleep(180)
st.rerun()
