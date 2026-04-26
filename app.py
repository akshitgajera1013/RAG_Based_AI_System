# =========================================================================================
# 🎓 SIGMA KNOWLEDGE NEXUS (ENTERPRISE RAG EDITION - HYBRID BUILD)
# Version: 18.4.0 | Build: Local DB + Official HF SDK + Gemini Synthesis
# Description: Advanced Vector Search Dashboard for Video Course Navigation.
# Features Local Repo DB loading, Cosine Similarity Analytics, and Hybrid AI Routing.
# Theme: Sigma Nexus (Deep Space Black, Neon Purple, Cyber Blue)
# =========================================================================================

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd
import time
import base64
import json
from datetime import datetime
import uuid
import os
import joblib
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

# Initialize environment variables from .env file
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# =========================================================================================
# 1. PAGE CONFIGURATION & SECURE INITIALIZATION
# =========================================================================================
st.set_page_config(
    page_title="Sigma RAG | Course Navigator",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================================================
# 2. VECTOR DATABASE INGESTION (LOCAL REPOSITORY FETCH)
# =========================================================================================
DB_FILE = "embeddings.joblib"

@st.cache_resource(show_spinner=False)
def load_vector_database():
    """
    Loads the Joblib DataFrame containing Whisper transcripts and bge-m3 embeddings
    directly from the local repository directory.
    """
    try:
        if os.path.exists(DB_FILE):
            return joblib.load(DB_FILE)
        else:
            st.sidebar.error(f"Vector DB File ('{DB_FILE}') not found in the root directory.")
            return None
    except Exception as e:
        st.sidebar.error(f"Vector DB Load Error: {str(e)}")
        return None

vector_df = load_vector_database()

# =========================================================================================
# 3. ENTERPRISE CSS INJECTION (SIGMA THEME)
# =========================================================================================
st.markdown(
"""<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800;900&family=Inter:wght@300;400;500;700&family=Space+Mono:wght@400;700&display=swap');

:root {
    --bg-dark: #0a0a0f;
    --bg-panel: rgba(15, 15, 25, 0.8);
    --neon-purple: #b535ff;  
    --cyber-blue: #00e5ff; 
    --text-main: #f8fafc;
    --text-muted: #8b8b9f;
    --glass-border: rgba(181, 53, 255, 0.2);
    --glow-purple: 0 0 35px rgba(181, 53, 255, 0.2);
    --glow-blue: 0 0 30px rgba(0, 229, 255, 0.15);
}

.stApp { background: var(--bg-dark); font-family: 'Inter', sans-serif; color: var(--text-muted); overflow-x: hidden; }
h1, h2, h3, h4, h5, h6 { font-family: 'Outfit', sans-serif; color: var(--text-main); }

/* Grid Background Animation */
.stApp::before {
    content: ''; position: fixed; inset: 0;
    background: radial-gradient(circle at 50% 0%, rgba(181, 53, 255, 0.05) 0%, transparent 50%), radial-gradient(circle at 50% 100%, rgba(0, 229, 255, 0.05) 0%, transparent 50%);
    z-index: 0; pointer-events: none;
}

/* Container Spacing */
.main .block-container { position: relative; z-index: 1; padding-top: 30px; padding-bottom: 90px; max-width: 1600px; }

/* Hero Section */
.hero { text-align: center; padding: 50px 20px 30px; animation: slideDown 0.8s ease-out both; }
@keyframes slideDown { from { opacity: 0; transform: translateY(-30px); } to { opacity: 1; transform: translateY(0); } }

.hero-badge {
    display: inline-flex; align-items: center; gap: 12px;
    background: rgba(181, 53, 255, 0.05); border: 1px solid rgba(181, 53, 255, 0.3);
    border-radius: 5px; padding: 8px 25px; font-family: 'Space Mono', monospace; font-size: 12px;
    color: var(--neon-purple); letter-spacing: 3px; text-transform: uppercase; margin-bottom: 20px; box-shadow: var(--glow-purple);
}
.hero-title { font-family: 'Outfit', sans-serif; font-size: clamp(35px, 5vw, 70px); font-weight: 900; letter-spacing: 2px; line-height: 1.1; margin-bottom: 15px; text-transform: uppercase; }
.hero-title em { font-style: normal; color: var(--cyber-blue); text-shadow: var(--glow-blue); }
.hero-sub { font-family: 'Space Mono', monospace; font-size: 14px; font-weight: 400; color: var(--text-muted); letter-spacing: 5px; text-transform: uppercase; }

/* Glass Panels */
.glass-panel { background: var(--bg-panel); border: 1px solid var(--glass-border); border-radius: 12px; padding: 35px; margin-bottom: 30px; position: relative; overflow: hidden; backdrop-filter: blur(20px); transition: all 0.3s ease; }
.glass-panel:hover { border-color: rgba(0, 229, 255, 0.5); box-shadow: var(--glow-blue); transform: translateY(-2px); }
.panel-heading { font-family: 'Outfit', sans-serif; font-size: 22px; font-weight: 800; color: var(--text-main); letter-spacing: 1px; margin-bottom: 30px; border-bottom: 1px solid rgba(181, 53, 255, 0.2); padding-bottom: 12px; text-transform: uppercase; }

/* Text Area */
div[data-testid="stTextArea"] label, div[data-testid="stTextInput"] label { display: none !important; }
div[data-testid="stTextArea"] > div > div > textarea, div[data-testid="stTextInput"] > div > div > input {
    background: rgba(0, 0, 0, 0.6) !important; border: 1px solid rgba(0, 229, 255, 0.3) !important;
    color: var(--cyber-blue) !important; font-family: 'Inter', sans-serif !important; border-radius: 8px !important;
    padding: 20px !important; transition: all 0.3s ease !important; font-size: 18px !important;
}
div[data-testid="stTextArea"] > div > div > textarea:focus, div[data-testid="stTextInput"] > div > div > input:focus { border-color: var(--neon-purple) !important; box-shadow: inset 0 0 15px rgba(181, 53, 255, 0.1) !important; }

/* Execute Button */
div.stButton > button {
    width: 100% !important; background: transparent !important; color: var(--neon-purple) !important; font-family: 'Space Mono', monospace !important;
    font-size: 16px !important; font-weight: 700 !important; letter-spacing: 4px !important; text-transform: uppercase !important; border: 1px solid var(--neon-purple) !important;
    border-radius: 8px !important; padding: 25px !important; cursor: pointer !important; transition: all 0.3s ease !important;
    background-color: rgba(181, 53, 255, 0.05) !important; margin-top: 20px !important; box-shadow: 0 5px 20px rgba(181, 53, 255, 0.1) !important;
}
div.stButton > button:hover { background-color: var(--neon-purple) !important; transform: translateY(-2px) !important; box-shadow: var(--glow-purple) !important; color: #fff !important; }

/* RAG Response Box */
.rag-response { background: rgba(0, 229, 255, 0.05) !important; border: 1px solid var(--cyber-blue) !important; padding: 40px !important; border-radius: 12px !important; position: relative !important; margin-top: 20px !important; box-shadow: var(--glow-blue) !important; animation: popIn 0.8s ease both !important; }
.rag-response p { font-family: 'Inter', sans-serif; font-size: 16px; color: var(--text-main); line-height: 1.8; }
.rag-response strong { color: var(--neon-purple); }

/* Chunk Cards */
.chunk-card { background: rgba(0,0,0,0.5); border-left: 4px solid var(--neon-purple); padding: 20px; border-radius: 6px; margin-bottom: 15px; }
.chunk-title { font-family: 'Outfit', sans-serif; font-size: 18px; color: var(--text-main); font-weight: 700; margin-bottom: 8px;}
.chunk-meta { font-family: 'Space Mono', monospace; font-size: 12px; color: var(--cyber-blue); margin-bottom: 12px;}
.chunk-text { font-family: 'Inter', sans-serif; font-size: 14px; color: var(--text-muted); line-height: 1.6;}

@keyframes popIn { from { opacity: 0; transform: scale(0.98); } to { opacity: 1; transform: scale(1); } }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { background: rgba(0,0,0,0.5) !important; border-radius: 8px !important; border: 1px solid rgba(181, 53, 255, 0.1) !important; padding: 6px !important; gap: 8px !important; }
.stTabs [data-baseweb="tab"] { font-family: 'Space Mono', monospace !important; font-size: 12px !important; font-weight: 700 !important; letter-spacing: 2px !important; text-transform: uppercase !important; color: var(--text-muted) !important; border-radius: 6px !important; padding: 15px 25px !important; transition: 0.3s !important; }
.stTabs [aria-selected="true"] { background: rgba(181, 53, 255, 0.1) !important; color: var(--neon-purple) !important; border: 1px solid rgba(181, 53, 255, 0.3) !important; box-shadow: inset 0 0 15px rgba(181, 53, 255, 0.05) !important; }

/* Sidebar */
section[data-testid="stSidebar"] { background: #040406 !important; border-right: 1px solid rgba(0, 229, 255, 0.15) !important; }
.sb-logo-text { font-family: 'Outfit', sans-serif; font-size: 26px; font-weight: 900; color: var(--text-main); letter-spacing: 4px; text-transform: uppercase; }
.sb-title { font-family: 'Space Mono', monospace; font-size: 12px; font-weight: 700; color: var(--text-muted); letter-spacing: 4px; text-transform: uppercase; margin-bottom: 15px; border-bottom: 1px solid rgba(255, 255, 255, 0.05); padding-bottom: 8px; margin-top: 30px; }
.telemetry-card { background: rgba(0, 0, 0, 0.8) !important; border: 1px solid rgba(181, 53, 255, 0.2) !important; padding: 18px !important; border-radius: 8px !important; text-align: center !important; margin-bottom: 12px !important; }
.telemetry-val { font-family: 'Outfit', sans-serif; font-size: 20px; font-weight: 800; color: var(--cyber-blue); }
.telemetry-lbl { font-family: 'Space Mono', monospace; font-size: 9px; color: var(--text-muted); letter-spacing: 2px; text-transform: uppercase; margin-top: 6px; }
</style>""", unsafe_allow_html=True)

# =========================================================================================
# 4. SESSION STATE MANAGEMENT & HYBRID API ROUTING
# =========================================================================================
if "session_id" not in st.session_state: st.session_state["session_id"] = f"RAG-IDX-{str(uuid.uuid4())[:8].upper()}"
if "query" not in st.session_state: st.session_state["query"] = ""
if "llm_response" not in st.session_state: st.session_state["llm_response"] = None
if "context_chunks" not in st.session_state: st.session_state["context_chunks"] = None
if "similarity_scores" not in st.session_state: st.session_state["similarity_scores"] = None
if "compute_latency" not in st.session_state: st.session_state["compute_latency"] = 0.0

def create_embedding(text_list, hf_token):
    """Hits HuggingFace Serverless API using the official, robust Python SDK."""
    try:
        client = InferenceClient(token=hf_token)
        
        # The SDK completely handles URL routing, headers, and parsing
        embeddings = client.feature_extraction(text_list, model="BAAI/bge-m3")
        
        # Convert NumPy array to standard Python list if necessary
        if hasattr(embeddings, "tolist"):
            return embeddings.tolist()
            
        return embeddings
        
    except Exception as e:
        st.error(f"Hugging Face SDK Error: {str(e)}")
        return None

def inference(prompt):
    """Hits Google Gemini API by automatically discovering your allowed models."""
    try:
        # Step 1: Ask Google for a list of all models your API key has access to
        valid_models = []
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                valid_models.append(m.name)
                
        if not valid_models:
            return "Gemini API Error: Your API key does not have access to any text generation models."
            
        # Step 2: Automatically select the best available model (Prefer 'flash' for speed)
        target_model = valid_models[0] # Default to the first valid one
        for model_name in valid_models:
            if "flash" in model_name.lower():
                target_model = model_name
                break
                
        # Step 3: Generate the response using the verified model
        model = genai.GenerativeModel(target_model)
        response = model.generate_content(prompt)
        return response.text
        
    except Exception as e:
        return f"Gemini Dynamic API Error: {str(e)}"

# =========================================================================================
# 5. ENTERPRISE SIDEBAR LOGIC (SYSTEM TELEMETRY)
# =========================================================================================
with st.sidebar:
    st.markdown(
f"""<div style='text-align:center; padding:20px 0 30px;'>
<div class="sb-logo-text">SIGMA RAG</div>
<div style="font-family:'Space Mono'; font-size:10px; color:var(--neon-purple); letter-spacing:3px; margin-top:8px;">KNOWLEDGE KERNEL</div>
<div style="font-family:'Space Mono'; font-size:9px; color:rgba(255,255,255,0.2); margin-top:12px;">ID: {st.session_state["session_id"]}</div>
</div>""", unsafe_allow_html=True)

    st.markdown('<div class="sb-title">⚙️ Pipeline Specs</div>', unsafe_allow_html=True)
    st.markdown(
"""<div style="background:rgba(0,0,0,0.6); padding:18px; border-radius:8px; border:1px solid rgba(181, 53, 255, 0.15); font-family:Inter; font-size:12px; color:rgba(248,250,252,0.7); line-height:1.8;">
<b>Audio Extraction:</b> FFmpeg<br>
<b>Transcription:</b> Whisper (Large-v2)<br>
<b>Vectorization:</b> Hugging Face (bge-m3)<br>
<b>Generator:</b> Gemini (1.5 Flash)<br>
<b>Security:</b> .env Synchronized<br>
</div>""", unsafe_allow_html=True)

    st.markdown('<div class="sb-title">📊 Database Telemetry</div>', unsafe_allow_html=True)
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        v_count = len(vector_df) if vector_df is not None else 0
        st.markdown(f'<div class="telemetry-card"><div class="telemetry-val" style="color:var(--neon-purple);">{v_count:,}</div><div class="telemetry-lbl">Vectors</div></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="telemetry-card"><div class="telemetry-val" style="font-size:20px;">Top 5</div><div class="telemetry-lbl">K-Retrieval</div></div>', unsafe_allow_html=True)
    with col_s2:
        st.markdown('<div class="telemetry-card"><div class="telemetry-val" style="color:var(--cyber-blue);">1024</div><div class="telemetry-lbl">Dimensions</div></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="telemetry-card"><div class="telemetry-val" style="font-size:20px;">{st.session_state["compute_latency"]}s</div><div class="telemetry-lbl">Latency</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.session_state["llm_response"] is None:
        st.markdown("""<div style="padding:15px; border-left:3px solid var(--text-muted); background:rgba(255,255,255,0.02); font-family:Inter; font-size:12px; color:var(--text-muted);"><b>STANDBY</b>: Awaiting student query.</div>""", unsafe_allow_html=True)
    else:
        st.markdown("""<div style="padding:15px; border-left:3px solid var(--neon-purple); background:rgba(181, 53, 255,0.05); font-family:Inter; font-size:12px; color:var(--neon-purple);"><b>RAG SYNTHESIS COMPLETE</b></div>""", unsafe_allow_html=True)

# =========================================================================================
# 6. HERO HEADER SECTION
# =========================================================================================
st.markdown(
"""<div class="hero">
<div class="hero-badge">SERVERLESS HYBRID RAG | HUGGINGFACE + GEMINI</div>
<div class="hero-title">SIGMA COURSE <em>NAVIGATOR</em></div>
<div class="hero-sub">AI-Powered Vector Search & Video Timestamp Retrieval</div>
</div>""", unsafe_allow_html=True)

# =========================================================================================
# 7. MAIN APPLICATION TABS
# =========================================================================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🔍 KNOWLEDGE SEARCH", 
    "🎞️ SOURCE CHUNKS", 
    "🧠 RAG TOPOLOGY", 
    "📊 VECTOR ANALYTICS",
    "⚙️ SECURITY STATUS",
    "📋 EXPORT DOSSIER"
])

# =========================================================================================
# TAB 1 - KNOWLEDGE SEARCH (INFERENCE)
# =========================================================================================
with tab1:
    st.markdown('<div class="glass-panel"><div class="panel-heading">💬 Ask The Course Database</div>', unsafe_allow_html=True)
    
    query = st.text_input("Query Input", placeholder="E.g., Which video teaches about MongoDB connections, and at what time?")
    
    if st.button("EXECUTE VECTOR SEARCH & SYNTHESIZE"):
        if not GEMINI_API_KEY or not HF_TOKEN:
            st.error("⚠️ Missing API Keys. Please ensure your `.env` file contains both `GEMINI_API_KEY` and `HF_TOKEN`.")
        elif not query.strip():
            st.warning("Please enter a query.")
        elif vector_df is None:
            st.error("Vector Database not loaded.")
        else:
            with st.spinner("Embedding query via Hugging Face, calculating Cosine Similarity, and triggering Synthesis..."):
                start_time = time.time()
                st.session_state["query"] = query
                
                # 1. Embed Query via Hugging Face SDK
                q_embedding = create_embedding([query], HF_TOKEN)
                
                if q_embedding is None:
                    # Error is handled directly within create_embedding now
                    pass
                else:
                    # 2. Calculate Cosine Similarity
                    question_vector = q_embedding[0]
                    similarities = cosine_similarity(np.vstack(vector_df['embedding']), [question_vector]).flatten()
                    
                    # 3. Retrieve Top-K
                    top_results = 5
                    max_indx = similarities.argsort()[::-1][0:top_results]
                    new_df = vector_df.iloc[max_indx]
                    
                    st.session_state["context_chunks"] = new_df
                    st.session_state["similarity_scores"] = similarities[max_indx]
                    
                    # 4. Construct Prompt
                    json_context = new_df[["title", "number", "start", "end", "text"]].to_json(orient="records")
                    
                    prompt = f"""I am teaching web development in my Sigma web development course. Here are video subtitle chunks containing video title, video number, start time in seconds, end time in seconds, the text at that time:

                    {json_context}
                    ---------------------------------
                    "{query}"
                    User asked this question related to the video chunks, you have to answer in a human way (dont mention the above format, its just for you) where and how much content is taught in which video (in which video and at what timestamp) and guide the user to go to that particular video. If user asks unrelated question, tell him that you can only answer questions related to the course"""

                    # 5. Call Generator LLM (Gemini)
                    response = inference(prompt)
                    
                    st.session_state["llm_response"] = response
                    st.session_state["compute_latency"] = round(time.time() - start_time, 2)
                    st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

    # Render RAG Response
    if st.session_state["llm_response"]:
        st.markdown(
f"""<div class="rag-response">
<div style="font-family:'Space Mono'; font-size:12px; color:var(--cyber-blue); letter-spacing:4px; text-transform:uppercase; margin-bottom:15px;">⚡ GEMINI SYNTHESIZED OUTPUT</div>
{st.session_state["llm_response"]}
</div>""", unsafe_allow_html=True)

# =========================================================================================
# TAB 2 - SOURCE CHUNKS (RETRIEVED CONTEXT)
# =========================================================================================
with tab2:
    if st.session_state["context_chunks"] is None:
        st.markdown("""<div style='text-align:center; padding:150px 20px; font-family:"Outfit"; font-size:18px; letter-spacing:4px; color:rgba(255,255,255,0.2);'>⚠️ Execute Search To View Retrieved Video Chunks</div>""", unsafe_allow_html=True)
    else:
        st.markdown('<div class="panel-heading" style="border:none;">🎞️ Top 5 Relevant Subtitle Vectors</div>', unsafe_allow_html=True)
        
        df = st.session_state["context_chunks"]
        scores = st.session_state["similarity_scores"]
        
        for i, (index, row) in enumerate(df.iterrows()):
            # Format timestamps from seconds to MM:SS
            start_m, start_s = divmod(int(row['start']), 60)
            end_m, end_s = divmod(int(row['end']), 60)
            time_str = f"{start_m:02d}:{start_s:02d} - {end_m:02d}:{end_s:02d}"
            
            st.markdown(
f"""<div class="chunk-card">
<div class="chunk-title">Video #{row['number']}: {row['title']}</div>
<div class="chunk-meta">⏱️ Timestamp: {time_str} | 🎯 Vector Match: {scores[i]*100:.1f}%</div>
<div class="chunk-text">"{row['text']}"</div>
</div>""", unsafe_allow_html=True)

# =========================================================================================
# TAB 3 - RAG TOPOLOGY
# =========================================================================================
with tab3:
    st.markdown('<div class="panel-heading" style="border:none;">🧠 Retrieval-Augmented Generation (RAG)</div>', unsafe_allow_html=True)
    st.info("💡 **Architectural Insight:** LLMs do not inherently 'watch' your videos. This system bridges the gap by transcribing videos, converting those transcripts into high-dimensional mathematics (vectors), and finding the closest mathematical match to your specific question.")
    
    st.markdown(
"""<div style="background:rgba(0,0,0,0.4); border:1px solid rgba(181, 53, 255,0.3); border-radius:12px; padding:30px; margin-bottom:40px;">
<h3 style="color:var(--neon-purple); margin-top:0; font-family:'Space Mono'; border-bottom:1px solid rgba(181, 53, 255,0.2); padding-bottom:10px;">🧬 PIPELINE EXTRACTION LAYERS</h3>
<div style="display:flex; flex-wrap:wrap; gap:20px; margin-top:20px;">
<div style="flex: 1 1 30%; background:rgba(255,255,255,0.02); padding:20px; border-radius:8px;">
<code style="color:var(--neon-purple); font-size:16px;">1. Whisper Transcription</code>
<p style="color:var(--text-muted); font-size:13px; margin-top:10px;">Audio is extracted via FFmpeg and passed to OpenAI Whisper. The model transcribes the Hindi speech into English text, attaching exact start/end timestamps to every sentence.</p>
</div>
<div style="flex: 1 1 30%; background:rgba(255,255,255,0.02); padding:20px; border-radius:8px;">
<code style="color:var(--neon-purple); font-size:16px;">2. Text Vectorization (Hugging Face)</code>
<p style="color:var(--text-muted); font-size:13px; margin-top:10px;">Each text chunk is sent to Hugging Face via the official SDK, where the BGE-M3 model maps the semantic meaning of the sentence into a 1024-dimensional array of numbers.</p>
</div>
<div style="flex: 1 1 30%; background:rgba(255,255,255,0.02); padding:20px; border-radius:8px;">
<code style="color:var(--neon-purple); font-size:16px;">3. Cosine Similarity Engine</code>
<p style="color:var(--text-muted); font-size:13px; margin-top:10px;">When you ask a question, your query is also vectorized. The system calculates the angle (Cosine) between your query vector and all video chunk vectors. The closest angles win.</p>
</div>
<div style="flex: 1 1 30%; background:rgba(255,255,255,0.02); padding:20px; border-radius:8px;">
<code style="color:var(--neon-purple); font-size:16px;">4. Context Injection</code>
<p style="color:var(--text-muted); font-size:13px; margin-top:10px;">The top 5 matching chunks (including their timestamps and video titles) are formatted into a JSON string and injected directly into a hidden prompt.</p>
</div>
<div style="flex: 1 1 30%; background:rgba(255,255,255,0.02); padding:20px; border-radius:8px;">
<code style="color:var(--neon-purple); font-size:16px;">5. LLM Synthesis (Gemini 1.5)</code>
<p style="color:var(--text-muted); font-size:13px; margin-top:10px;">Google Gemini receives your query AND the exact context. It is instructed to read the context and generate a helpful, conversational answer guiding you to the right video instantly.</p>
</div>
</div>
</div>""", unsafe_allow_html=True)

# =========================================================================================
# TAB 4 - VECTOR ANALYTICS
# =========================================================================================
with tab4:
    if st.session_state["similarity_scores"] is None:
        st.markdown("""<div style='text-align:center; padding:150px 20px; font-family:"Outfit"; font-size:18px; letter-spacing:4px; color:rgba(255,255,255,0.2);'>⚠️ Execute Search To Access Vector Analytics</div>""", unsafe_allow_html=True)
    else:
        st.markdown('<div class="panel-heading" style="border:none;">📊 Top-K Cosine Similarity Distribution</div>', unsafe_allow_html=True)
        st.info("This chart visualizes the mathematical proximity (Cosine Similarity) of the Top 5 retrieved video chunks relative to your specific query vector.")
        
        scores = st.session_state["similarity_scores"]
        df = st.session_state["context_chunks"]
        labels = [f"Vid #{row['number']} - {row['start']}s" for idx, row in df.iterrows()]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=scores, y=labels, orientation='h',
            marker=dict(color=scores, colorscale='Purples', line=dict(color='#b535ff', width=1))
        ))
        
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.01)", 
            font=dict(family="Inter", color="#f8fafc"), 
            xaxis=dict(title="Cosine Similarity Score (1.0 = Perfect Semantic Match)", gridcolor="rgba(255,255,255,0.05)"), 
            yaxis=dict(autorange="reversed"),
            height=450, margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)

# =========================================================================================
# TAB 5 - SYSTEM CONFIG (.ENV SECURITY STATUS)
# =========================================================================================
with tab5:
    st.markdown('<div class="glass-panel"><div class="panel-heading">🔐 Environment Security Status</div>', unsafe_allow_html=True)
    
    if GEMINI_API_KEY:
        st.success("✅ **Gemini API Key:** Successfully loaded from secure `.env` file or Streamlit Secrets.")
    else:
        st.error("❌ **Gemini API Key:** Not found. Please add `GEMINI_API_KEY` to your environment.")
        
    if HF_TOKEN:
        st.success("✅ **Hugging Face Token:** Successfully loaded from secure `.env` file or Streamlit Secrets.")
    else:
        st.error("❌ **Hugging Face Token:** Not found. Please add `HF_TOKEN` to your environment.")
        
    st.info("The system is now fully localized and secure. API Keys are no longer stored in active session state.")
    st.markdown('</div>', unsafe_allow_html=True)

# =========================================================================================
# TAB 6 - EXPORT DOSSIER
# =========================================================================================
with tab6:
    if st.session_state["llm_response"] is None:
        st.markdown("""<div style='text-align:center; padding:150px 20px; font-family:"Outfit"; font-size:18px; letter-spacing:4px; color:rgba(255,255,255,0.2);'>⚠️ Execute Search To Generate Export Dossier</div>""", unsafe_allow_html=True)
    else:
        st.markdown('<div class="panel-heading" style="border:none;">💾 Export Search Artifacts</div>', unsafe_allow_html=True)
        col_exp1, col_exp2 = st.columns(2)
        
        json_payload = {
            "metadata": {"query": st.session_state["query"], "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")},
            "llm_synthesized_response": st.session_state["llm_response"],
            "retrieved_context": st.session_state["context_chunks"][["title", "number", "start", "end", "text"]].to_dict(orient="records")
        }
        json_str = json.dumps(json_payload, indent=4)
        b64_json = base64.b64encode(json_str.encode()).decode()
        
        with col_exp1:
            href_json = f'<a href="data:application/json;base64,{b64_json}" download="Sigma_RAG_Export.json" style="display:block; text-align:center; padding:25px; background:rgba(181, 53, 255, 0.1); border:1px solid var(--neon-purple); color:var(--neon-purple); text-decoration:none; font-family:\'Space Mono\'; font-weight:700; font-size:16px; border-radius:8px; letter-spacing:2px; transition:all 0.3s ease;">⬇️ EXPORT JSON DOSSIER</a>'
            st.markdown(href_json, unsafe_allow_html=True)

        st.markdown('<div class="panel-heading" style="border:none; margin-top:70px;">💻 Raw Context Payload</div>', unsafe_allow_html=True)
        st.json(json_payload)

# =========================================================================================
# 8. GLOBAL FOOTER
# =========================================================================================
st.markdown(
"""<div style="text-align:center; padding:70px; margin-top:100px; border-top:1px solid rgba(255,255,255,0.05); font-family:'Space Mono'; font-size:11px; color:rgba(148,163,184,0.3); letter-spacing:4px; text-transform:uppercase;">
&copy; 2026 | Sigma Knowledge Nexus v18.4<br>
<span style="color:rgba(0, 229, 255,0.5); font-size:10px; display:block; margin-top:10px;">Powered by Hybrid Google Gemini + HuggingFace Architecture</span>
</div>""", unsafe_allow_html=True)