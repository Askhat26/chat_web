import streamlit as st
from extractor import WebsiteExtractor
from summarizer import WebsiteSummarizer
from indexer import WebsiteIndexer
from chatbot import WebsiteChatbot
from langchain_community.llms import Ollama
from config import Config

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set page config
st.set_page_config(
    page_title="💬 Chat With Any Website",
    page_icon="🌐",
    layout="wide"
)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'website_processed' not in st.session_state:
    st.session_state.website_processed = False
if 'vector_db' not in st.session_state:
    st.session_state.vector_db = None
if 'ollama_configured' not in st.session_state:
    st.session_state.ollama_configured = False

# Sidebar for configuration
with st.sidebar:
    st.title("Configuration")
    
    # Model selection (always visible)
    Config.LLM_MODEL = st.selectbox(
        "Select Model",
        ["llama3", "mistral", "gemma:7b"],
        index=0,
        help="Choose which local LLM to use"
    )
    
    # Temperature
    temperature = st.slider(
        "Creativity (Temperature)",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1,
        help="Higher values make outputs more random"
    )
    
    # Max pages to crawl
    max_pages = st.number_input(
        "Max Pages to Crawl",
        min_value=1,
        max_value=50,
        value=10,
        help="Limit the number of pages processed"
    )
    
    # Ollama connection check
    if st.button("Verify Ollama Connection"):
        try:
            test_llm = Ollama(model=Config.LLM_MODEL)
            test_llm("Hello")
            st.session_state.ollama_configured = True
            st.success("✅ Ollama connection successful!")
        except Exception as e:
            st.error(f"Failed to connect to Ollama: {str(e)}")
            st.session_state.ollama_configured = False
    
    if st.session_state.ollama_configured:
        st.success("Ollama is ready to use")

# Main app interface
st.title("🌐 Chat With Any Website (Local)")
st.markdown("Enter a website URL to crawl, summarize, and chat with its content using local LLMs")

# Only show main functionality if Ollama is configured
if not st.session_state.ollama_configured:
    st.warning("Please verify Ollama connection in the sidebar to continue")
    st.stop()

# URL input and processing
url = st.text_input("Enter Website URL (e.g., https://example.com)", "")
process_button = st.button("Process Website")

if process_button and url:
    with st.spinner("Crawling website..."):
        try:
            # Initialize components
            llm = Ollama(
                model=Config.LLM_MODEL,
                temperature=temperature
            )
            
            # Step 1: Extract content
            extractor = WebsiteExtractor(url, max_pages=max_pages)
            pages = extractor.crawl_website()
            
            if not pages:
                st.error("No content found on this website. Try a different URL.")
                st.stop()
            
            # Step 2: Summarize
            summarizer = WebsiteSummarizer(llm)
            st.session_state.summary = summarizer.summarize_site(pages)
            pages_with_summaries = summarizer.summarize_each_page(pages)
            
            # Step 3: Index
            indexer = WebsiteIndexer(persist_dir="./chroma_db")
            num_chunks = indexer.index_website(pages_with_summaries)
            st.session_state.vector_db = indexer.get_vector_db()
            
            # Step 4: Initialize chatbot
            st.session_state.chatbot = WebsiteChatbot(llm)
            st.session_state.chatbot.vector_db = st.session_state.vector_db
            
            st.session_state.website_processed = True
            st.success(f"Successfully processed {len(pages)} pages and created {num_chunks} chunks!")
            
            # Show summary
            with st.expander("Website Summary"):
                st.write(st.session_state.summary)
                
        except Exception as e:
            st.error(f"Error processing website: {str(e)}")

# Chat interface
if st.session_state.website_processed:
    st.subheader("Chat with the Website")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                with st.expander("Sources"):
                    for source in message["sources"]:
                        st.write(f"- [{source['title']}]({source['source']})")
    
    # Chat input
    if prompt := st.chat_input("Ask something about the website..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get chatbot response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.chatbot.chat(prompt)
                    st.markdown(response["answer"])
                    
                    # Display sources if available
                    if response["sources"]:
                        with st.expander("Sources"):
                            for source in response["sources"]:
                                st.write(f"- [{source['title']}]({source['source']})")
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response["answer"],
                        "sources": response["sources"]
                    })
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")

# Instructions
with st.expander("How to use this app"):
    st.markdown("""
    1. **Select a local LLM model** in the sidebar
    2. **Verify Ollama connection** (make sure Ollama is running)
    3. **Enter a website URL** you want to chat with
    4. Click **"Process Website"** to crawl and index the site
    5. Once processed, **ask questions** about the website content
    
    **Tips:**
    - Start with smaller websites (processing large sites may take time)
    - For better performance, use smaller models like 'mistral'
    - Check the "Sources" to see where the information came from
    """)

# Add some styling
st.markdown("""
<style>
    .stChatMessage {
        padding: 12px;
        border-radius: 8px;
        margin-bottom: 12px;
    }
    .stChatMessage.user {
        background-color: #f0f2f6;
    }
    .stChatMessage.assistant {
        background-color: #e6f7ff;
    }
</style>
""", unsafe_allow_html=True)