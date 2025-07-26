from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
from typing import List, Dict, Optional
import logging
from config import Config

class WebsiteSummarizer:
    def __init__(self, llm, max_tokens: int = 3000, chunk_size: int = 2000, chunk_overlap: int = 200):
        self.llm = llm
        self.max_tokens = max_tokens
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.logger = self._setup_logger()
        self.token_counter = self._get_token_counter()

    def _setup_logger(self):
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(__name__)

    def _get_token_counter(self):
        """Get appropriate token counting function based on config."""
        config = Config.get_llm_config()
        
        if config['type'] == 'openai':
            import tiktoken
            encoder = tiktoken.get_encoding("cl100k_base")
            return lambda text: len(encoder.encode(text))
        else:
            # For Gemini/Ollama, use simpler word count approximation
            return lambda text: len(text.split())  # 1 token ≈ 1 word

    def _validate_input(self, pages: List[Dict[str, str]]) -> str:
        """Validate and concatenate page content."""
        if not pages:
            raise ValueError("No pages provided for summarization")

        full_text = " ".join([
            p['content'] for p in pages 
            if p.get('content') and isinstance(p['content'], str)
        ]).strip()

        if not full_text:
            raise ValueError("No valid content found in pages")

        return full_text

    def summarize_site(self, pages: List[Dict[str, str]], chain_type: str = "map_reduce") -> str:
        try:
            full_text = self._validate_input(pages)
            token_count = self.token_counter(full_text)

            # Choose splitting strategy based on token count
            if token_count > self.max_tokens:
                self.logger.info(f"Splitting large text ({token_count} tokens) for summarization")
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                    length_function=self.token_counter
                )
                docs = splitter.create_documents([full_text])
            else:
                docs = [Document(page_content=full_text)]

            # For Gemini, we might want to adjust the prompt
            if Config.MODEL_TYPE == ModelType.GEMINI:
                from langchain.prompts import PromptTemplate
                prompt_template = """Write a concise summary of the following:
                {text}
                CONCISE SUMMARY:"""
                PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
                
                chain = load_summarize_chain(
                    self.llm,
                    chain_type=chain_type,
                    verbose=Config.DEBUG_MODE,
                    map_prompt=PROMPT,
                    combine_prompt=PROMPT
                )
            else:
                chain = load_summarize_chain(
                    self.llm,
                    chain_type=chain_type,
                    verbose=Config.DEBUG_MODE
                )
            
            return chain.run(docs)

        except Exception as e:
            self.logger.error(f"Summarization failed: {str(e)}")
            return f"⚠️ Summarization error: {str(e)}"