from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from typing import List, Dict, Optional
import logging
from config import Config

class WebsiteSummarizer:
    def __init__(self, llm, max_tokens: int = 3000, chunk_size: int = 2000, chunk_overlap: int = 200):
        """
        Website summarizer optimized for Gemini.
        
        Args:
            llm: Initialized Gemini language model
            max_tokens: Maximum tokens to process at once
            chunk_size: Target size for text chunks
            chunk_overlap: Overlap between chunks
        """
        self.llm = llm
        self.max_tokens = max_tokens
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.logger = self._setup_logger()
        self.token_counter = self._get_token_counter()

    def _setup_logger(self):
        """Configure logging for the summarizer."""
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(__name__)

    def _get_token_counter(self):
        """Token counter optimized for Gemini (word-based approximation)."""
        return lambda text: len(text.split())  # Simple word count for Gemini

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
        """
        Generate comprehensive summary of website content.
        
        Args:
            pages: List of page dictionaries with 'content'
            chain_type: Type of summarization chain ('map_reduce', 'stuff', 'refine')
            
        Returns:
            Concise summary text
        """
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

            # Gemini-optimized prompt
            prompt_template = """Write a concise summary of the following:
            {text}
            
            CONCISE SUMMARY (3-5 bullet points):"""
            PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
            
            chain = load_summarize_chain(
                self.llm,
                chain_type=chain_type,
                verbose=Config.DEBUG_MODE,
                map_prompt=PROMPT,
                combine_prompt=PROMPT
            )
            
            return chain.run(docs)

        except Exception as e:
            self.logger.error(f"Summarization failed: {str(e)}")
            return f"⚠️ Summarization error: {str(e)}"

    def summarize_each_page(self, pages: List[Dict[str, str]], max_summary_length: int = 500) -> List[Dict[str, str]]:
        """
        Generate individual summaries for each page.
        
        Args:
            pages: List of page dicts with 'url', 'title', and 'content'
            max_summary_length: Target maximum length for each summary
            
        Returns:
            List of page dicts with added 'summary' field
        """
        results = []
        for page in pages:
            try:
                if not page.get('content'):
                    self.logger.warning(f"Skipping empty page: {page.get('url')}")
                    continue

                self.logger.info(f"Summarizing: {page.get('title', page['url'])}")
                
                prompt = f"""
                Summarize this page in under {max_summary_length} characters:
                {page['content'][:10000]}  # Limit input size
                """
                
                docs = [Document(page_content=prompt)]
                chain = load_summarize_chain(
                    self.llm,
                    chain_type="stuff",
                    verbose=Config.DEBUG_MODE
                )
                
                summary = chain.run(docs)
                results.append({
                    **page,
                    'summary': summary[:max_summary_length]  # Enforce length limit
                })

            except Exception as e:
                self.logger.error(f"Failed to summarize {page.get('url')}: {str(e)}")
                results.append({
                    **page,
                    'summary': f"⚠️ Summary error: {str(e)}"
                })

        return results