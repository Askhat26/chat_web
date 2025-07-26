

from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from typing import List, Dict
from config import Config

import os

class WebsiteChatbot:
    def __init__(self, llm):
        self.llm = llm
        self.embeddings = OllamaEmbeddings(model=Config.EMBEDDING_MODEL)
        self.vector_db = None

    def _count_tokens(self, text: str) -> int:
        return len(text.split())

    def prepare_vector_db(self, pages: List[Dict[str, str]], persist_dir: str = None) -> int:
        documents = []
        seen = set()
        for page in pages:
            if page['url'] in seen or not page['content']:
                continue
            seen.add(page['url'])
            metadata = {
                'source': page['url'],
                'title': page['title'],
                'summary': page.get('summary', '')
            }
            documents.append(Document(page_content=page['content'], metadata=metadata))

        if not documents:
            raise ValueError("No content found in pages to index.")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=self._count_tokens
        )
        split_docs = splitter.split_documents(documents)

        self.vector_db = Chroma.from_documents(
            documents=split_docs,
            embedding=self.embeddings,
            persist_directory=persist_dir
        )

        if persist_dir:
            self.vector_db.persist()

        return len(split_docs)

    def get_qa_chain(self, chain_type: str = "stuff"):
        if not self.vector_db:
            raise ValueError("Vector DB not initialized. Run prepare_vector_db first.")

        retriever = self.vector_db.as_retriever(search_kwargs={"k": 3})
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=retriever,
            chain_type=chain_type,
            return_source_documents=True
        )

    def chat(self, question: str, qa_chain=None) -> Dict:
        if qa_chain is None:
            qa_chain = self.get_qa_chain()
        result = qa_chain({"query": question})
        return {
            "answer": result["result"],
            "sources": [doc.metadata for doc in result["source_documents"]]
        }

    def load_vector_db(self, persist_dir: str, collection_name: str = "website_content"):
        if not os.path.exists(persist_dir):
            raise FileNotFoundError(f"No vector DB found at: {persist_dir}")
        self.vector_db = Chroma(
            persist_directory=persist_dir,
            collection_name=collection_name,
            embedding_function=self.embeddings
        )
