import time
import os
import json
import chromadb
import re
import numpy as np
from src.custom_logger import logging
from typing import List
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader, 
    TextLoader, 
    UnstructuredMarkdownLoader
)
from src.config import VECTOR_DB_PATH, EMBEDDING_MODEL

class EmbeddingManager:
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model_name = model_name
        self.model = None
        logging.info("Initializing Embedding Manager with model: %s", self.model_name)
        
        self._load_model()

    def _load_model(self):
        # Loading the model
        try:
            print(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
        except Exception as e:
            print(f"Error loading model {self.model_name}: {e}")
            raise
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        if not self.model:
            raise ValueError("Model not loaded")
        embeddings = self.model.encode(texts)
        return embeddings

def load_any_document(file_path):
    """Selects the correct loader based on file extension."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return PyPDFLoader(file_path).load()
    elif ext == ".md":
        return UnstructuredMarkdownLoader(file_path).load()
    elif ext == ".txt":
        return TextLoader(file_path).load()
    else:
        return []
    
def load_and_split_data(DOCS_FOLDER: str) -> list:
    """
    Loads text files and chunks them using a Context-Aware Strategy:
    1. Tries to split by Numbered Headers (e.g., "1. POLICY") and injects context.
    2. Falls back to Recursive Splitting if no headers are found.
    """
    
    final_chunks = []
    
    # Fallback splitter for unstructured files
    fallback_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    for filename in os.listdir(DOCS_FOLDER):
        if not filename.endswith(".txt"):
            continue

        file_path = os.path.join(DOCS_FOLDER, filename)
        
        try:
            logging.info("Loading %s...", filename)
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            logging.info(f"Error reading {filename}: {e}")
            continue

        # Context-Aware Splitting (Secion Numbers)
        # Regex looks for "\n1. " to capture the header.
        sections = re.split(r'\n(\d+\.\s.*)', content)
        
        doc_chunks = []
        policy_category = filename.replace("_", " ").replace(".txt", "").title()
        document_title = content.split('\n')[0].strip() # Assume first line is title
        
        logging.info(f"Chunking the {filename} file content...")
        
        
        # If regex found sections (Length > 1 means split was successful)
        if len(sections) > 1:
            # sections[0] is usually intro text, skip it or treat as intro
            # Loop starts from 1 and steps by 2 because regex split keeps delimiters
            for i in range(1, len(sections), 2):
                header = sections[i].strip()       # e.g. "1. REFUND PROCESS"
                body = sections[i+1].strip() if i+1 < len(sections) else ""
                
                enriched_text = (
                    f"Section: {header}\n"
                    f"Content: {body}"
                )
                
                doc = Document(
                    page_content=enriched_text,
                    metadata={
                        "policy_type": policy_category,
                        "file_source": filename,
                        'document_title': document_title,
                        "section_header": header,
                        "chunk_id": i // 2, # simple counter
                        "strategy": "header_injection",
                        "last_updated": time.ctime(os.path.getmtime(file_path))
                    }
                )
                doc_chunks.append(doc)
                
        # --- STRATEGY 2: Fallback (Standard Recursive Splitting) ---
        else:
            print(f"No numbered sections found in {filename}. Using standard splitter.")
            raw_doc = Document(page_content=content, metadata={"source": filename})
            split_docs = fallback_splitter.split_documents([raw_doc])
            
            for i, doc in enumerate(split_docs):
                doc.metadata.update({
                    "policy_type": policy_category,
                    "file_source": filename,
                    "chunk_id": i,
                    "strategy": "recursive_fallback",
                    "last_updated": time.ctime(os.path.getmtime(file_path))
                })
                doc_chunks.append(doc)

        final_chunks.extend(doc_chunks)

    print(f"Processed {len(os.listdir(DOCS_FOLDER))} files into {len(final_chunks)} high-quality chunks.")
    logging.info(f"Processed {len(os.listdir(DOCS_FOLDER))} files into {len(final_chunks)} high-quality chunks.")
    
    return final_chunks
def init_vector_db(knowledge_base_path):
    embedding_manager = EmbeddingManager()
    logging.info('Initializing vector database...')
    client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
    
    docs = load_and_split_data(knowledge_base_path)
    embeddings = embedding_manager.generate_embeddings([doc.page_content for doc in docs])
    
    db = client.get_or_create_collection(
        name='drive_it_policies',
        metadata={'description':'Collection of policy documents for DriveIt RAG system'}
    )
    
    ids = [str(1000+i) for i in range(len(docs))]
    metadatas = [dict(doc.metadata) for doc in docs]
    
    logging.info(f"Adding {len(docs)} documents to the vector database...")
    db.add(
        ids=ids, 
        embeddings=embeddings.tolist(), 
        metadatas=metadatas, 
        documents=[doc.page_content for doc in docs]
        )
        
    return db, embedding_manager