import tempfile
import os
import re
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever 

def is_garbage(text):
    if len(text) < 100: return True
    if len(re.findall(r'\d', text)) / len(text) > 0.2: return True
    return False

@st.cache_resource
def process_pdf(file_bytes, _embeddings, _top_k):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    loader = PyPDFLoader(tmp_path)
    pages = loader.load()
    
    # 요약 모드를 위해 전체 텍스트 추출
    full_text = "\n\n".join([p.page_content for p in pages])
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, 
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
    )
    raw_splits = splitter.split_documents(pages)
    clean_splits = [d for d in raw_splits if not is_garbage(d.page_content)]
    
    vectorstore = Chroma.from_documents(documents=clean_splits, embedding=_embeddings)
    chroma_retriever = vectorstore.as_retriever(search_kwargs={"k": _top_k})
    
    bm25_retriever = BM25Retriever.from_documents(clean_splits)
    bm25_retriever.k = _top_k
    
    ensemble = EnsembleRetriever(
        retrievers=[bm25_retriever, chroma_retriever],
        weights=[0.3, 0.7]
    )
    
    os.unlink(tmp_path)
    
    # 리트리버와 전체 텍스트를 함께 반환
    return ensemble, full_text