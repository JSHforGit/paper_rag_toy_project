import tempfile
import os
import re
import streamlit as st
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

def is_garbage(text):
    if len(text) < 100: return True
    # 숫자가 20% 이상이면 쓰레기 데이터(표, 참고문헌 등)로 간주
    if len(re.findall(r'\d', text)) / len(text) > 0.2: return True
    return False

def get_pages_for_range(start_idx, end_idx, page_map):
    """
    텍스트의 시작(start_idx)과 끝(end_idx) 좌표를 받아, 
    해당 구간이 포함된 실제 PDF 페이지 번호들을 리스트로 반환합니다.
    """
    pages = set()
    for p_start, p_end, p_num in page_map:
        # 구간이 겹치는지 확인 (Overlap logic)
        # 텍스트 조각의 시작이 페이지 끝보다 앞이고, 텍스트 조각의 끝이 페이지 시작보다 뒤면 겹침
        if start_idx < p_end and end_idx > p_start:
            pages.add(p_num)
    return sorted(list(pages))

def split_text_by_chapters(full_text, page_map):
    """
    가설:
    "이전 줄이 공백이고, 현재 줄이 숫자로 시작하면 챕터의 시작이다."
    + 페이지 번호 추적 기능 포함
    """
    lines = full_text.split('\n')
    chunks = []
    current_chunk_lines = []
    
    # 커서(Cursor): 현재 처리 중인 라인이 전체 텍스트에서 몇 번째 글자인지 추적
    cursor = 0 
    chunk_start_cursor = 0 
    
    # 첫 번째 덩어리의 제목 (Introduction 이전의 Abstract 등)
    current_chapter_title = "Abstract/Intro" 
    
    for i, line in enumerate(lines):
        line_len = len(line) + 1 
        line = line.strip()
        
        # ---------------------------------------------------------
        # 챕터 감지 조건
        # 1. i > 0: 첫 줄은 제외
        # 2. lines[i-1].strip() == "": 바로 윗줄이 빈 줄이어야 함
        # 3. re.match(r'^\d', line): 현재 줄이 숫자로 시작해야 함 (예: 1. Introduction)
        # 4. 숫자 뒤에 점(.)이나 공백이 와야함 (단순 숫자인주석 방지)
        # ---------------------------------------------------------
        prev_is_empty = (i > 0) and (lines[i-1].strip() == "")
        is_chapter_start = re.match(r'^\d+(\.\d+)*\.?\s+', line)
        
        if prev_is_empty and is_chapter_start:
            # 이전 챕터가 존재하면 저장
            if current_chunk_lines:
                chunk_end_cursor = cursor
                
                # 페이지 번호 복원
                detected_pages = get_pages_for_range(chunk_start_cursor, chunk_end_cursor, page_map)
                
                pages_str = ", ".join(map(str, detected_pages))
                
                joined_content = "\n".join(current_chunk_lines).strip()
                
                if joined_content and not is_garbage(joined_content):
                    chunks.append(Document(
                        page_content=joined_content,
                        metadata={
                            "chapter": current_chapter_title,
                            "pages": pages_str 
                        }
                    ))
            
            # 새로운 챕터 시작
            current_chunk_lines = [line]
            current_chapter_title = line # 현재 줄(예: "2. Methods")을 메타데이터 제목으로 사용
            chunk_start_cursor = cursor
        else:
            # 일반 텍스트는 현재 챕터에 계속 추가
            current_chunk_lines.append(line)
        
        cursor += line_len
            
    # 마지막 남은 덩어리 저장
    if current_chunk_lines:
        chunk_end_cursor = cursor
        detected_pages = get_pages_for_range(chunk_start_cursor, chunk_end_cursor, page_map)
        
        pages_str = ", ".join(map(str, detected_pages))
        
        joined_content = "\n".join(current_chunk_lines).strip()
        
        if joined_content and not is_garbage(joined_content):
            chunks.append(Document(
                page_content=joined_content,
                metadata={
                    "chapter": current_chapter_title,
                    "pages": pages_str
                }
            ))
            
    return chunks


# --------------------------------------------------------------------------
# Main Process Function
# --------------------------------------------------------------------------
@st.cache_resource
def process_pdf(file_bytes, _embeddings, _top_k):
    # 1. 임시 파일 생성
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        loader = PyPDFLoader(tmp_path)
        pages = loader.load()
        
        # 1. 전체 텍스트 추출 (페이지 구분 없이 하나로 합침)
        
        # 2. 전체 텍스트 병합 & [페이지 좌표 지도] 생성
        # 페이지 정보를 잃지 않기 위해, 텍스트를 합치면서 좌표(start, end)를 기록합니다.
        full_text = ""
        page_map = [] # [(start, end, page_num), ...]
        current_idx = 0
        
        for p in pages:
                content = p.page_content
                # PyPDFLoader의 page 메타데이터는 0부터 시작하므로 +1 보정
                p_num = p.metadata.get("page", 0) + 1 
                
                start = current_idx
                end = current_idx + len(content)
                
                page_map.append((start, end, p_num))
                
                full_text += content + "\n" # 페이지 간 줄바꿈 추가
                current_idx = end + 1 
            

        # 3. [1차 분할] 챕터(Semantic) 단위 분할
        semantic_chapters = split_text_by_chapters(full_text, page_map)
        
        # 4. [2차 분할] Fallback: 너무 긴 챕터는 기존 방식으로 쪼개기
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=4096,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        
        final_splits = []
        
        for doc in semantic_chapters:

            if len(doc.page_content) > 4096:
                sub_splits = text_splitter.split_documents([doc])
                final_splits.extend(sub_splits)
            else:
                final_splits.append(doc)
        
        # 5. 가비지 데이터 제거
        clean_splits = [d for d in final_splits if not is_garbage(d.page_content)]
        
        # 모든 텍스트가 걸러졌을 경우 예외 처리
        if not clean_splits:
                st.error("유효한 텍스트를 추출하지 못했습니다.")
                return None, full_text
        
        # 6. 리트리버 생성
        vectorstore = Chroma.from_documents(documents=clean_splits, embedding=_embeddings)
        chroma_retriever = vectorstore.as_retriever(search_kwargs={"k": _top_k})
        
        bm25_retriever = BM25Retriever.from_documents(clean_splits)
        bm25_retriever.k = _top_k
        
        ensemble = EnsembleRetriever(
            retrievers=[bm25_retriever, chroma_retriever],
            weights=[0.3, 0.7]
        )
        
        os.unlink(tmp_path)
        
        return ensemble, full_text
    # 항상 파일 삭제 실행
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)