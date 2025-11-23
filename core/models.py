import re
import requests
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
import streamlit as st

try:
    import lmstudio as lms
    HAS_LMS_SDK = True
except ImportError:
    HAS_LMS_SDK = False
    
def get_downloaded_models():
    """
    [Core] LM Studio에 다운로드된 모델 목록을 파싱하여 반환
    Returns: [{'path': '실제경로', 'label': '표시이름'}, ...]
    """
    if not HAS_LMS_SDK:
        return []
    
    try:
        # 1. 모델 객체 리스트 가져오기
        raw_models = lms.list_downloaded_models("llm")
        parsed_models = []

        for m in raw_models:
            m_str = str(m)
            m_path = getattr(m, 'path', m_str) # 실제 로드에 필요한 ID
            
            # 2. display_name 추출 시도 (정규표현식 사용)
            # 예: DownloadedLlm(..., display_name='Meta-Llama-3', ...)
            match = re.search(r"display_name='([^']+)'", m_str)
            
            if match:
                # 괄호 안의 깔끔한 이름 추출
                label = match.group(1)
            else:
                # 실패 시 경로에서 앞부분(publisher) 제거하고 뒷부분만 사용
                label = m_path.split('/')[-1] if '/' in m_path else m_path

            parsed_models.append({
                "path": m_path,  # 로드할 때 쓸 ID
                "label": label   # UI에 보여줄 이름
            })
            
        return parsed_models

    except Exception as e:
        print(f"모델 목록 파싱 실패: {e}")
        return []

def switch_model_via_sdk(model_path):
    """
    선택한 모델을 LM Studio 서버에 로드(Load) 시킴
    """
    if not HAS_LMS_SDK:
        return None, "SDK 미설치"
        
    try:
        # SDK를 통해 해당 모델을 지정하여 로드 시도
        # lms.llm(path)를 호출하면 LM Studio가 해당 모델을 준비합니다.
        # 주의: 구체적인 context_length 설정은 LM Studio의 'Preset' 설정을 따릅니다.
        model = lms.llm(model_path)
        context_length = model.get_context_length()
        # 모델이 실제로 로드되었는지 확인하기 위해 간단한 속성 조회
        ctx = context_length
        
        return int(ctx) if ctx else 4096, None
    except Exception as e:
        return None, str(e)    

# 모델 로드
@st.cache_resource
def load_models(_lm_studio_url, _temperature, _max_tokens):
    """임베딩 및 LLM 로드"""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    base_url = _lm_studio_url
    if not base_url.endswith('/v1'):
        base_url = base_url.rstrip('/') + '/v1'
    
    llm = ChatOpenAI(
        base_url=base_url,
        api_key="lm-studio",
        temperature=_temperature,
        max_tokens=_max_tokens,
        streaming=True
    )
    return llm, embeddings