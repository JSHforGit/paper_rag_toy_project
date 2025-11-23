import streamlit as st
import os
import re
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ==========================================
# 1. 설정 및 캐싱 (성능 최적화)
# ==========================================
st.set_page_config(page_title="Private Knowledge Brain", page_icon="🧠")
st.title("🧠 Private Knowledge Brain (LM Studio)")

# 사이드바 설정
with st.sidebar:
    st.header("⚙️ LM Studio 설정")
    lm_studio_url = st.text_input(
        "LM Studio API URL", 
        value="http://localhost:1234/v1",
        help="LM Studio의 Local Server 주소 (기본값: http://localhost:1234/v1)"
    )
    
    # 고급 설정
    with st.expander("🎛️ 모델 파라미터"):
        temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.1)
        max_tokens = st.slider("Max Tokens", 128, 2048, 512, 128)
        top_k = st.slider("Top-K (검색 결과 수)", 1, 10, 5, 1)
    
    # LM Studio 연결 상태 확인
    if st.button("🔌 연결 테스트"):
        try:
            import requests
            response = requests.get(
                f"{lm_studio_url.replace('/v1', '')}/v1/models", 
                timeout=3
            )
            if response.status_code == 200:
                models = response.json()
                model_list = models.get('data', [])
                st.success(f"✅ 연결 성공!")
                if model_list:
                    st.info(f"📦 로드된 모델:\n{model_list[0].get('id', 'Unknown')}")
                else:
                    st.warning("⚠️ 로드된 모델이 없습니다. LM Studio에서 모델을 로드하세요.")
            else:
                st.error("❌ 연결 실패")
        except Exception as e:
            st.error(f"❌ 연결 불가: {str(e)}")
            st.info("💡 LM Studio에서 'Start Server'를 눌렀는지 확인하세요.")
    
    st.markdown("---")
    st.header("📄 Upload Document")
    uploaded_file = st.file_uploader("PDF 파일을 올려주세요", type="pdf")
    st.markdown("---")
    
    # 사용 가이드
    with st.expander("📖 사용 방법"):
        st.markdown("""
        **1단계: LM Studio 준비**
        - LM Studio 실행
        - 모델 다운로드 및 로드
        - 'Local Server' 탭에서 'Start Server' 클릭
        
        **2단계: 문서 업로드**
        - 왼쪽에서 PDF 파일 업로드
        
        **3단계: 질문하기**
        - 아래 채팅창에서 질문 입력
        """)
    
    st.info("💻 Windows 로컬 환경에서 실행 중")

# 모델 로드 (캐싱하여 매번 로딩하지 않도록 함)
@st.cache_resource
def load_llm_and_embeddings(_lm_studio_url, _temperature, _max_tokens):
    """LLM 및 임베딩 모델 로드"""
    # 임베딩 모델 (로컬에서 실행)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},  # Windows에서 안정적인 CPU 사용
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # LM Studio LLM (OpenAI 호환)
    llm = ChatOpenAI(
        base_url=_lm_studio_url,
        api_key="lm-studio",  # LM Studio는 dummy key 사용
        temperature=_temperature,
        max_tokens=_max_tokens,
        streaming=True  # 스트리밍 응답
    )
    return llm, embeddings

llm, embeddings = load_llm_and_embeddings(lm_studio_url, temperature, max_tokens)

# ==========================================
# 2. 데이터 처리 로직
# ==========================================
def is_garbage(text):
    """노이즈 텍스트 필터링"""
    if len(text) < 100: 
        return True
    num_count = len(re.findall(r'\d', text))
    if num_count / len(text) > 0.2: 
        return True
    return False

@st.cache_data
def process_pdf(file_bytes, _embeddings, _top_k):
    """PDF 처리 및 검색기 생성"""
    # 임시 파일로 저장
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(file_bytes)
        tmp_path = tmp_file.name

    # PDF 로드
    loader = PyPDFLoader(tmp_path)
    pages = loader.load()
    
    # 텍스트 분할
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, 
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
    )
    raw_splits = text_splitter.split_documents(pages)
    
    # 정제 (Garbage Collection)
    clean_splits = [doc for doc in raw_splits if not is_garbage(doc.page_content)]
    
    st.info(f"📊 총 {len(clean_splits)}개의 텍스트 청크 생성됨")
    
    # 벡터 DB & 검색기 생성
    vectorstore = Chroma.from_documents(
        documents=clean_splits, 
        embedding=_embeddings,
        persist_directory=None  # 메모리에만 저장 (빠른 처리)
    )
    chroma_retriever = vectorstore.as_retriever(search_kwargs={"k": _top_k})
    
    bm25_retriever = BM25Retriever.from_documents(clean_splits)
    bm25_retriever.k = _top_k
    
    # 하이브리드 검색
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, chroma_retriever],
        weights=[0.3, 0.7]  # Semantic search에 더 높은 가중치
    )
    
    # 임시 파일 삭제
    os.unlink(tmp_path)
    
    return ensemble_retriever

# ==========================================
# 3. UI 및 채팅 로직
# ==========================================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "retriever" not in st.session_state:
    st.session_state.retriever = None

# PDF 업로드 처리
if uploaded_file and st.session_state.retriever is None:
    with st.spinner("📄 PDF를 분석하고 인덱싱하는 중... (잠시만 기다려주세요)"):
        file_bytes = uploaded_file.getvalue()
        st.session_state.retriever = process_pdf(file_bytes, embeddings, top_k)
    st.success("✅ 분석 완료! 질문을 입력하세요.")
    st.balloons()

# 채팅 기록 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 사용자 입력 처리
if prompt := st.chat_input("논문에 대해 질문하세요..."):
    if st.session_state.retriever is None:
        st.error("❌ 먼저 PDF 파일을 업로드해주세요.")
    else:
        # 사용자 메시지 표시
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # RAG 파이프라인
        def format_docs(docs):
            """검색된 문서 포맷팅"""
            return "\n\n".join([
                f"[출처: Page {d.metadata.get('page', '?')}]\n{d.page_content}" 
                for d in docs
            ])

        # 프롬프트 템플릿
        template = """당신은 논문 분석 전문가입니다. 주어진 문맥을 기반으로 질문에 답변하세요.

[제약 조건]
1. 제공된 문맥(Context)만을 사용하여 답변하세요.
2. 전문 용어는 영어 원문을 유지하세요 (예: 'Diffusion Model', 'Attention Mechanism').
3. 답변은 한국어로 작성하되, 필요시 영어 용어를 병기하세요.
4. 문맥에서 답을 찾을 수 없다면 "제공된 문서에서 해당 내용을 찾을 수 없습니다"라고 답하세요.
5. 답변 마지막에 참고한 페이지 번호를 명시하세요.

[문맥]:
{context}

[질문]: {question}

[답변]:"""
        
        prompt_template = ChatPromptTemplate.from_template(template)
        
        # RAG Chain 구성
        rag_chain = (
            {
                "context": st.session_state.retriever | format_docs, 
                "question": RunnablePassthrough()
            }
            | prompt_template
            | llm
            | StrOutputParser()
        )
        
        # 스트리밍 응답
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""
            
            try:
                for chunk in rag_chain.stream(prompt):
                    full_response += chunk
                    response_placeholder.markdown(full_response + "▌")
                
                response_placeholder.markdown(full_response)
                
            except Exception as e:
                error_msg = f"❌ 에러 발생: {str(e)}"
                st.error(error_msg)
                
                # 디버깅 정보
                with st.expander("🔍 에러 상세 정보"):
                    st.code(str(e))
                    st.markdown("""
                    **해결 방법:**
                    1. LM Studio에서 모델이 로드되어 있는지 확인
                    2. LM Studio의 'Local Server'가 실행 중인지 확인
                    3. 왼쪽 사이드바에서 '🔌 연결 테스트' 버튼 클릭
                    4. 포트 번호가 맞는지 확인 (기본값: 1234)
                    """)
                
                full_response = "죄송합니다. 응답 생성 중 오류가 발생했습니다."
        
        st.session_state.messages.append({
            "role": "assistant", 
            "content": full_response
        })

# 채팅 초기화 버튼
if st.session_state.messages:
    if st.sidebar.button("🗑️ 채팅 기록 초기화"):
        st.session_state.messages = []
        st.rerun()