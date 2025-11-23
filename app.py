import re
import os
import streamlit as st
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from core.models import (
    load_models, 
    get_downloaded_models,
    switch_model_via_sdk, 
    HAS_LMS_SDK
)
from core.loader import process_pdf


load_dotenv()
st.set_page_config(page_title="Private Knowledge Brain", page_icon="🧠")
st.title("Private Knowledge Brain")


# 사이드바 폰트 및 줄바꿈 처리
st.markdown("""
<style>
    [data-testid="stSidebar"] [data-baseweb="select"] span {
        font-size: 0.9rem !important;
        white-space: normal !important; /* 긴 이름 줄바꿈 허용 */
        line-height: 1.2 !important;
        height: auto !important;
    }
    ul[data-testid="stSelectboxVirtualDropdown"] li span {
        font-size: 0.85rem !important;
        font-family: monospace !important;
    }
</style>
""", unsafe_allow_html=True)


# 사이드바 설정
with st.sidebar:
    st.header("LM Studio 설정")
    
    # 1. URL 설정
    lm_studio_url = st.text_input(
        "LM Studio API URL", 
        value=os.getenv("LM_STUDIO_URL", "http://localhost:1234"),
        help="LM Studio의 Local Server 주소 (기본값: http://localhost:1234)"
    )

    # 2. 모델 선택 및 로드 (SDK 기능 통합)
    if HAS_LMS_SDK:
        # 1. 모델 리스트 가져오기 (딕셔너리 형태)
        raw_list = get_downloaded_models()
        
        if raw_list:
            # 2. { '화면에_보여줄_이름': '실제_경로' } 형태의 맵(Map) 생성
            # 이름이 중복될 경우를 대비해 인덱스를 살짝 붙여주거나 파일명을 괄호에 넣음
            model_map = {}
            for item in raw_list:
                label = item['label']
                path = item['path']
                
                # 키 중복 방지 (이미 같은 이름이 있으면 파일명 일부 추가)
                if label in model_map:
                    # 예: EXAONE (Q4_K_M.gguf)
                    filename = path.split('/')[-1]
                    label = f"{label} ({filename})"
                
                model_map[label] = path

            # 3. Selectbox에는 '키(이름)'만 넘겨줌 -> UI에 절대 딕셔너리가 안 뜸
            selected_label = st.selectbox(
                "사용할 모델 선택", 
                options=list(model_map.keys()), # 문자열 리스트만 전달
                index=0
            )
            
            # 4. 선택된 이름으로 실제 경로 찾기
            target_path = model_map[selected_label]
            
            # 모델 로드 버튼
            if st.button("모델 로드 및 연결", use_container_width=True):
                with st.spinner(f"'{selected_label}' 모델 로드 중..."):
                    # 실제 경로는 여기서 사용
                    ctx, err = switch_model_via_sdk(target_path)
                    
                    if err:
                        st.error(f"로드 실패: {err}")
                    else:
                        st.session_state.model_id = selected_label
                        st.session_state.detected_ctx = ctx
                        st.success("로드 완료!")
                        st.rerun()
        else:
            st.warning("다운로드된 모델을 찾을 수 없습니다.")
    else:
        st.error("SDK 미설치 (pip install lmstudio)")

    # 3. 현재 연결 정보 표시 (레퍼런스 스타일)
    current_model = st.session_state.get('model_id', 'Unknown')
    current_ctx = st.session_state.get('detected_ctx', 4096)
    
    if current_model != "Unknown":
        with st.expander("현재 모델 정보", expanded=True):
            st.markdown(f"**모델명:** `{current_model}`")
            st.markdown(f"**최대 컨텍스트:** `{current_ctx:,}` tokens")
            st.info("SDK를 통해 모델 정보를 자동으로 가져왔습니다.")

    st.markdown("---")
    st.header("모델 파라미터")

    # 4. 파라미터 설정
    temperature = st.slider(
        "Temperature", 
        0.0, 1.0, 0.1, 0.1,
        help="높을수록 창의적, 낮을수록 일관적입니다."
    )
    
    # Max Tokens 프리셋 버튼
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("짧게 (512)", use_container_width=True):
            st.session_state.preset_tokens = 512
    with col2:
        if st.button("보통 (2K)", use_container_width=True):
            st.session_state.preset_tokens = 2048
    with col3:
        if st.button("길게 (Max)", use_container_width=True):
            st.session_state.preset_tokens = min(8192, current_ctx)

    # 프리셋 적용 로직
    default_max = min(2048, current_ctx)
    if 'preset_tokens' in st.session_state:
        default_max = st.session_state.preset_tokens
        # 슬라이더 값 반영을 위해 session state 정리 (선택사항)
        del st.session_state.preset_tokens
        st.rerun()

    max_tokens = st.slider(
        "Max Tokens (출력 길이)", 
        128, 
        current_ctx, 
        default_max,
        128,
        help=f"현재 모델의 최대 컨텍스트: {current_ctx:,} tokens"
    )
    
    top_k = st.slider(
        "검색 결과 수 (Top-K)", 
        1, 10, 5, 1,
        help="문서에서 가져올 참조 청크의 개수입니다."
    )
    
    st.markdown("---")
    st.header("문서 업로드")
    uploaded_file = st.file_uploader("PDF 파일", type="pdf")

    st.markdown("---")
    
    # 5. 도움말 및 정보 (레퍼런스 내용 복원)
    with st.expander("사용 방법"):
        st.markdown("""
        **1단계: 모델 준비**
        - 위 목록에서 모델을 선택하고 **'모델 로드 및 연결'** 버튼을 누르세요.
        - LM Studio가 실행 중이어야 합니다.
        
        **2단계: 설정 확인**
        - '현재 모델 정보'에서 컨텍스트 길이가 올바른지 확인하세요.
        
        **3단계: 문서 업로드**
        - PDF 파일을 업로드하면 자동으로 분석이 시작됩니다.
        
        **4단계: 질문하기**
        - 채팅창에 논문 내용을 질문하거나 요약을 요청하세요.
        """)
    
    with st.expander("모델이 목록에 안 보이나요?"):
        st.markdown("""
        **LM Studio 확인:**
        1. LM Studio의 'My Models' 폴더에 모델이 있는지 확인하세요.
        2. LM Studio 프로그램이 실행 중인지 확인하세요.
        3. `pip install lmstudio`가 설치되어 있는지 확인하세요.
        """)
    
    st.info("Windows 로컬 환경")


llm, embeddings = load_models(lm_studio_url, temperature, max_tokens)



# ==========================================
# 메인 로직: 데이터 처리 및 채팅
# ==========================================

if "messages" not in st.session_state: st.session_state.messages = []
if "retriever" not in st.session_state: st.session_state.retriever = None
if "full_text" not in st.session_state: st.session_state.full_text = None

# PDF 처리
if uploaded_file and (st.session_state.retriever is None or st.session_state.full_text is None):
    with st.spinner("PDF 분석 중..."):
        # process_pdf는 (retriever, full_text) 두 개를 반환해야 함
        retriever, full_text = process_pdf(
            uploaded_file.getvalue(), embeddings, top_k
        )
        st.session_state.retriever = retriever
        st.session_state.full_text = full_text
    st.success("분석 완료!")

# 채팅 히스토리
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 사용자 입력
if prompt := st.chat_input("질문을 입력하세요..."):
    if not st.session_state.retriever:
        st.error("PDF를 먼저 업로드하세요")
    else:
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # ==========================================
        # [NEW] LLM Router: 의도 분류
        # ==========================================
        # 질문의 의도를 파악하는 가벼운 체인
        router_template = """Task: Classify the user's question into 'SUMMARY' or 'SEARCH'.
        
        Rules:
        1. "SUMMARY": Broad questions, summaries, overviews, main topics.
        2. "SEARCH": Specific facts, numbers, page lookup, definitions.
        3. Do NOT generate <think> tags or reasoning. 
        4. Output ONLY the class name.

        Question: {question}
        Class:"""
        
        router_chain = ChatPromptTemplate.from_template(router_template) | llm | StrOutputParser()
        
        # UI에 분류 결과 표시 (디버깅용, 원치 않으면 주석 처리)
        with st.status("질문 분석 중...", expanded=False) as status:
            raw_intent = router_chain.invoke({"question": prompt}).strip().upper()
            clean_intent = re.sub(r'<think>.*?</think>', '', raw_intent, flags=re.DOTALL).strip().upper()
        
            # 텍스트에 SUMMARY나 SEARCH가 포함되어 있는지 확인 (더 안전하게)
            intent = "SUMMARY" if "SUMMARY" in clean_intent else "SEARCH"
            status.update(label=f"질문 유형 감지: {intent}", state="complete")

        # Context 설정 (Routing 결과 적용)
        context_data = ""
        source_info = ""

        if "SUMMARY" in intent:
            # [SUMMARY 모드] 전체 텍스트 사용
            # 모델 Context Limit에서 Output 토큰과 여유분을 뺀 만큼만 입력
            safe_limit = st.session_state.detected_ctx - max_tokens - 500
            # 한글/영어 섞임 고려하여 대략 3배수로 자름 (단순화된 로직)
            char_limit = int(safe_limit * 2.5)
            
            context_data = st.session_state.full_text[:char_limit]
            source_info = f"\n\n*( 전체 문서 분석 모드 | Context: {safe_limit} tokens )*"
        else:
            # [SEARCH 모드] RAG 검색 사용
            docs = st.session_state.retriever.invoke(prompt)
            context_data = "\n\n".join([f"[Page {d.metadata.get('page','?')}] {d.page_content}" for d in docs])
            source_info = "\n\n*( 정밀 검색 모드 )*"
        
        
        
        # 답변 생성    
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
        
        chain = (
            {"context": lambda x: context_data, "question": RunnablePassthrough()}
            | ChatPromptTemplate.from_template(template) 
            | llm 
            | StrOutputParser()
        )

        with st.chat_message("assistant"):
            # 두 개의 영역 준비: 사고 과정(Expander) + 최종 답변(Main)
            reasoning_area = st.empty()
            answer_area = st.empty()
            
            # 상태 변수
            full_response = ""       # 전체 로그 저장용
            reasoning_content = ""   # 사고 과정 텍스트
            answer_content = ""      # 최종 답변 텍스트
            is_thinking = False      # 현재 사고 중인가?
            
            try:
                for chunk in chain.stream(prompt):
                    full_response += chunk

                    # [State Machine] 태그 감지 및 모드 전환
                    # 1. 사고 시작 감지 (<think>)
                    if "<think>" in chunk:
                        is_thinking = True
                        chunk = chunk.replace("<think>", "")
                        
                        # UI: 사고 과정 영역 생성
                        with reasoning_area.container():
                            with st.expander("💭 사고 과정 (Thinking Process)", expanded=True):
                                reasoning_placeholder = st.empty()
                    
                    # 2. 사고 종료 감지 (</think>)
                    if "</think>" in chunk:
                        is_thinking = False
                        chunk = chunk.replace("</think>", "")
                        
                        # UI: 사고 과정 완료 상태로 업데이트 (접힌 상태로 바꾸거나 유지)
                        with reasoning_area.container():
                            with st.expander("💭 사고 과정 (Thinking Process)", expanded=False):
                                st.markdown(reasoning_content)
                    

                    # [Display] 모드에 따른 출력 위치 결정
                    if is_thinking:
                        reasoning_content += chunk
                        # expander 내부 placeholder 업데이트
                        try:
                            reasoning_placeholder.markdown(reasoning_content + "▌")
                        except:
                            pass
                    else:
                        answer_content += chunk
                        answer_area.markdown(answer_content + "▌")

                # 스트리밍 종료 후 마무리 (커서 제거 및 출처 부착)
                answer_area.markdown(answer_content + source_info)
                
                # 사고 과정이 있었던 경우, 최종적으로 깔끔하게 렌더링
                if reasoning_content:
                    reasoning_area.empty() # 기존 placeholder 제거
                    with reasoning_area.container():
                        with st.expander("💭 사고 과정 (Thinking Process)", expanded=False):
                            st.markdown(reasoning_content)

            except Exception as e:
                st.error(f"Error: {e}")
        
        # 히스토리에는 '최종 답변'만 저장할지, '사고 과정'도 포함할지 결정
        # 보통은 깔끔하게 최종 답변만 저장하거나, 포맷팅해서 저장함
        final_save_content = answer_content + source_info
        
        # (선택사항) 히스토리에서도 사고 과정을 보고 싶다면 아래 주석 해제
        # if reasoning_content:
        #     final_save_content = f"<details><summary>사고 과정</summary>{reasoning_content}</details>\n\n" + final_save_content
            
        st.session_state.messages.append({"role": "assistant", "content": final_save_content})

# 채팅 초기화
if st.session_state.messages:
    if st.sidebar.button("채팅 기록 초기화"):
        st.session_state.messages = []
        st.rerun()