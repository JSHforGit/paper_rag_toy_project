# 🧠 Private Knowledge Brain (LM Studio + RAG)

Windows 로컬에서 LM Studio와 연동하여 실행하는 RAG 기반 논문 분석 챗봇

## 📋 사전 준비

### 1. LM Studio 설치
1. https://lmstudio.ai/ 에서 다운로드
2. 설치 후 실행

### 2. 모델 다운로드 (LM Studio에서)
- 추천 모델:
  - `exaone-4.0-1.2b`


### 3. Python 환경 설정
```bash
# Python 3.9+ 필요
python --version

# 가상환경 생성 (선택사항)
python -m venv venv
venv\Scripts\activate  # Windows

# 패키지 설치
pip install -r requirements.txt
```

## 🚀 실행 방법

### Step 1: LM Studio 서버 시작
1. LM Studio 실행
2. 왼쪽 탭에서 다운로드한 모델 로드
3. 상단 'Local Server' 탭 클릭
4. **"Start Server"** 버튼 클릭
5. 주소 확인: `http://localhost:1234/v1`

### Step 2: Streamlit 앱 실행
```bash
# 터미널에서 실행
streamlit run app.py
```

### Step 3: 브라우저에서 사용
- 자동으로 브라우저가 열림 (http://localhost:8501)
- PDF 업로드 후 질문 시작!

## 🎛️ 포트 변경 방법

**LM Studio 포트 변경 시:**
```bash
# app.py 실행 전 환경변수 설정
set LM_STUDIO_URL=http://localhost:5678/v1
streamlit run app.py
```

**Streamlit 포트 변경:**
```bash
streamlit run app.py --server.port 8080
```

## 🔧 트러블슈팅

### 연결 실패 시
1. LM Studio에서 "Start Server" 확인
2. 방화벽에서 localhost 허용 확인
3. 다른 프로그램이 1234 포트 사용 중인지 확인

### 메모리 부족 시
- LM Studio에서 더 작은 모델 선택 (예: 3B 파라미터)
- `app.py`에서 `chunk_size` 줄이기 (500 → 300)

### 느린 응답 시
- GPU 사용 가능 시 LM Studio 설정에서 GPU 가속 활성화
- `max_tokens` 줄이기 (512 → 256)