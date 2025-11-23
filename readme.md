# 🧠 Private Knowledge Brain

> **완전 로컬 환경에서 실행되는 RAG 기반 논문 분석 챗봇**  
> LM Studio + Streamlit 기반으로 개인정보 유출 걱정 없이 논문을 분석하고 질문할 수 있습니다.

---

##  주요 특징

- **완전한 프라이버시**: 모든 처리가 로컬에서 실행 (로컬에 모델만 있다면 인터넷 연결 불필요)
- **자동 의도 분류**: 질문 유형에 따라 전체 문서 분석 또는 부분 검색 자동 선택
- **스마트 문서 분할**: 챕터 기반 Semantic Chunking으로 정확한 검색
- **하이브리드 검색**: BM25(키워드) + Chroma(벡터) 앙상블 방식
- **출처 추적**: 모든 답변에 페이지 번호 자동 표기
- **SDK 통합**: LM Studio SDK로 모델 자동 감지 및 간편 전환

---

## 📋 시스템 요구사항

### 필수 사항
- **OS**: Windows 10/11 (Linux/macOS도 가능하나 테스트 안 됨)
- **Python**: 3.12

---

## 설치 가이드

### Step 1: LM Studio 설치 및 설정

#### 1.1 다운로드 및 설치
```
1. https://lmstudio.ai/ 접속
2. Windows용 설치 파일 다운로드 (.exe)
3. 설치 후 실행
```

#### 1.2 모델 다운로드
LM Studio 실행 후:
```
1. 왼쪽 'Discover' 탭 클릭
2. 검색창에 모델 이름 입력
3. 원하는 모델 선택 후 'Download' 클릭
```

> **💡 팁**: 처음에는 2B~3B 모델로 시작해서 작동을 확인한 후, 더 큰 모델을 시도하세요.

#### 1.3 서버 설정 (중요!)
```
1. LM Studio 상단 'Local Server' 탭 클릭
2. 다운로드한 모델을 선택하여 로드
3. Server Options에서 다음 확인:
   - Port: 1234 (기본값)
   - Context Length: 4096 이상 (모델이 지원하는 최대값 권장)
   - GPU Offload: 100% (GPU 있을 경우)
4. "Start Server" 버튼 클릭
5. 초록색 "Server Running" 표시 확인
```

---

### Step 2: Python 환경 설정

#### 2.1 가상환경 생성 및 설치 (권장)

**방법 1: Conda 사용 (권장)**
```bash
# Conda가 설치되어 있는 경우
cd [프로젝트 폴더 경로]

# environment.yml로 환경 생성 (Python 3.12 + 모든 패키지 자동 설치)
conda env create -f environment.yml

# 환경 활성화
conda activate paper_rag

# 설치 확인
python --version  # Python 3.12.x 출력되어야 함
```

**방법 2: Python venv 사용**
```bash
# CMD 또는 PowerShell에서 실행
cd [프로젝트 폴더 경로]

# Python 버전 확인 (3.12 이상 필요)
python --version

# 가상환경 생성
python -m venv venv

# 가상환경 활성화
# Windows (CMD)
venv\Scripts\activate

# Windows (PowerShell)
venv\Scripts\Activate.ps1

# 활성화 확인 (프롬프트 앞에 (venv) 표시됨)

# 패키지 설치
pip install -r requirements.txt
```

**설치 중 오류 발생 시**:
```bash
# Conda 환경 재생성
conda env remove -n paper_rag
conda env create -f environment.yml

# 또는 pip 사용 시 업그레이드 후 재시도
python -m pip install --upgrade pip
pip install -r requirements.txt

# 개별 패키지 설치가 필요한 경우
pip install streamlit langchain langchain-openai langchain-community
pip install pypdf chromadb sentence-transformers huggingface-hub
pip install lmstudio rank_bm25 python-dotenv langchain-huggingface langchain-classic
```

> **💡 추천**: Conda를 사용하면 Python 버전과 모든 의존성이 자동으로 설치되어 편리합니다.  
> Conda가 없다면 [Miniconda](https://docs.conda.io/en/latest/miniconda.html) 또는 [Anaconda](https://www.anaconda.com/download)를 먼저 설치하세요.

---

## 📖 사용 방법

### 기본 실행

```bash
# 1. 가상환경 활성화 (위에서 생성한 경우)
venv\Scripts\activate

# 2. Streamlit 앱 실행
streamlit run app.py

# 3. 브라우저 자동 오픈 (http://localhost:8501)
```

### 단계별 사용 가이드

#### 1단계: 모델 로드
```
[사이드바]
1. "사용할 모델 선택" 드롭다운에서 원하는 모델 선택
2. "모델 로드 및 연결" 버튼 클릭
3. "로드 완료!" 메시지 확인
4. "현재 모델 정보"에서 컨텍스트 길이 확인
```

**주의**: 모델이 목록에 안 보이면:
- LM Studio의 'My Models' 폴더에 모델이 있는지 확인
- LM Studio 프로그램이 실행 중인지 확인
- `pip install lmstudio` 설치 여부 확인

#### 2단계: 파라미터 조정
```
[사이드바 - 모델 파라미터]
- Max Tokens: 적절한 답변 길이로 설정
- Top-K: 5 (검색 결과 개수)
```

#### 3단계: 문서 업로드
```
[사이드바 - 문서 업로드]
1. "Browse files" 클릭
2. PDF 논문 선택 (최대 50MB 권장)
3. 자동 분석 시작 (진행 상태 표시)
4. "분석 완료!" 메시지 확인
```

#### 4단계: 질문하기
```
[메인 화면 - 채팅]
1. 하단 입력창에 질문 입력
2. Enter 또는 전송 버튼 클릭
3. "질문 유형 감지" 자동 표시
   - SUMMARY: 전체 문서 기반 답변
   - SEARCH: 부분 검색 기반 답변
4. 답변 확인 (페이지 번호 포함)
```

---

##  스마트 질문 가이드

### 전체 문서 분석 (SUMMARY 모드)
다음과 같은 질문은 **전체 문서**를 기반으로 답변합니다:
```
✅ "이 논문의 주요 contribution은 무엇인가요?"
✅ "전체적인 방법론을 요약해주세요"
✅ "Introduction부터 Conclusion까지 흐름을 설명해주세요"
✅ "이 연구의 한계점은 무엇인가요?"
```

### 부분 검색 (SEARCH 모드)
다음과 같은 질문은 **관련 섹션만 검색**하여 답변합니다:
```
✅ "Table 3의 결과를 설명해주세요"
✅ "Attention Mechanism은 어떻게 구현했나요?"
✅ "실험 환경(데이터셋, 하이퍼파라미터)은?"
✅ "BLEU 점수가 가장 높은 모델은?"
```

### 효과적인 질문 팁
```
❌ "이 논문 뭐에 관한 거야?"  
✅ "이 논문의 핵심 연구 주제와 목표를 설명해주세요"

❌ "결과가 어때?"  
✅ "Table 2의 실험 결과를 다른 baseline 모델과 비교해서 설명해주세요"

❌ "방법 알려줘"  
✅ "제안된 모델의 아키텍처를 단계별로 설명해주세요"
```

---

## ⚙️ 고급 설정

### 환경 변수 설정 (.env 파일)
프로젝트 루트에 `.env` 파일 생성:
```env
# LM Studio 서버 주소 (기본값: http://127.0.0.1:1234)
LM_STUDIO_URL=http://127.0.0.1:1234
```

### Streamlit 커스터마이징
```bash
# 포트 변경
streamlit run app.py --server.port 8080

# 브라우저 자동 열기 비활성화
streamlit run app.py --server.headless true

# 테마 변경 (.streamlit/config.toml 생성)
[theme]
primaryColor = "#FF4B4B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
```

### 문서 분할 로직 조정 (core/loader.py)
```python
# 더 긴 청크 (메모리 여유 있을 때)
chunk_size=9192  # 기본값: 4096
chunk_overlap=300  # 기본값: 200

# 더 짧은 청크 (정확도 우선)
chunk_size=2048
chunk_overlap=100
```

---

## 문제 해결

### 1. 연결 오류: "Connection refused" 또는 "404 Not Found"

**증상**: 
```
Error: Could not connect to LM Studio
httpx.ConnectError: [Errno 10061]
```

**해결 방법**:
```
1. LM Studio에서 "Start Server" 버튼이 눌러져 있는지 확인
2. 주소 확인: http://localhost:1234 (포트 번호 주의)
3. 방화벽 설정:
   - Windows Defender 방화벽 → 고급 설정
   - 인바운드 규칙 → 새 규칙 → 포트 → 1234 허용
4. 다른 프로그램이 1234 포트 사용 중인지 확인:
   netstat -ano | findstr :1234
```

### 2. 메모리 부족 오류: "CUDA out of memory" 또는 "MemoryError"

**해결 방법**:
```
[LM Studio 설정]
1. 더 작은 모델 사용 (7B → 3B → 2B)
2. Context Length 줄이기 (8192 → 4096 → 2048)
3. GPU Offload 비율 낮추기 (100% → 50%)

[앱 설정]
1. Top-K 줄이기 (5 → 3)
2. PDF 크기 줄이기 (summary의 경우)
```

### 3. 느린 응답 속도

**CPU 사용 시** (GPU 없음):
```
- 2B~3B 모델 사용 (7B 이상은 느림)
- Max Tokens 512 이하로 제한
- Context Length 2048로 설정
```

**GPU 사용 시**:
```
1. LM Studio에서 GPU Offload 100% 설정
2. CUDA 드라이버 최신 버전 확인:
   nvidia-smi
3. GPU 메모리 확인 (최소 4GB VRAM 권장)
```

### 4. 모델이 목록에 안 나타남

**확인 사항**:
```
1. LM Studio의 'My Models' 탭에서 다운로드 완료 확인
2. SDK 설치 확인:
   pip show lmstudio
3. LM Studio 재시작
4. Python 스크립트 재실행
```

### 5. PDF 분석 실패

**일반적인 원인**:
```
❌ 스캔된 이미지 PDF (텍스트 추출 불가)
❌ 암호화된 PDF
❌ 손상된 파일
❌ 너무 큰 파일 (>100MB)

✅ 해결: 
   - PDF를 OCR 처리 후 재저장
   - 암호 해제 후 업로드
   - 파일을 여러 부분으로 나누기
```

---

## 주의사항 및 제한사항

### 지원되는 형식
- ✅ **PDF**: 텍스트 기반 PDF (학술 논문)
- ❌ **이미지 PDF**: 스캔본은 OCR 전처리 필요
- ❌ **Word/PPT**: 현재 미지원 (PDF 변환 후 사용)

### 언어 지원
- **영어 논문**: 모든 모델 잘 작동
- **한국어 논문**: EXAONE, Qwen 모델 권장
- **다국어**: Llama-3.2 이상 권장

### 성능 한계
```
- 매우 긴 논문 (100페이지 이상): 처리 시간 5분 이상 소요
- 복잡한 수식: LaTeX 수식 인식 제한적
- 표와 그래프: 텍스트로만 해석 (이미지 분석 불가)
- Context Window: 모델의 최대 길이 초과 시 답변 품질 저하
```

### 보안 및 프라이버시
- **완전 로컬 실행**: 외부 서버 전송 없음
- **데이터 저장**: 메모리에만 임시 저장 (앱 종료 시 삭제)
- **주의**: `.env` 파일에 API 키 등 민감 정보 넣지 말 


---

## 기술 스택

| 항목 | 기술 | 역할 |
|------|------|------|
| **LLM** | LM Studio | 로컬 추론 엔진 |
| **Frontend** | Streamlit | 웹 UI |
| **Embeddings** | HuggingFace Transformers | 문서 벡터화 |
| **Vector DB** | ChromaDB | 벡터 저장/검색 |
| **Keyword Search** | BM25 | 키워드 검색 |
| **PDF Parser** | PyPDF | 텍스트 추출 |
| **Orchestration** | LangChain | RAG 파이프라인 |

---

## 🔄 업데이트 예상?(안할수도?)

- [ ] 이미지 PDF OCR 지원 (PaddleOCR 통합)
- [ ] 다중 문서 비교 기능
- [ ] 대화 히스토리 저장/불러오기
- [ ] 한국어 전용 Embeddings 적용
- [ ] Docker 컨테이너 배포 지원

---

## 지원 및 피드백

**문제 발생 시**:
1. 위 '문제 해결' 섹션 참조
2. LM Studio 공식 문서: https://lmstudio.ai/docs
3. LangChain 문서: https://python.langchain.com/


---

## 라이선스

이 프로젝트는 교육 및 연구 목적으로 자유롭게 사용 가능합니다.  
상업적 사용 시 각 라이브러리의 라이선스를 확인하세요.

---