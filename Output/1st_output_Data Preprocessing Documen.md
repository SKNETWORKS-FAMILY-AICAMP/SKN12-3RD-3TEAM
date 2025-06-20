# AI 기반 코드 분석 파이프라인 상세 설계

## 1. 개요

본 문서는 GitHub 저장소의 소스 코드를 분석하고, 이에 대한 질의응답 및 코드 수정을 지원하는 AI 기반 코드 분석 파이프라인의 상세 설계를 기술합니다. 파이프라인은 데이터 수집, 전처리, 코드 청킹, 메타데이터 추출, 저장, 그리고 품질 관리에 이르는 전 과정을 포함합니다.

## 2. 데이터 수집 (Data Collection)

### 2.1. 수집 대상
- **소스**: 공개 GitHub 저장소
- **데이터 유형**:
    - 소스 코드 파일 (주요 프로그래밍 언어 대상)
    - 저장소 메타데이터 (이름, 설명, 언어 분포 등)
    - 커밋 히스토리 (변경 사항 추적에 활용 가능)
    - 이슈 및 PR 데이터 (개발 컨텍스트 파악에 활용 가능)

### 2.2. 수집 방법
- GitHub API (PyGithub 라이브러리 활용) 를 사용하여 저장소의 콘텐츠 및 메타데이터를 프로그래밍 방식으로 접근 및 다운로드합니다.
- 특정 저장소 URL을 입력받아 해당 저장소 전체를 클론하거나, API를 통해 파일 목록 및 내용을 가져옵니다.

### 2.3. 수집 주기 및 전략
- 사용자의 요청 시점에 실시간으로 수집합니다.
- 향후 주기적인 업데이트나 웹훅을 통한 자동 동기화 기능을 고려할 수 있습니다.

## 3. 데이터 전처리 (Data Preprocessing)

### 3.1. 파일 필터링
- **포함 대상**: 주요 프로그래밍 언어 파일 (예: Python, JavaScript, Java, C++, C#, Go, Ruby, PHP, Swift, Kotlin, TypeScript 등). `pygments` 라이브러리 등을 활용하여 언어 감지 및 지원 언어 목록 기반 필터링.
- **제외 대상**:
    - 바이너리 파일 (이미지, 실행 파일, 라이브러리 파일 등)
    - 특정 설정 파일 (예: `.git`, `.idea`, `node_modules`, `venv` 등)
    - 문서 파일 (Markdown, TXT 등은 별도 처리 또는 내용 분석에 포함)
    - 대용량 파일 (일정 크기 이상, 예: 1MB 초과)
- **처리**: UTF-8 인코딩으로 통일. 인코딩 오류 발생 시 예외 처리 및 로깅.

### 3.2. 기본 코드 정리
- 불필요한 공백, 주석 (옵션에 따라 제거 또는 유지) 등을 정리하여 분석 효율을 높입니다.
- 라이선스 정보, 자동 생성 코드 등의 특정 패턴을 식별하고 필요시 제외합니다.

## 4. 코드 청킹 (Code Chunking)

### 4.1. 청킹 전략
본 시스템은 계층적이고 의미론적인 코드 청킹 전략을 채택하여 LLM이 코드의 구조와 컨텍스트를 효과적으로 이해하도록 지원합니다.

- **1단계: AST (Abstract Syntax Tree) 기반 의미론적 청킹**
    - `tree-sitter`와 같은 라이브러리를 활용하여 소스 코드를 파싱하고 AST를 생성합니다.
    - AST를 순회하며 주요 구조적 단위(클래스, 함수, 메서드, 인터페이스, 주요 제어문 블록 등)를 식별하고, 이를 기준으로 1차 청크를 생성합니다.
    - 각 청크는 가능한 독립적으로 이해될 수 있는 논리적 단위를 형성하도록 합니다.
    - **장점**: 코드의 논리적 구조를 보존하여 컨텍스트 이해도를 높이고, 정확한 참조 및 수정 지점 식별에 유리합니다.

- **2단계: 토큰 기반 크기 조절 청킹 (Fallback 및 세분화)**
    - AST 기반 청크가 설정된 최대 토큰 크기 (예: 256 토큰, OpenAI `text-embedding-3-small` 모델의 컨텍스트 창 고려)를 초과하는 경우, 해당 청크를 더 작은 하위 청크로 분할합니다.
    - 분할 시에는 문장이나 논리적 블록의 경계를 최대한 존중하며, `tiktoken` 라이브러리를 사용하여 토큰 수를 계산합니다.
    - **중첩 (Overlap)**: 청크 간의 연속성을 유지하고 컨텍스트 손실을 최소화하기 위해 약 64 토큰 정도의 중첩을 허용합니다.
    - AST 분석이 어렵거나 지원되지 않는 파일 형식의 경우, 처음부터 토큰 기반 청킹(고정 크기 또는 문장 단위 분할 후 크기 조절)을 적용합니다.

### 4.2. 청킹 파라미터
- **최대 토큰 수 (Max Tokens)**: 256 (조정 가능)
- **중첩 토큰 수 (Overlap Tokens)**: 64 (조정 가능)
- **청킹 단위 우선순위**: 1. 클래스/모듈 2. 함수/메서드 3. 논리적 코드 블록 4. 고정 토큰 크기

### 4.3. 코드 청킹의 장점
- **컨텍스트 정확도 향상**: LLM이 특정 코드 조각의 정확한 컨텍스트 내에서 분석을 수행하도록 지원합니다.
- **효율적인 검색**: 관련성 높은 코드 청크를 빠르게 식별하여 검색 시간을 단축합니다.
- **메모리 관리 용이**: 대용량 파일을 작은 단위로 나누어 처리함으로써 메모리 사용량을 최적화합니다.
- **정밀한 수정 지원**: 코드 수정 시 변경 범위를 특정 청크로 국한하여 안전하고 정확한 수정이 가능하도록 합니다.

## 5. 메타데이터 추출 및 활용 (Metadata Extraction and Utilization)

각 코드 청크 및 파일에 대해 풍부하고 구조화된 메타데이터를 추출하여 저장합니다. 이는 LLM의 이해도를 높이고, 검색 정확성을 향상시키며, 다양한 분석 기능을 지원하는 데 핵심적인 역할을 합니다.

### 5.1. 추출 대상 메타데이터

- **파일 수준 메타데이터 (File-level Metadata)**:
    - `file_id`: 파일의 고유 식별자 (예: UUID)
    - `file_path`: 저장소 내 상대 경로
    - `absolute_path`: 시스템 내 절대 경로
    - `file_name`: 파일 이름
    - `file_extension`: 파일 확장자
    - `language`: 프로그래밍 언어 (자동 감지)
    - `size_bytes`: 파일 크기 (바이트 단위)
    - `last_modified_at`: 최종 수정 일시 (Git 커밋 정보 활용)
    - `created_at`: 생성 일시 (Git 커밋 정보 활용)
    - `owner_repository`: 소속 저장소 정보 (URL, 이름)
    - `total_lines`: 전체 라인 수
    - `avg_line_length`: 평균 라인 길이
    - `checksum`: 파일 내용 해시 (무결성 검증용, 예: SHA256)
    - `dependencies`: 파일 수준에서 식별된 주요 라이브러리 및 모듈 의존성 (import/require 구문 분석)
    - `summary`: (선택적) 파일의 주요 기능에 대한 간략한 요약 (LLM 생성 또는 주석 기반)

- **청크 수준 메타데이터 (Chunk-level Metadata)**:
    - `chunk_id`: 청크의 고유 식별자 (예: UUID)
    - `parent_file_id`: 상위 파일 ID (파일 수준 메타데이터와 연결)
    - `chunk_text`: 실제 코드 청크 내용
    - `start_line`: 원본 파일에서의 시작 라인 번호
    - `end_line`: 원본 파일에서의 종료 라인 번호
    - `start_char_offset`: 원본 파일에서의 시작 문자 오프셋
    - `end_char_offset`: 원본 파일에서의 종료 문자 오프셋
    - `token_count`: 청크의 토큰 수 (`tiktoken` 기준)
    - `chunk_type`: 청크의 의미론적 유형 (예: `class_definition`, `function_definition`, `method_definition`, `block`, `comment_block`, `import_statement`, `standalone_code`) - AST 분석 결과 활용
    - `parent_constructs`: 상위 구조 정보 (예: 소속 클래스명, 함수명) - 계층 구조 표현
    - `child_constructs`: 하위 구조 정보 (예: 내부 함수, 내부 클래스)
    - `cyclomatic_complexity`: (선택적) 코드 복잡도 지표 (함수/메서드 청크 대상)
    - `role_tags`: (선택적) 청크의 주요 역할 태그 (예: `data_processing`, `ui_rendering`, `api_endpoint`, `error_handling`, `authentication`) - LLM 또는 패턴 기반 부여
    - `has_comments`: 주석 포함 여부
    - `comment_density`: 주석 밀도 (주석 라인 수 / 전체 라인 수)
    - `embedding_model`: 사용된 임베딩 모델명 (예: `text-embedding-3-small`)
    - `extracted_at`: 메타데이터 추출 시각

- **분석 정보 메타데이터 (Analysis Information Metadata)**:
    - `analysis_session_id`: 분석 세션 ID (동일 세션 내 분석 결과 그룹화)
    - `analysis_timestamp`: 분석 수행 시각
    - `quality_metrics`: (선택적) 정적 분석 도구 결과 (예: lint 경고/오류 수)
    - `version_info`: 코드 버전 정보 (Git commit hash 등)

### 5.2. 메타데이터 활용 방안

- **LLM 프롬프트 컨텍스트 강화**:
    - 질문과 관련된 코드 청크를 검색한 후, 해당 청크의 메타데이터(예: 파일 경로, 상위 클래스/함수명, 언어)를 함께 LLM 프롬프트에 제공하여 컨텍스트 이해도를 높입니다.
    - "이 함수는 `user_service.py` 파일의 `UserService` 클래스 내에 정의된 `get_user_details` 메서드입니다." 와 같은 형태로 컨텍스트 제공.

- **답변 정확성 및 관련성 향상**:
    - 메타데이터를 필터링 조건으로 사용하여(예: 특정 파일, 특정 클래스 내 함수 검색), 더 정확하고 관련성 높은 코드 청크를 검색 결과로 제시합니다.
    - "Python으로 작성된 인증 관련 함수 찾아줘" -> `language: python`, `role_tags: authentication` 필터링.

- **계층적 이해 및 탐색 지원**:
    - `parent_constructs` 및 `child_constructs` 메타데이터를 활용하여 코드의 계층 구조(파일 -> 클래스 -> 함수)를 이해하고, 사용자가 관련 코드를 탐색할 수 있도록 지원합니다.
    - "이 함수의 호출 관계를 알려줘" -> 의존성 및 호출 그래프 생성에 활용.

- **안전한 코드 수정 제안**:
    - 코드 수정 요청 시, `start_line`, `end_line`, `chunk_type` 등의 메타데이터를 활용하여 정확한 수정 위치를 식별하고, 변경 범위를 명확히 합니다.
    - 수정 전후의 코드 비교 및 영향 분석에 활용.

- **다단계 검색 전략 (Multi-stage Search)**:
    - 1단계: 키워드 및 의미론적 검색으로 후보 청크 그룹 식별.
    - 2단계: 후보 청크들의 메타데이터(예: 파일 경로, 의존성, 역할 태그)를 분석하여 관련성 점수 재조정 및 최종 결과 필터링.
    - 예: 사용자가 "결제 모듈에서 주문 처리 로직 찾아줘"라고 질문 시, 먼저 '결제', '주문 처리' 키워드로 청크 검색 후, `file_path`에 'payment' 또는 'order'가 포함되거나 `role_tags`에 'payment_processing'이 있는 청크의 우선순위를 높임.

- **코드 요약 및 문서화 지원**:
    - 파일 또는 클래스 단위의 청크들과 그 메타데이터를 종합하여 코드의 기능, 구조, 의존성에 대한 요약을 생성하는 데 활용합니다.

## 6. 데이터 저장 (Data Storage)

### 6.1. 저장소 선택: ChromaDB
- **선택 이유**:
    - 오픈소스 벡터 데이터베이스로, 임베딩 및 메타데이터 저장에 특화됨.
    - 로컬 실행 및 관리가 용이하며, Python 클라이언트 지원.
    - 메타데이터 필터링을 통한 효율적인 검색 지원.
- **대안**: FAISS (메타데이터 저장 별도 필요), Pinecone (클라우드 기반), Weaviate 등

### 6.2. 데이터베이스 스키마 및 구조
- **Collection**: 각 분석 세션 또는 저장소별로 별도의 컬렉션을 생성하여 데이터 격리.
    - 컬렉션 이름 규칙: `repo_{repository_name}_{session_id}` 또는 `user_{user_id}_session_{session_id}`
- **저장 항목**:
    - **ID**: 각 청크의 고유 ID (`chunk_id`)
    - **Embedding**: 코드 청크의 벡터 임베딩 (OpenAI `text-embedding-3-small` 모델 사용, 1536 차원)
    - **Document**: 원본 코드 청크 텍스트 (`chunk_text`)
    - **Metadata**: 해당 청크의 모든 추출된 메타데이터 (JSON 형식으로 저장)
        - `file_path`, `start_line`, `end_line`, `language`, `chunk_type`, `parent_constructs` 등 위에서 정의된 메타데이터 포함.

### 6.3. 데이터 추가 및 업데이트
- 새로운 저장소 분석 시, 해당 저장소의 모든 코드 파일에 대해 청킹 및 메타데이터 추출 후 ChromaDB 컬렉션에 일괄 추가.
- 기존 저장소 업데이트 시, 변경된 파일만 식별하여 해당 파일의 청크를 업데이트하거나 재분석 후 추가/삭제. (Git diff 활용)

## 7. 데이터 품질 관리 (Data Quality Management)

### 7.1. 유효성 검사 (Validation)
- **GitHub API 응답 유효성**: API 요청 성공 여부, 예상 데이터 형식 일치 여부 확인. Rate limit 초과 시 대기 및 재시도 로직 구현.
- **파일 인코딩 검사**: UTF-8 인코딩 시도, 실패 시 예외 처리 및 로깅. 필요시 `chardet` 등으로 원본 인코딩 감지 후 변환.
- **청킹 결과 검증**: 청크 크기가 과도하게 작거나 크지 않은지, 중첩이 적절한지 모니터링.
- **메타데이터 일관성**: 필수 메타데이터 필드가 누락되지 않았는지, 데이터 타입이 올바른지 검사.

### 7.2. 오류 처리 및 로깅
- **파싱 오류**: AST 파싱 실패 시 (지원하지 않는 문법, 손상된 파일 등), 해당 파일/청크는 건너뛰고 오류 로깅. Fallback 청킹 전략 적용.
- **임베딩 생성 오류**: API 호출 실패, 네트워크 문제 등으로 임베딩 생성 불가 시 재시도 또는 해당 청크 제외 및 로깅.
- **데이터베이스 오류**: ChromaDB 연결 실패, 쓰기/읽기 오류 발생 시 시스템 알림 및 로깅.
- **상세 로깅**: 파이프라인 각 단계(수집, 전처리, 청킹, 메타데이터 추출, 저장)의 진행 상황, 성공/실패 여부, 처리된 파일/청크 수 등을 상세히 로깅하여 문제 추적 및 디버깅 용이.

### 7.3. 중복 데이터 처리
- 파일 내용 해시(checksum)를 비교하여 동일 파일 재처리 방지.
- 청크 수준에서는 내용 기반 유사도 비교를 통해 잠재적 중복 식별 가능 (선택적).

## 8. 파이프라인 실행 흐름

1.  **사용자 요청**: 분석 대상 GitHub 저장소 URL 입력.
2.  **데이터 수집**: GitHub API를 통해 저장소 코드 및 메타데이터 다운로드.
3.  **전처리**: 파일 필터링, 기본 코드 정리.
4.  **코드 청킹**: AST 기반 의미론적 청킹 수행, 필요시 토큰 기반 크기 조절.
5.  **메타데이터 추출**: 파일 및 청크 수준에서 상세 메타데이터 생성.
6.  **임베딩 생성**: 각 코드 청크에 대해 `text-embedding-3-small` 모델로 임베딩 벡터 생성.
7.  **데이터 저장**: ChromaDB에 청크 ID, 임베딩, 원본 텍스트, 메타데이터 저장.
8.  **완료 알림**: 사용자에게 분석 완료 및 질의응답 가능 상태 알림.

## 9. 향후 개선 방향

- **실시간 동기화**: Git 웹훅을 사용하여 저장소 변경 사항을 실시간으로 파이프라인에 반영.
- **다국어 지원 확대**: 더 많은 프로그래밍 언어 및 자연어 지원.
- **고급 정적 분석 통합**: SonarQube, PMD 등과 같은 도구를 통합하여 코드 품질 지표, 버그 패턴 등을 메타데이터로 추가.
- **사용자 피드백 루프**: LLM 답변의 유용성, 검색 결과의 정확성에 대한 사용자 피드백을 수집하여 파이프라인 및 모델 개선에 활용.
- **시각화 대시보드**: 분석된 저장소의 통계, 코드 구조, 복잡도 등을 시각적으로 보여주는 대시보드 개발.

---

이 문서는 AI 코드 분석 파이프라인의 핵심 구성 요소와 작동 방식을 정의하며, 지속적인 개선과 확장을 위한 기반을 마련합니다.

### 가. 청킹 방법 및 과정

1.  **코드 분석 및 분할 (Code Parsing and Segmentation):**
    *   시스템은 분석 대상 GitHub 저장소의 각 소스 코드 파일을 가져옵니다.
    *   코드는 주로 의미론적 단위(예: 함수, 클래스, 메서드, 또는 독립적인 주요 코드 블록)로 분할됩니다. 이 과정은 코드의 구조를 파악하여 논리적으로 관련된 부분들을 하나의 청크로 묶는 것을 목표로 합니다.
    *   이를 위해 AST(Abstract Syntax Tree) 분석 등의 기법을 활용하여 코드의 구조적 경계를 식별할 수 있습니다.

2.  **메타데이터 추출 (Metadata Extraction):**
    *   각 코드 청크에 대해 다양한 메타데이터가 추출되어 저장됩니다. (자세한 내용은 아래 "2. 추출하는 메타데이터 및 메타데이터 활용" 섹션 참조)

3.  **임베딩 생성 및 저장 (Embedding Generation and Storage):**
    *   각 코드 청크의 텍스트 내용과 중요한 메타데이터 일부를 결합하여 임베딩(텍스트를 숫자 벡터로 변환한 것)을 생성합니다. 이 임베딩은 OpenAI의 `text-embedding-3-small` 모델을 사용합니다.
    *   생성된 임베딩과 해당 청크의 텍스트, 그리고 추출된 메타데이터는 ChromaDB라는 벡터 데이터베이스에 저장됩니다. 각 세션별로 `repo_{session_id}`와 같은 이름의 컬렉션에 저장되어 관리됩니다.

### 나. 청킹 방법의 장점

*   **효율적인 검색 (Efficient Retrieval):**
    *   사용자의 질문이 들어오면, 질문 또한 임베딩으로 변환된 후 ChromaDB에서 유사도 검색을 통해 가장 관련성 높은 코드 청크들을 빠르게 찾아낼 수 있습니다. 이는 전체 코드를 검색하는 것보다 훨씬 빠르고 효율적입니다.
*   **정확한 컨텍스트 제공 (Precise Context Provisioning):**
    *   작은 단위의 청크와 풍부한 메타데이터는 LLM(대규모 언어 모델)에게 코드의 특정 부분에 대한 명확하고 집중된 컨텍스트를 제공하여 답변의 정확도를 높입니다. LLM이 불필요한 정보에 혼란스러워하지 않도록 돕습니다.
*   **계층적 이해 증진 (Enhanced Hierarchical Understanding):**
    *   메타데이터(예: `parent_entity_names`, `filepath`, `class_name`)는 LLM이 코드의 전체적인 구조 내에서 청크의 위치와 역할을 파악하는 데 도움을 줍니다. 이는 코드 수정이나 복잡한 질문에 대한 답변 생성 시 중요합니다.
*   **2단계 검색 전략 지원 (Support for Two-Stage Search Strategy):**
    1.  **1단계 (Chunk-Level Retrieval):** 유사도 검색을 통해 관련성이 높은 코드 청크들을 먼저 신속하게 식별합니다.
    2.  **2단계 (File-Level Context Augmentation):** 1단계에서 식별된 청크들이 속한 파일들의 전체 내용을 로드하여 LLM에게 더 넓고 풍부한 컨텍스트를 제공합니다. 이를 통해 답변의 깊이와 품질을 크게 향상시킬 수 있으며, 청크만으로는 파악하기 어려운 파일 전체의 맥락을 이해할 수 있게 됩니다.
*   **유지보수성 및 확장성 (Maintainability and Scalability):**
    *   코드를 의미론적 단위로 청킹하면, 코드베이스가 변경되거나 확장될 때 영향을 받는 청크만 업데이트하거나 재생성하면 되므로 유지보수가 용이합니다.

## 2. 추출하는 메타데이터 및 메타데이터 활용

각 코드 청크에서 추출되는 메타데이터는 AI가 코드의 의미와 구조를 더 깊이 이해하는 데 결정적인 역할을 합니다.

### 가. 주요 추출 메타데이터 항목

*   **기본 정보 (Basic Information):**
    *   `filepath`: 청크가 속한 파일의 전체 경로
    *   `filename`: 파일 이름
    *   `start_line`, `end_line`: 청크의 시작 및 종료 줄 번호
*   **구조적 정보 (Structural Information):**
    *   `chunk_type`: 청크의 유형 (예: 'class', 'method', 'function', 'interface', 'enum', 'global_variable', 'code_block')
    *   `function_name`: 함수 이름 (해당하는 경우)
    *   `class_name`: 클래스 이름 (해당하는 경우)
    *   `parent_entity_names`: 상위 엔티티(예: 클래스 내 메서드인 경우 클래스 이름, 함수 내 지역 함수인 경우 바깥 함수 이름)의 계층적 목록
*   **의미론적 정보 (Semantic Information):**
    *   `docstrings`: 함수나 클래스, 모듈의 설명 문자열
    *   `parameters`: 함수의 매개변수 이름, 타입, 기본값
    *   `return_types`: 함수의 반환 유형
    *   `decorators`: 함수나 클래스에 사용된 데코레이터 목록
*   **복잡도 및 중요도 (Complexity and Importance):**
    *   `complexity_scores`: 코드 복잡도 점수 (예: Cyclomatic Complexity, Halstead Metrics 등) - 중요한 로직이나 복잡한 부분을 식별하는 데 사용
    *   `line_count`: 청크의 실제 코드 라인 수
*   **관계 정보 (Relational Information):**
    *   `inheritance_info`: 클래스 상속 정보 (부모 클래스, 인터페이스 구현 등)
    *   `dependency_info`: 임포트된 모듈, 라이브러리, 또는 다른 파일과의 의존성 정보
    *   `called_functions_classes`: 청크 내에서 호출하는 다른 함수나 사용하는 클래스 목록
*   **기능적 태그 (Functional Tags):**
    *   `role_tags`: 청크의 기능적 목적을 나타내는 태그 (예: 'data_processing', 'ui_component', 'api_endpoint', 'error_handling', 'authentication', 'database_interaction') - 도메인 지식이나 패턴 분석을 통해 부여 가능

### 나. 메타데이터 활용 방안

*   **LLM 프롬프트 강화 (Enhancing LLM Prompts):**
    *   추출된 메타데이터는 LLM에게 전달되는 프롬프트에 포함되어, 질문에 대한 컨텍스트를 풍부하게 합니다. 예를 들어, "이 `calculate_total` 함수(파일: `orders.py`, 클래스: `OrderProcessor`, 역할: '금액 계산')는 어떤 로직을 수행하나요?" 와 같이 구체적인 정보를 제공합니다.
*   **정확한 답변 생성 유도 (Guiding Accurate Answer Generation):**
    *   LLM은 메타데이터를 참조하여 코드의 특정 부분(예: 특정 함수, 클래스)에 집중하고, 파일 경로, 함수/클래스 이름 등을 명시적으로 언급하며 더 정확하고 명확한 답변을 생성할 수 있습니다.
*   **계층적 및 맥락적 이해 지원 (Supporting Hierarchical and Contextual Understanding):**
    *   `parent_entity_names`, `filepath`, `dependency_info` 등의 메타데이터는 LLM이 코드 청크를 고립된 조각이 아닌, 전체 프로젝트 구조 내의 한 부분으로 이해하도록 돕습니다.
*   **안전하고 맥락에 맞는 코드 수정 제안 (Facilitating Safe and Context-Aware Code Modification):**
    *   코드 수정 요청 시, 메타데이터(특히 의존성, 상속 정보, 역할 태그)는 LLM이 변경 사항의 잠재적 영향을 더 잘 예측하고 안전하며 맥락에 맞는 수정안을 제안하는 데 도움을 줍니다.
*   **2단계 검색 효율성 증대 (Increasing Efficiency of Two-Stage Search):**
    *   1단계에서 검색된 청크의 메타데이터(`filepath`)는 2단계에서 어떤 파일의 전체 내용을 로드해야 할지 정확히 알려주어 효율성을 높입니다.
*   **프로젝트 전체 구조 및 대화 컨텍스트 통합 (Integrating Project-Wide Structure and Conversation Context):**
    *   디렉토리 구조 정보(별도 추출)와 함께 청크 메타데이터는 프로젝트 전체에 대한 이해를 돕고, 이전 대화에서 언급된 코드 요소(메타데이터로 식별 가능)와의 연관성을 파악하는 데 사용될 수 있습니다.
*   **중요 코드 식별 (Identifying Critical Code Sections):**
    *   `complexity_scores`나 특정 `role_tags`를 가진 청크는 중요한 로직을 포함할 가능성이 높으므로, 분석 시 우선순위를 두거나 더 주의 깊게 다룰 수 있습니다.

이러한 청킹 방법과 메타데이터 활용은 AI 챗봇이 복잡한 코드베이스를 효과적으로 이해하고, 사용자의 다양한 질문에 대해 정확하고 맥락에 맞는 답변을 생성하며, 안전한 코드 수정을 지원하는 핵심적인 기술입니다.
