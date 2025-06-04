import pytest
import json
import logging
from datetime import datetime
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app import app
from github_analyzer import analyze_repository
from chat_handler import handle_chat
from dotenv import load_dotenv

load_dotenv()

# 로깅 설정
def setup_logging():
    """로깅 설정"""
    log_dir = r'han\test_logs'
    os.makedirs(log_dir, exist_ok=True)
    
    # 로그 파일명에 타임스탬프 추가
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f'{log_dir}/test_run_{timestamp}.log'
    
    # 로거 설정
    logger = logging.getLogger('test_logger')
    logger.setLevel(logging.DEBUG)
    
    # 파일 핸들러 설정
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    
    # 콘솔 핸들러 설정
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 포맷터 설정
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 핸들러 추가
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# 테스트 데이터
TEST_DATA = {
    "repo_url": os.getenv("OPENAI_API_KEY"),
    "token": os.getenv("GITHUB_TOKEN"),
    "questions": [
        "이 프로젝트의 주요 기능은 무엇인가요?",
        "크롤링 코드를 보여주세요.",
    ]
}

@pytest.fixture
def client():
    """Flask 테스트 클라이언트 생성"""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_repository_analysis_and_chat(client):
    """저장소 분석 및 채팅 테스트를 통합"""
    logger = setup_logging()
    logger.info("테스트 시작")
    logger.info(f"테스트 저장소: {TEST_DATA['repo_url']}")
    
    try:
        # 1. 저장소 분석
        logger.info("저장소 분석 시작")
        response = client.post('/analyze', 
            json={
                'repo_url': TEST_DATA['repo_url'],
                'token': TEST_DATA['token']
            }
        )
        
        # 응답 확인
        assert response.status_code == 200
        logger.info("저장소 분석 요청 성공")
        
        # 스트리밍 응답 처리
        session_id = None
        for line in response.data.decode('utf-8').split('\n'):
            if line.strip():  # 빈 라인 무시
                try:
                    data = json.loads(line)
                    logger.debug(f"분석 진행 상황: {data}")
                    if data.get('progress') == 100:  # 마지막 응답
                        session_id = data.get('session_id')
                        assert session_id is not None, "세션 ID가 생성되지 않았습니다."
                        logger.info(f"세션 ID 생성 완료: {session_id}")
                except json.JSONDecodeError as e:
                    logger.error(f"JSON 파싱 오류: {e}")
                    continue
        
        assert session_id is not None, "저장소 분석이 실패했습니다."
        
        # 2. 채팅 테스트
        logger.info("채팅 테스트 시작")
        responses = []
        for question in TEST_DATA['questions']:
            logger.info(f"질문 처리 중: {question}")
            response = client.post('/chat',
                json={
                    'session_id': session_id,
                    'message': question
                }
            )
            
            assert response.status_code == 200
            data = json.loads(response.data)
            logger.debug(f"채팅 응답 데이터: {data}")
            
            # 응답 데이터 구조 확인 및 저장
            response_data = {
                'question': question,
                'answer': data.get('response', '') if 'response' in data else data.get('answer', ''),
                'status': 'success' if 'error' not in data else 'error',
                'raw_response': data  # 디버깅을 위해 원본 응답도 저장
            }
            
            # 응답이 비어있거나 에러인 경우 테스트 실패
            assert response_data['status'] == 'success', f"채팅 응답 실패: {data.get('error', '알 수 없는 오류')}"
            assert response_data['answer'], "응답이 비어있습니다."
            
            responses.append(response_data)
            logger.info(f"질문 응답 완료: {response_data['status']}")
            logger.debug(f"응답 내용: {response_data['answer'][:200]}...")
        
        # 결과를 파일로 저장
        save_test_results(responses, session_id, logger)
        logger.info("테스트 완료")
        
    except Exception as e:
        logger.error(f"테스트 중 오류 발생: {str(e)}", exc_info=True)
        raise

def save_test_results(responses, session_id, logger):
    """테스트 결과를 파일로 저장"""
    try:
        results_dir = r'han\test_results'
        os.makedirs(results_dir, exist_ok=True)
        
        results = {
            'repo_url': TEST_DATA['repo_url'],
            'test_time': str(datetime.now()),
            'session_id': session_id,
            'responses': responses
        }
        
        result_file = f'{results_dir}/chat_test_results.json'
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"테스트 결과 저장 완료: {result_file}")
        
    except Exception as e:
        logger.error(f"결과 저장 중 오류 발생: {str(e)}", exc_info=True)
        raise

if __name__ == '__main__':
    pytest.main(['-v', 'test_chatbot.py']) 