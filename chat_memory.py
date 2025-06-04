"""
대화 기록 관리 모듈.
이전 대화를 저장하고 검색하는 기능을 제공합니다.
"""

import chromadb
from chromadb.utils import embedding_functions
import os
import hashlib
import time
from datetime import datetime
import traceback
import openai
import numpy as np
import unicodedata
import re
from dotenv import load_dotenv

load_dotenv()

# ChromaDB를 위한 디렉토리
MEMORY_DB_PATH = "./chat_memory_db"

# 해시 생성 함수
def compute_hash(text: str) -> str:
    """입력 텍스트의 해시값을 생성합니다."""
    # 정규화된 텍스트의 해시값 생성
    hash_value = hashlib.md5(text.encode('utf-8')).hexdigest()
    print(f"[DEBUG] 생성된 해시값: {hash_value}")
    return hash_value

# ChromaDB 클라이언트 초기화
def init_memory_client():
    """ChromaDB 클라이언트를 초기화합니다."""
    try:
        os.makedirs(MEMORY_DB_PATH, exist_ok=True)
        client = chromadb.PersistentClient(path=MEMORY_DB_PATH)
        print(f"[DEBUG] 대화 기록 DB 클라이언트 초기화 완료: {MEMORY_DB_PATH}")
        return client
    except Exception as e:
        print(f"[ERROR] 대화 기록 DB 클라이언트 초기화 실패: {e}")
        traceback.print_exc()
        return None

# OpenAI 임베딩 함수 생성
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.getenv("OPENAI_API_KEY"),
    model_name="text-embedding-3-small"
)

# 메모리 클라이언트 초기화
memory_client = init_memory_client()

def get_or_create_collection(session_id: str):
    """세션별 대화 기록 컬렉션을 가져오거나 생성합니다."""
    if not memory_client:
        print("[ERROR] 메모리 클라이언트가 초기화되지 않았습니다.")
        return None
    
    collection_name = f"chat_memory_{session_id}"
    try:
        # 컬렉션 존재 여부 확인
        collections = [col.name for col in memory_client.list_collections()]
        
        if collection_name not in collections:
            print(f"[DEBUG] 컬렉션 생성: {collection_name}")
            collection = memory_client.create_collection(
                name=collection_name,
                embedding_function=openai_ef,
                metadata={"description": f"대화 기록 - 세션 {session_id}"}
            )
        else:
            print(f"[DEBUG] 기존 컬렉션 사용: {collection_name}")
            collection = memory_client.get_collection(
                name=collection_name,
                embedding_function=openai_ef
            )
        
        return collection
    except Exception as e:
        print(f"[ERROR] 컬렉션 가져오기/생성 실패: {e}")
        traceback.print_exc()
        return None

def normalize_question(question: str) -> str:
    """질문을 정규화합니다."""
    # 1. 유니코드 정규화 (NFKC)
    question = unicodedata.normalize('NFKC', question)
    
    # 2. 소문자 변환
    question = question.lower()
    
    # 3. 특수문자 및 공백 처리
    # 모든 공백 문자를 일반 공백으로 변환
    question = re.sub(r'\s+', ' ', question)
    # 앞뒤 공백 제거
    question = question.strip()
    
    # 4. 문장부호 정규화
    # 쉼표, 마침표 등을 공백으로 변환
    question = re.sub(r'[,.!?;:，。！？；：]', ' ', question)
    # 연속된 공백을 하나로
    question = re.sub(r'\s+', ' ', question)
    # 앞뒤 공백 제거
    question = question.strip()
    
    print(f"[DEBUG] 정규화 전 질문: '{question}'")
    print(f"[DEBUG] 정규화 후 질문: '{question}'")
    
    return question

def normalize_embedding(embedding):
    """임베딩 벡터를 정규화하고 반올림합니다."""
    if isinstance(embedding, list):
        embedding = np.array(embedding)
    
    # L2 정규화
    norm = np.linalg.norm(embedding)
    if norm == 0:
        return embedding
    
    # 정규화 및 반올림
    normalized = embedding / norm
    # 소수점 6자리까지 반올림
    normalized = np.round(normalized, decimals=6)
    
    # 정규화 확인
    norm_after = np.linalg.norm(normalized)
    print(f"[DEBUG] 정규화 후 L2 norm: {norm_after:.6f}")
    
    return normalized

def get_embedding(text: str) -> np.ndarray:
    """텍스트의 임베딩을 생성하고 정규화합니다."""
    try:
        # 1. 텍스트 정규화
        normalized_text = normalize_question(text)
        
        # 2. 임베딩 생성
        embedding = openai_ef(normalized_text)
        if hasattr(embedding, 'tolist'):
            embedding = embedding.tolist()
        if not isinstance(embedding, list):
            embedding = [embedding]
        
        # 3. 임베딩 정규화
        normalized_embedding = normalize_embedding(embedding[0])
        
        print(f"[DEBUG] 임베딩 생성 완료 (차원: {len(normalized_embedding)})")
        print(f"[DEBUG] 임베딩 첫 5개 값: {normalized_embedding[:5]}")
        
        return normalized_embedding
    except Exception as e:
        print(f"[ERROR] 임베딩 생성 중 오류 발생: {e}")
        raise

def cosine_similarity(v1, v2):
    """두 벡터 간의 코사인 유사도를 계산합니다."""
    # 벡터가 리스트인 경우 numpy 배열로 변환
    if isinstance(v1, list):
        v1 = np.array(v1)
    if isinstance(v2, list):
        v2 = np.array(v2)
    
    # 정규화 확인
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    print(f"[DEBUG] 벡터 정규화 확인 - v1: {norm_v1:.6f}, v2: {norm_v2:.6f}")
    
    # 코사인 유사도 계산
    dot_product = np.dot(v1, v2)
    similarity = dot_product / (norm_v1 * norm_v2)
    
    # 반올림하여 반환
    return np.round(similarity, decimals=6)

def save_conversation(session_id, question, answer):
    """대화 내용을 저장하고 중복을 제거합니다."""
    try:
        # 1. 질문 정규화
        normalized_question = normalize_question(question)
        print(f"[DEBUG] 정규화된 질문: '{normalized_question}'")
        
        # 2. 현재 대화의 해시값 계산
        current_hash = compute_hash(normalized_question)
        print(f"[DEBUG] 현재 질문 해시: {current_hash}")
        
        # 3. 컬렉션 가져오기 또는 생성
        collection = get_or_create_collection(session_id)
        if not collection:
            print("[ERROR] 컬렉션을 가져오거나 생성할 수 없습니다.")
            return
        
        # 4. 현재 질문의 임베딩 생성
        embedding = get_embedding(normalized_question)
        
        # 5. 해시값으로 중복 체크
        results = collection.get(
            where={"question_hash": current_hash}
        )
        
        if results and len(results['ids']) > 0:
            print(f"[DEBUG] 동일한 해시값의 이전 질문이 발견되어 중복을 제거합니다.")
            return
        
        # 6. 유사한 이전 대화 검색
        results = collection.query(
            query_embeddings=[embedding.tolist()],
            n_results=5,
            include=['embeddings', 'documents', 'metadatas']
        )
        
        # 7. 직접 코사인 유사도 계산
        if 'embeddings' in results and results['embeddings']:
            print("\n[DEBUG] 유사도 검사 결과:")
            for idx, stored_embedding in enumerate(results['embeddings'][0]):
                # 저장된 임베딩 정규화
                stored_embedding = normalize_embedding(stored_embedding)
                # 코사인 유사도 계산
                similarity = cosine_similarity(embedding, stored_embedding)
                print(f"대화 {idx+1} 코사인 유사도: {similarity:.6f}")
                
                # 유사도가 0.95 이상이면 중복으로 간주
                if similarity > 0.95:
                    print(f"[DEBUG] 유사도가 높은 이전 대화 발견 (유사도: {similarity:.6f})")
                    return
        
        # 8. 새로운 대화 저장
        collection.add(
            documents=[f"Q: {normalized_question}\nA: {answer}"],
            metadatas=[{
                "session_id": session_id,
                "question_hash": current_hash,
                "timestamp": datetime.now().isoformat(),
                "original_question": question
            }],
            ids=[f"{session_id}_{current_hash}"]
        )
        print(f"[DEBUG] 새로운 대화 저장 완료: {current_hash}")
        
    except Exception as e:
        print(f"[ERROR] 대화 저장 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

def get_relevant_conversations(session_id: str, query: str, top_k: int = 3) -> str:
    """현재 질문과 관련된 이전 대화를 검색합니다."""
    try:
        # 1. 질문 정규화
        normalized_query = normalize_question(query)
        print(f"[DEBUG] 정규화된 검색 질문: '{normalized_query}'")
        
        # 2. 현재 질문의 해시값 계산
        current_hash = compute_hash(normalized_query)
        print(f"[DEBUG] 현재 질문 해시: {current_hash}")
        
        # 3. 컬렉션 가져오기 또는 생성
        collection = get_or_create_collection(session_id)
        if not collection:
            print("[ERROR] 컬렉션을 가져오거나 생성할 수 없습니다.")
            return "이전 대화 없음"
        
        # 4. 해시값으로 정확히 일치하는 대화 검색
        results = collection.get(
            where={"question_hash": current_hash}
        )
        
        if results and len(results['ids']) > 0:
            print(f"[DEBUG] 정확히 일치하는 이전 대화를 찾았습니다.")
            return results['documents'][0]
        
        # 5. 현재 질문의 임베딩 생성
        embedding = get_embedding(normalized_query)
        
        # 6. 유사한 이전 대화 검색
        results = collection.query(
            query_embeddings=[embedding.tolist()],
            n_results=top_k,
            include=['embeddings', 'documents', 'metadatas']
        )
        
        # 7. 검색 결과 처리
        if not results or 'documents' not in results or not results['documents'] or not results['documents'][0]:
            print("[DEBUG] 관련된 이전 대화가 없습니다.")
            return "이전 대화 없음"
        
        # 8. 대화 기록 포맷팅
        conversations = []
        if 'embeddings' in results and results['embeddings']:
            print("\n[DEBUG] 유사도 검사 결과:")
            for idx, stored_embedding in enumerate(results['embeddings'][0]):
                # 저장된 임베딩 정규화
                stored_embedding = normalize_embedding(stored_embedding)
                # 코사인 유사도 계산
                similarity = cosine_similarity(embedding, stored_embedding)
                print(f"대화 {idx+1} 코사인 유사도: {similarity:.6f}")
                
                # 유사도가 0.8 이상인 대화만 포함
                if similarity > 0.8:
                    # 원본 질문이 있으면 표시
                    original_question = results['metadatas'][0][idx].get('original_question', '')
                    if original_question and original_question != normalized_query:
                        conversations.append(f"[관련 대화 {idx+1} (유사도: {similarity:.6f}, 원본: '{original_question}')]\n{results['documents'][0][idx]}")
                    else:
                        conversations.append(f"[관련 대화 {idx+1} (유사도: {similarity:.6f})]\n{results['documents'][0][idx]}")
                else:
                    print(f"[DEBUG] 대화 {idx+1} 유사도가 너무 낮아 제외됨: {similarity:.6f}")
        
        if not conversations:
            print("[DEBUG] 유사도가 충분히 높은 이전 대화가 없습니다.")
            return "이전 대화 없음"
        
        # 9. 대화 기록 반환
        return "\n\n".join(conversations)
        
    except Exception as e:
        print(f"[ERROR] 대화 검색 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return "이전 대화 없음"

def reset_memory(session_id=None):
    """대화 기록을 초기화합니다. session_id가 None이면 모든 세션의 기록을 삭제합니다."""
    if not memory_client:
        print("[ERROR] 메모리 클라이언트가 초기화되지 않았습니다.")
        return False
    
    try:
        if session_id:
            collection_name = f"memory_{session_id}"
            try:
                memory_client.delete_collection(collection_name)
                print(f"[DEBUG] 세션 {session_id}의 대화 기록 삭제 완료")
            except Exception:
                print(f"[WARNING] 세션 {session_id}의 컬렉션이 존재하지 않거나 삭제할 수 없습니다.")
        else:
            # 모든 메모리 컬렉션 삭제
            collections = memory_client.list_collections()
            for collection in collections:
                if collection.name.startswith("memory_"):
                    memory_client.delete_collection(collection.name)
                    print(f"[DEBUG] 컬렉션 {collection.name} 삭제 완료")
        return True
    except Exception as e:
        print(f"[ERROR] 대화 기록 초기화 실패: {e}")
        traceback.print_exc()
        return False