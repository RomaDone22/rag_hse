import pickle
import numpy as np
import faiss
import random
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder  
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
import retriever

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

sentence_model = SentenceTransformer('multi-qa-mpnet-base-dot-v1', device=device)

MODEL_PATH = "data/fine_tuned_openllama" # local deploy
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
llama_model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to(device)

sentence_model = SentenceTransformer('multi-qa-mpnet-base-dot-v1', device=device)

reranker = CrossEncoder('BAAI/bge-reranker-large', device=device, max_length=512)

def normalize_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Zа-яА-Я0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Clean generated text
def clean_generated_text(text):
    text = re.sub(r'Ответ:\s*', '', text)  
    text = re.sub(r'\b([^\s]+)\b(\s+\1\b)+', r'\1', text)  
    text = re.sub(r'\s+', ' ', text).strip()
    return text


with open('./data/qa_embeddings.pkl', 'rb') as f:
    qa_data = pickle.load(f)
qa_embeddings = np.array([entry['embedding'] for entry in qa_data]).astype('float32')
qa_texts = [entry['question'] for entry in qa_data]


with open('./data/document_embeddings.pkl', 'rb') as f:
    doc_data = pickle.load(f)


doc_embeddings = np.array([entry['embedding'] for entry in doc_data]).astype('float32')
doc_texts = [entry['text'] for entry in doc_data]

retriever.doc_texts = doc_texts
retriever.init_retriever(doc_embeddings)


qa_embeddings = qa_embeddings / np.linalg.norm(qa_embeddings, axis=1, keepdims=True)
doc_embeddings = doc_embeddings / np.linalg.norm(doc_embeddings, axis=1, keepdims=True)


qa_index = faiss.IndexFlatIP(qa_embeddings.shape[1])
qa_index.add(qa_embeddings)

doc_index = faiss.IndexFlatIP(doc_embeddings.shape[1])
doc_index.add(doc_embeddings)

# Function to search in Faiss - just for QA
def search_faiss(query_embedding, faiss_index, texts, top_n=1):
    query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
    distances, indices = faiss_index.search(query_embedding, top_n)
    return [(texts[i], distances[0][j]) for j, i in enumerate(indices[0])]


def generate_human_response(context, query, max_new_tokens=300, temperature=0.5, top_p=0.8):
    input_text = f"Context: {context}\nQuestion: {query}\nAnswer:"
    inputs = tokenizer(input_text, return_tensors="pt").to(device)

    torch.manual_seed(SEED)  
    outputs = llama_model.generate(
        inputs["input_ids"],
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=True
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return clean_generated_text(response)

def get_response(query, threshold=0.7):
    """
    Возвращает ответ на запрос, используя приоритет документации с reranker.
    """
    normalized_query = normalize_text(query)
    query_embedding = sentence_model.encode([normalized_query]).astype('float32')
    query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)

    # Поиск по базе документов (top-5) через retriever (Milvus или Faiss - нужный флаг выбрать)
    doc_results = retriever.search_documents(query_embedding, top_k=5)

    # Поиск по базе QA (top-1) через Faiss
    qa_results = search_faiss(query_embedding, qa_index, qa_texts, top_n=1)

    # Реранк документов через cross-encoder (если что-то найдено)
    reranked_docs = []
    if doc_results:
        pairs = [(query, text) for text, _ in doc_results]
        cross_scores = reranker.predict(pairs)
        reranked_docs = sorted(
            [(doc_results[i][0], cross_scores[i]) for i in range(len(doc_results))],
            key=lambda x: x[1],
            reverse=True
        )

    # Если хороший документ найден — генерируем ответ
    if reranked_docs and reranked_docs[0][1] >= threshold:
        top_contexts = [text for text, score in reranked_docs[:3]]
        context = "\n\n".join(top_contexts)
        return generate_human_response(context, query)

    # Иначе fallback на QA (если результат достаточно релевантный)
    if qa_results and qa_results[0][1] >= threshold:
        return qa_results[0][0]

    # Вообще ничего не найдено
    return "Не найдено подходящей информации."



if __name__ == "__main__":
    query = "Как удалить товар?"
    response = get_response(query)
    print(f"Query: {query}")
    print(f"Answer: {response}")
