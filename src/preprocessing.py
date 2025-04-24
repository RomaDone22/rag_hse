import pandas as pd
from sentence_transformers import SentenceTransformer
import re
import pickle
import json
from tqdm import tqdm
import torch
import os

# Проверяем доступность CUDA
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load SentenceTransformer model
embedding_model = SentenceTransformer('multi-qa-mpnet-base-dot-v1', device=device)
embedding_model.max_seq_length = 512
# Text normalization function
def normalize_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^a-zA-Zа-яА-Я0-9\s]', '', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

# Function to process QA dataset (CSV)
def process_qa_dataset(input_csv):
    print("Processing QA dataset...")
    qa_data = pd.read_csv(input_csv, delimiter='\t')
    qa_data['intent_texts'] = qa_data['intent_texts'].fillna('').astype(str)

    embeddings = []
    records = []

    for _, row in tqdm(qa_data.iterrows(), desc="Generating QA Embeddings", total=len(qa_data)):
        question = row['answer_text']
        normalized_question = normalize_text(question)
        embedding = embedding_model.encode(normalized_question)

        records.append({
            'question': question,
            'normalized_question': normalized_question,
            'embedding': embedding
        })
        embeddings.append(embedding)

    # Save embeddings and records to a pickle file
    output_path = './data/qa_embeddings.pkl'
    with open(output_path, 'wb') as f:
        pickle.dump(records, f)

    print(f"QA embeddings saved to {output_path}")

# Function to process knowledge base (JSON)
def process_knowledge_base(input_json):
    print("Processing knowledge base...")
    with open(input_json, 'r', encoding='utf-8') as f:
        kb_data = json.load(f)

    embeddings = []
    records = []

    for entry in tqdm(kb_data, desc="Generating Knowledge Base Embeddings"):
        url = entry.get('url', '')
        title = entry.get('title', '')

        for block in entry['content']:
            if block['type'] == 'paragraph':
                text = block['text']
                normalized_text = normalize_text(text)
                embedding = embedding_model.encode(normalized_text)

                records.append({
                    'url': url,
                    'title': title,
                    'text': text,
                    'normalized_text': normalized_text,
                    'embedding': embedding
                })
                embeddings.append(embedding)

    # Save embeddings and records to a pickle file
    output_path = './data/document_embeddings.pkl'
    with open(output_path, 'wb') as f:
        pickle.dump(records, f)

    print(f"Knowledge base embeddings saved to {output_path}")

# Main function
def preprocess_and_save(qa_csv, kb_json):
    # Ensure output directory exists
    os.makedirs('./data', exist_ok=True)

    # Process QA dataset
    process_qa_dataset(qa_csv)

    # Process Knowledge Base
    process_knowledge_base(kb_json)

# Run preprocessing
if __name__ == "__main__":
    qa_csv_path = './data/result.csv'  # Path to QA dataset
    kb_json_path = './data/selsup_articles.json'  # Path to JSON knowledge base

    preprocess_and_save(qa_csv_path, kb_json_path)
