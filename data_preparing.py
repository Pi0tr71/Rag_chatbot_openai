import os
import re
import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt_tab')

def clean_text(text: str) -> str:
    text = re.sub(r'[^a-zA-Z0-9 \.,;\?\!\-]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

raw_dir = 'data/raw'
documents = []
for file_name in os.listdir(raw_dir):
    if file_name.endswith('.txt'):
        with open(os.path.join(raw_dir, file_name), 'r', encoding='utf-8') as f:
            text = f.read()
            text = clean_text(text)
            documents.append((file_name, text))

total_words = sum(len(doc[1].split()) for doc in documents)
print(f"Liczba dokumentów: {len(documents)}")
print(f"Łączna liczba słów: {total_words}")

chunked_docs = []
for file_name, text in documents:
    sentences = sent_tokenize(text, language='english') 
    chunk = []
    for sent in sentences:
        if sum(len(s.split()) for s in chunk) + len(sent.split()) < 200:
            chunk.append(sent)
        else:
            chunked_docs.append({
                'source': file_name,
                'content': ' '.join(chunk)
            })
            chunk = [sent]
    if chunk:
        chunked_docs.append({
            'source': file_name,
            'content': ' '.join(chunk)
        })

print(f"Liczba chunków: {len(chunked_docs)}")

import json
with open('data/processed/chunked_docs.json', 'w', encoding='utf-8') as f:
    json.dump(chunked_docs, f, ensure_ascii=False, indent=2)
