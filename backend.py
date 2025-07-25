import pdfplumber
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import ollama
import os
import kagglehub
from fastapi import FastAPI
import uvicorn

# Download dataset locally
dataset_path = kagglehub.dataset_download('ruchi798/100-llm-papers-to-explore')

# Initialize sentence transformer
text_model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize FAISS-GPU index
dimension = 384  # all-MiniLM-L6-v2 embedding size
res = faiss.StandardGpuResources()  # GPU resources
index = faiss.IndexFlatL2(dimension)
gpu_index = faiss.index_cpu_to_gpu(res, 0, index)  # Move to GPU
metadata = []

# Extract text from PDF
def extract_text(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = "".join(page.extract_text() or "" for page in pdf.pages)
    return text

# Process and store documents
def process_pdfs(pdf_dir=dataset_path):
    for pdf_file in os.listdir(pdf_dir):
        if pdf_file.endswith(".pdf"):
            pdf_path = os.path.join(pdf_dir, pdf_file)
            text = extract_text(pdf_path)
            embedding = text_model.encode(text)
            gpu_index.add(np.array([embedding], dtype=np.float32))
            metadata.append({"id": pdf_file, "title": pdf_file})

# Query RAG
def query_rag(query, top_k=3):
    query_emb = text_model.encode(query)
    distances, indices = gpu_index.search(np.array([query_emb], dtype=np.float32), top_k)
    return [metadata[i] for i in indices[0]]

# Generate response with Ollama Mistral
def generate_response(query, context):
    prompt = f"Context: {context}\nQuery: {query}\nAnswer:"
    response = ollama.generate(model="mistral", prompt=prompt, options={"num_predict": 200})
    return response['response']

# FastAPI app
app = FastAPI()

@app.get("/query")
async def query_endpoint(query: str):
    contexts = query_rag(query)
    context_text = " ".join([meta["title"] for meta in contexts])
    response = generate_response(query, context_text)
    return {"response": response}

# Process PDFs
process_pdfs()

# Run server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
