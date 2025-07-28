import os
import fitz  # PyMuPDF
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline
from sklearn.metrics import precision_score, recall_score, f1_score
from rouge_score import rouge_scorer

# Setup models
try:
    text_embedder = SentenceTransformer('all-MiniLM-L6-v2')
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    gpt = pipeline("text-generation", model="distilgpt2")
except Exception as e:
    print(f"Error loading models: {e}")
    exit(1)

# Global variables
all_chunks = []
all_images = []
all_captions = []
index = None

# 1. Extract text and images from PDFs
def extract_data_from_pdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        texts, images = [], []
        for page in doc:
            text = page.get_text("text")[:5000]  # Limit text length
            if text:
                texts.append(text)
            for img in page.get_images(full=True):
                xref = img[0]
                base_image = doc.extract_image(xref)
                images.append(base_image["image"])
        doc.close()
        return texts, images
    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")
        return [], []

# 2. Generate image captions
def generate_caption(image):
    try:
        inputs = blip_processor(images=image, return_tensors="pt")
        out = blip_model.generate(**inputs)
        return blip_processor.decode(out[0], skip_special_tokens=True)
    except Exception as e:
        print(f"Error generating caption: {e}")
        return ""

# 3. Index text and captions in FAISS
def create_index(texts, captions, images):
    global all_chunks, all_images, all_captions, index
    all_chunks = texts + captions
    all_images = images
    all_captions = captions
    
    if not all_chunks:
        return False
    
    try:
        embeddings = text_embedder.encode(all_chunks, show_progress_bar=True)
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        return True
    except Exception as e:
        print(f"Error indexing data: {e}")
        return False

# 4. Query the system
def answer_query(query, top_k=5):
    global index, all_chunks
    if not all_chunks or index is None:
        return "No data indexed. Please process PDFs first.", [], []
    
    try:
        query_embedding = text_embedder.encode([query])
        distances, indices = index.search(query_embedding, top_k)
        retrieved_chunks = [all_chunks[i] for i in indices[0]]
        
        image_results = []
        for chunk in retrieved_chunks:
            if chunk in all_captions:
                img_idx = all_captions.index(chunk)
                image_results.append({"caption": chunk, "image": all_images[img_idx]})
        
        context = "\n".join(retrieved_chunks)
        answer = gpt(f"Question: {query}\nContext: {context}", max_length=150, num_return_sequences=1)[0]["generated_text"]
        return answer, image_results, retrieved_chunks
    except Exception as e:
        print(f"Error querying: {e}")
        return "Error retrieving data.", [], []

# 5. Evaluate retrieval and generation
def evaluate_system():
    ground_truth = [
        {
            "query": "What is fine-tuning in LLMs?",
            "relevant_chunks": ["Fine-tuning is adjusting a pre-trained LLM for specific tasks."],
            "reference_answer": "Fine-tuning is the process of further training a pre-trained large language model on a specific dataset to improve its performance for a particular task."
        },
        {
            "query": "What does the model architecture look like?",
            "relevant_chunks": ["A diagram showing a transformer-based architecture."],
            "reference_answer": "The model architecture typically includes multiple transformer layers with attention mechanisms and feed-forward networks."
        }
    ]
    
    retrieval_metrics = {"precision": [], "recall": [], "f1": []}
    generation_metrics = {"rouge1": [], "rougeL": []}
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    
    for item in ground_truth:
        query = item["query"]
        relevant_chunks = item["relevant_chunks"]
        reference_answer = item["reference_answer"]
        
        answer, image_results, retrieved_chunks = answer_query(query, top_k=5)
        
        y_true = [1 if chunk in relevant_chunks else 0 for chunk in retrieved_chunks]
        y_pred = [1] * len(retrieved_chunks)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        retrieval_metrics["precision"].append(precision)
        retrieval_metrics["recall"].append(recall)
        retrieval_metrics["f1"].append(f1)
        
        scores = scorer.score(reference_answer, answer)
        generation_metrics["rouge1"].append(scores["rouge1"].f_measure)
        generation_metrics["rougeL"].append(scores["rougeL"].f_measure)
        
        print(f"\nQuery: {query}")
        print(f"Answer: {answer}")
        if image_results:
            print("Relevant Images:")
            for i, img_result in enumerate(image_results, 1):
                img_path = f"output_image_{i}.png"
                try:
                    with open(img_path, "wb") as f:
                        f.write(img_result["image"])
                    print(f"Image {i} Caption: {img_result['caption']}")
                except Exception as e:
                    print(f"Error saving image {i}: {e}")
    
    avg_metrics = {
        "precision": np.mean(retrieval_metrics["precision"]),
        "recall": np.mean(retrieval_metrics["recall"]),
        "f1": np.mean(retrieval_metrics["f1"]),
        "rouge1": np.mean(generation_metrics["rouge1"]),
        "rougeL": np.mean(generation_metrics["rougeL"])
    }
    return avg_metrics

# 6. Process PDFs and prepare data
def process_pdfs(pdf_dir="/kaggle/input/100-llm-papers-to-explore/"):
    global all_images, all_captions
    all_texts = []
    
    try:
        pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith(".pdf")][:100]
        for pdf_file in pdf_files:
            pdf_path = os.path.join(pdf_dir, pdf_file)
            print(f"Processing {pdf_file}")
            texts, images = extract_data_from_pdf(pdf_path)
            captions = [generate_caption(img) for img in images if img]
            all_texts.extend(texts)
            all_images.extend(images)
            all_captions.extend(captions)
        
        return create_index(all_texts, all_captions, all_images)
    except Exception as e:
        print(f"Error processing PDFs: {e}")
        return False

# Run the system
if __name__ == "__main__":
        success = process_pdfs()
        print("PDFs processed successfully." if success else "No data extracted from PDFs.")
        metrics = evaluate_system()
        print("\nEvaluation Metrics:")
        print(f"Retrieval Precision: {metrics['precision']:.3f}")
        print(f"Retrieval Recall: {metrics['recall']:.3f}")
        print(f"Retrieval F1: {metrics['f1']:.3f}")
        print(f"ROUGE-1: {metrics['rouge1']:.3f}")
        print(f"ROUGE-L: {metrics['rougeL']:.3f}")
