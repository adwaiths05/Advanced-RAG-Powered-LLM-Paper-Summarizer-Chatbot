import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from rouge_score import rouge_scorer
from backend import process_pdfs, answer_query

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
                print(f"Image {i} Caption: {img_result['caption']}")
    
    avg_metrics = {
        "precision": np.mean(retrieval_metrics["precision"]),
        "recall": np.mean(retrieval_metrics["recall"]),
        "f1": np.mean(retrieval_metrics["f1"]),
        "rouge1": np.mean(generation_metrics["rouge1"]),
        "rougeL": np.mean(generation_metrics["rougeL"])
    }
    return avg_metrics


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
