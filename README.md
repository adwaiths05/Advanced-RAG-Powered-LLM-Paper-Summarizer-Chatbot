# Advanced LLM Paper Summarizer Chatbot

## Overview

This project is a Retrieval-Augmented Generation (RAG)-powered chatbot that retrieves and summarizes research papers on Large Language Models (LLMs) from a dataset of 100 papers. It supports text and image queries, generates concise summaries, and creates visualizations (e.g., diagrams) for topics like fine-tuning, quantization, and mixture of experts. Integrated with Ollama for local LLM execution, it uses hybrid retrieval, parameter-efficient fine-tuning, and multimodal AI, deployed as a scalable API and interactive UI.

## Features

- **Hybrid Retrieval**: Combines DPR (dense) and BM25 (sparse) for accurate paper retrieval.
- **Summarization**: Fine-tuned T5 with LoRA and Ollama’s Llama 3.1 for high-quality summaries.
- **Multimodal AI**: Supports image queries with Ollama’s Llama 3.2 Vision and generates diagrams with Stable Diffusion.
- **Local Execution**: Ollama enables privacy-preserving, offline LLM inference.
- **Error Handling**: Robust handling for CUDA errors, invalid inputs, and retrieval failures.
- **Production-Ready**: Deployed via FastAPI and accessible through Gradio.

## Tech Stack

- **Ollama**: Local LLMs (Llama 3.1 8B, Llama 3.2 Vision 11B) and embeddings (mxbai-embed-large).
- **PyTorch**: Core framework for Stable Diffusion and optimization.
- **Hugging Face**: Transformers (T5, CLIP), Diffusers (Stable Diffusion), PEFT (LoRA).
- **FAISS**: Dense retrieval for RAG.
- **BM25**: Sparse retrieval (rank_bm25).
- **LangChain**: Integrates Ollama with RAG pipeline.
- **FastAPI**: API deployment.
- **Gradio**: Interactive UI.
- **Streamlit/Matplotlib**: Metrics visualization.

## Setup

1. **Install Ollama:**
   - Download from [ollama.com](https://ollama.com).
   - Pull models:
     ```bash
     ollama pull llama3.1:8b
     ollama pull llama3.2:11b-vision
     ollama pull mxbai-embed-large
     ```

2. **Clone the Repository:**
   ```bash
   git clone https://github.com/adwaiths05/advanced-llm-paper-summarizer.git
   cd advanced-llm-paper-summarizer
   ```

3. **Create Virtual Environment:**
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```

4. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

5. **Prepare Dataset:**
   - Place `sample_papers.csv` (paper abstracts) in the root directory.
   - Or run `preprocess.py` to extract abstracts (requires PyPDF2).

6. **Run Ollama Service:**
   ```bash
   ollama serve
   ```

7. **Run Application:**
   - **FastAPI**:
     ```bash
     uvicorn app:app --host 0.0.0.0 --port 8000
     ```
   - **Gradio**:
     ```bash
     python app.py
     ```

## Usage

- **Text Query**: Input a question (e.g., "Recent advances in LLM fine-tuning") via Gradio UI or FastAPI (`/summarize?query=...`).
- **Image Query**: Upload an image (e.g., transformer diagram) to find related papers.
- **Output**: Receive a summary of relevant papers and an optional generated visualization (e.g., "mixture of experts diagram").

**Example:**
- Query: `What is quantization in LLMs?`
- Output: `Quantization reduces model size by lowering precision (e.g., 8-bit). Recent methods improve efficiency without accuracy loss.`

## Requirements

- Python 3.8+
- NVIDIA GPU (recommended for Ollama and Stable Diffusion)
- See `requirements.txt` for full dependencies.

## Acknowledgments

Built with insights from RAG (Lewis et al., 2020), LoRA (Hu et al., 2021), CLIP (Radford et al., 2021), and Stable Diffusion (Rombach et al., 2022).
Thanks to the authors of the 100 LLM papers for their contributions.
Powered by Ollama for local, privacy-preserving LLM inference.

## License

This project is licensed under the MIT License. See LICENSE for details.
