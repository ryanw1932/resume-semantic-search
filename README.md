# resume-semantic-search
Semantic search engine for PDF resumes using LangChain, ChromaDB, and local LLMs (Ollama)

This project implements a full pipeline for semantic document retrieval:

Parses PDF resumes
Converts text into vector embeddings
Stores embeddings in a vector database (ChromaDB)
Performs similarity search based on user queries
Uses local LLM inference (Ollama / Llama 3) to avoid external APIs

Designed to simulate real-world search systems used in recruiting, knowledge retrieval, and recommendation engines.
## Setup
pip install -r requirements.txt
python main.py
