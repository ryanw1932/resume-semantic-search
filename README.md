# resume-semantic-search
Semantic search engine for PDF resumes using LangChain, ChromaDB, and local LLMs (Ollama)

This project implements a full pipeline for semantic document retrieval:

## Pipeline
- Parses PDF resumes  
- Converts text into vector embeddings  
- Stores embeddings in ChromaDB  
- Performs semantic similarity search  
- Uses local LLM inference (Ollama / Llama 3)  

Designed to simulate real-world search systems used in recruiting, knowledge retrieval, and recommendation engines. Good for data privacy too. 

Note: Update the PDF input file to use your own resume.

## How to Run
1. Install dependencies:
pip install -r requirements.txt

2. Make sure Ollama is running:
ollama run llama3

3. Add your PDF resume:
- Place it in the project folder
- Update the file path in app.py if needed

4. Run the app:
python app.py
