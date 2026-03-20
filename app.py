import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate


PDF_PATH = "Resume.pdf"
DB_DIR = "./chroma_db"


def build_vector_store(pdf_path: str, db_dir: str) -> Chroma:
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )
    chunks = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    db = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=db_dir
    )
    return db


def answer_questions(db: Chroma) -> None:
    llm = ChatOllama(model="llama3")

    template = """
Answer the question based ONLY on the following context from Ryan's resume:

{context}

Question: {question}
"""
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm

    while True:
        query = input("\nQuestion (type 'exit' to quit): ").strip()
        if query.lower() == "exit":
            break

        docs = db.similarity_search(query, k=3)
        context = "\n".join([doc.page_content for doc in docs])

        response = chain.invoke({
            "context": context,
            "question": query
        })

        print("\n--- AI ANSWER ---")
        print(response.content)


def main() -> None:
    if not os.path.exists(PDF_PATH):
        print(f"Error: could not find {PDF_PATH}")
        return

    db = build_vector_store(PDF_PATH, DB_DIR)
    answer_questions(db)


if __name__ == "__main__":
    main()