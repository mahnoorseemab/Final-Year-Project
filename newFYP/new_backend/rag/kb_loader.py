from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

def load_knowledge_base():
    KB_PATH = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "knowledge_base",
        "policies_knowledge_base.txt"
    )
    loader = TextLoader(KB_PATH, encoding="utf-8")
    documents = loader.load()
    print("Knowledge base loaded successfully!")
    return documents

def split_into_chunks(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)
    print(f"Total chunks created: {len(chunks)}")
    return chunks

def create_faiss_index(chunks):
    print("Creating embeddings, please wait...")
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.from_documents(chunks, embedding_model)
    vectorstore.save_local("rag/faiss_index")
    print("FAISS index created and saved successfully!")
    return vectorstore

if __name__ == "__main__":
    docs = load_knowledge_base()
    chunks = split_into_chunks(docs)
    vectorstore = create_faiss_index(chunks)
    print("Knowledge base is ready for RAG!")