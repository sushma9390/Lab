import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_ollama import OllamaEmbeddings

load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-20b"
)

DATABASE_URL = os.getenv("DATABASE_URL")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "rag_M3"
EMBEDDING_MODEL = "sentence-transformers/all-minilm-l6-v2"
