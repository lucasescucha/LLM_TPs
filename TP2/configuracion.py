import os

from langchain_huggingface import HuggingFaceEmbeddings

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

PINECONE_INDEX_NAME = 'cvs-index'
PINECONE_NAMESPACE_NAME = "default"

DEVICE = 'cuda'
EMBEDDINGS_MODEL = HuggingFaceEmbeddings(
    model_name='sentence-transformers/all-MiniLM-L6-v2',
    model_kwargs={'device': DEVICE},
    encode_kwargs={'normalize_embeddings': False})