import os

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

PINECONE_INDEX_NAME = 'cvs-index'
PINECONE_NAMESPACE_NAME = "default"