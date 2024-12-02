from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore

from pinecone import Pinecone
from pinecone import ServerlessSpec

from configuracion import PINECONE_NAMESPACE_NAME, EMBEDDINGS_MODEL

import re

from typing import Dict

NAME_PATTER = r'^.*CV (.*)\.pdf$'


def load_documents(path: str, pinecone_api: Pinecone, cloud: str = "aws",
                   region: str = "us-east-1", force_load: bool = False) -> Dict[str, str]:

    documents = PyPDFDirectoryLoader(path).load()

    files_names = list(set([document.metadata['source']
                       for document in documents]))

    documents_chunks = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=50).split_documents(documents)

    existing_indexes = [
        index_info["name"] for index_info in pinecone_api.list_indexes()
    ]

    for file_name in files_names:
        name_search_results = re.search(NAME_PATTER, file_name)

        if not name_search_results:
            raise ValueError('Incorrect filename')

        index_name = name_search_results.group(1).replace(' ', '-').lower()

        if index_name in existing_indexes:
            if force_load:
                pinecone_api.delete_index(index_name)
            else:
                continue

        print("Creating index name:", index_name)

        pinecone_api.create_index(
            index_name, dimension=384, metric='cosine',
            spec=ServerlessSpec(cloud=cloud, region=region))

        file_chunks = filter(
            lambda chk: chk.metadata['source'] == file_name, documents_chunks)

        PineconeVectorStore.from_documents(
            documents=file_chunks,
            index_name=index_name,
            embedding=EMBEDDINGS_MODEL,
            namespace=PINECONE_NAMESPACE_NAME)
