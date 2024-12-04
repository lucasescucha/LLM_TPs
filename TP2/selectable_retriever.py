from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document

from typing import Dict

from langchain_pinecone import PineconeVectorStore

from configuracion import PINECONE_API_KEY, PINECONE_NAMESPACE_NAME, EMBEDDINGS_MODEL, DEVICE
from pinecone import Pinecone

from pydantic import BaseModel, PrivateAttr


class SelectableRetriever(BaseRetriever, BaseModel):
    _retrievers: Dict[str, BaseRetriever] = PrivateAttr({})

    @staticmethod
    def __create_vector_store(index: str) -> PineconeVectorStore:
        return PineconeVectorStore(
            pinecone_api_key=PINECONE_API_KEY,
            index_name=index,
            embedding=EMBEDDINGS_MODEL,
            namespace=PINECONE_NAMESPACE_NAME,
        )

    @staticmethod
    def __get_index_from_prompt(prompt: str) -> str:
        prompt_lower = prompt.lower()
        if 'lucas' in prompt_lower or 'barrera' in prompt_lower:
            return "lucas-barrera"
        elif 'gloria' in prompt_lower or 'gonzalez' in prompt_lower or 'gonzález' in prompt_lower:
            return "gloria-gonzalez"
        else:
            raise KeyError()

    def __init__(self, pinecone_api: Pinecone) -> object:
        super().__init__()

        for index in pinecone_api.list_indexes():
            index_name = index["name"]
            vector_store = SelectableRetriever.__create_vector_store(index_name)
            self._retrievers[index_name] = vector_store.as_retriever()

    def _get_relevant_documents(self, query: str, *, run_manager) -> list[Document]:
        index_name = SelectableRetriever.__get_index_from_prompt(query)
        print("Retrieving documents for", index_name)
        return self._retrievers[index_name]._get_relevant_documents(
            query=query, run_manager=run_manager)
