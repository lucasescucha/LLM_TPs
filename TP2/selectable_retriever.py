from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document

from typing import Dict

from langchain_pinecone import PineconeVectorStore

from configuracion import PINECONE_API_KEY, PINECONE_NAMESPACE_NAME, EMBEDDINGS_MODEL, DEVICE
from pinecone import Pinecone

from pydantic import BaseModel, PrivateAttr

class SelectableRetriever(BaseRetriever, BaseModel):
    _retrievers: Dict[str, BaseRetriever] = PrivateAttr({})

    def __init__(self, pinecone_api: Pinecone) -> object:
        super().__init__()

        indexes = [index_info["name"]
                   for index_info in pinecone_api.list_indexes()]

        for index in indexes:
            vector_store = PineconeVectorStore(
                pinecone_api_key=PINECONE_API_KEY,
                index_name=index,
                embedding=EMBEDDINGS_MODEL,
                namespace=PINECONE_NAMESPACE_NAME,
            )

            self._retrievers[index] = vector_store.as_retriever()

    def _get_relevant_documents(self, query: str, *, run_manager) -> list[Document]:
        def get_person_name_from_prompt(prompt: str) -> str:
            prompt_lower = prompt.lower()
            if 'lucas' in prompt_lower or 'barrera' in prompt_lower:
                return "lucas-barrera"
            elif 'gloria' in prompt_lower or 'gonzalez' in prompt_lower or 'gonzález' in prompt_lower:
                return "gloria-gonzalez"
            else:
                raise KeyError()

        person_name = get_person_name_from_prompt(query)
        print("Retrieving documents for", person_name)
        return self._retrievers[person_name]._get_relevant_documents(query=query, run_manager=run_manager)

    def get_loaded_retrievers(self) -> Dict[str, BaseRetriever]:
        return self._retrievers
