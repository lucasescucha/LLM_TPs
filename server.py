from random import randint

from flask import Flask, render_template, request, session
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import MessagesPlaceholder

from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from langchain_huggingface import HuggingFaceEmbeddings

import configuracion

DEVICE = 'cuda'
CHAT_MODEL = 'llama3-8b-8192'

application = Flask(__name__)

groq_chat = ChatGroq(
    groq_api_key=configuracion.GROQ_API_KEY,
    model_name=CHAT_MODEL
)

embed_model = HuggingFaceEmbeddings(
    model_name='sentence-transformers/all-MiniLM-L6-v2',
    model_kwargs={'device': DEVICE},
    encode_kwargs={'normalize_embeddings': False}
)

vector_store = PineconeVectorStore(
    pinecone_api_key=configuracion.PINECONE_API_KEY,
    index_name=configuracion.PINECONE_INDEX_NAME,
    embedding=embed_model,
    namespace=configuracion.PINECONE_NAMESPACE_NAME,
)

retriever = vector_store.as_retriever()

system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. Text peaces are in spanish."
    "If you don't know the answer, say that you don't know."
    "Use three sentences maximum and keep the answer concise."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(groq_chat, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    print("session_id", session_id)
    return store[session_id]


conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)


def get_completion(usr_txt, session_id):
    conversational_rag_chain.invoke(
        {"input": usr_txt},
        config={
            "configurable": {"session_id": session_id}
        },
    )["answer"]
    last_message = store[session_id].messages[-1].content
    return last_message


@application.route("/")
def home():
    if 'session_id' not in session:
        session['session_id'] = randint(0, 9999)
    return render_template("./index.html")


@application.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    session_id = session.get('session_id')
    return get_completion(userText, session_id)


if __name__ == "__main__":
    application.secret_key = 'super secret key'

    application.debug = True
    application.run()
