import sys, os
sys.dont_write_bytecode = True

import time
from dotenv import load_dotenv

import pandas as pd
import streamlit as st
from streamlit_modal import Modal

from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.faiss import DistanceStrategy
from langchain_community.embeddings import HuggingFaceEmbeddings

from llm_agent import ChatBot
from ingest_data import ingest
from retriever import SelfQueryRetriever
import chatbot_verbosity as chatbot_verbosity

load_dotenv()

DATA_PATH = os.getenv("DATA_PATH")
FAISS_PATH = os.getenv("FAISS_PATH")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
LLM_MODEL = "llama3.2:3b"

print(DATA_PATH)
print(FAISS_PATH)

welcome_message = """

#### Introduction 🚀

The system is a RAG pipeline designed to assist hiring managers in searching for the most suitable candidates out of thousands of resumes more effectively. ⚡

The idea is to use a similarity retriever to identify the most suitable applicants with job descriptions.
This data is then augmented into an LLM generator for downstream tasks such as analysis, summarization, and decision-making.

#### Getting started 🛠️

1. The system uses a local LLM through Ollama. 🦙
2. Type in a job description query. 💬

Hint: The knowledge base has been loaded with a pre-existing vectorstore of [resumes](https://github.com/Fardeen627/ai-resume-screening/blob/main/data/main-data/synthetic-resumes.csv) to be used right away.

You may also find example job descriptions to test [here](https://github.com/Fardeen627/ai-resume-screening/blob/main/data/supplementary-data/job_title_des.csv).

Please make sure to check the sidebar for more useful information. 💡
"""

info_message = """

# Information

### 1. What if I want to use my own resumes?

If you want to load your own resumes file, simply use the uploading button above.
Please make sure to have the following column names: `Resume` and `ID`.

Keep in mind that the indexing process can take **quite some time** to complete. ⌛

### 2. What if I want to set your own parameters?

The application uses the local `llama3.2:3b` model through Ollama.
The RAG mode can be changed using the sidebar.

### 3. Is my uploaded data safe?

Uploaded resume data is processed locally by this application.
The data is not sent to OpenAI.

### 4. How does the chatbot work?

The chatbot retrieves relevant resumes using semantic similarity search.
RAG Fusion can generate multiple focused queries and combine their retrieval results.
The retrieved resumes are then provided to the local LLM as context for analysis and summarization.
"""

about_message = """

# About

This project demonstrates a RAG-based approach to resume screening using semantic retrieval and a local LLM.

The system combines FAISS vector search, sentence-transformer embeddings, RAG Fusion, and a local language model to assist with resume analysis.

The project is a work in progress. Contributions and feedback are welcome on [GitHub](https://github.com/Fardeen627/ai-resume-screening).
"""

st.set_page_config(page_title="Resume Screening AI")
st.title("Resume Screening AI")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [AIMessage(content=welcome_message)]

if "df" not in st.session_state:
    st.session_state.df = pd.read_csv(DATA_PATH)

if "embedding_model" not in st.session_state:
    st.session_state.embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"}
    )

if "rag_pipeline" not in st.session_state:
    vectordb = FAISS.load_local(
        FAISS_PATH,
        st.session_state.embedding_model,
        distance_strategy=DistanceStrategy.COSINE,
        allow_dangerous_deserialization=True
    )
    st.session_state.rag_pipeline = SelfQueryRetriever(
        vectordb,
        st.session_state.df
    )

if "resume_list" not in st.session_state:
    st.session_state.resume_list = []


def upload_file():
    modal = Modal(key="Demo Key", title="File Error", max_width=500)

    if st.session_state.uploaded_file is None:
        st.session_state.df = pd.read_csv(DATA_PATH)

        vectordb = FAISS.load_local(
            FAISS_PATH,
            st.session_state.embedding_model,
            distance_strategy=DistanceStrategy.COSINE,
            allow_dangerous_deserialization=True
        )

        st.session_state.rag_pipeline = SelfQueryRetriever(
            vectordb,
            st.session_state.df
        )
        return

    try:
        df_load = pd.read_csv(st.session_state.uploaded_file)
    except Exception as error:
        with modal.container():
            st.markdown(
                "The uploaded file returns the following error message. Please check your CSV file again."
            )
            st.error(error)
        return

    if "Resume" not in df_load.columns or "ID" not in df_load.columns:
        with modal.container():
            st.error('Please include the following columns in your data: "Resume", "ID".')
        return

    with st.toast("Indexing the uploaded data. This may take a while..."):
        st.session_state.df = df_load

        vectordb = ingest(
            st.session_state.df,
            "Resume",
            st.session_state.embedding_model
        )

        st.session_state.rag_pipeline = SelfQueryRetriever(
            vectordb,
            st.session_state.df
        )


def clear_message():
    st.session_state.resume_list = []
    st.session_state.chat_history = [AIMessage(content=welcome_message)]


user_query = st.chat_input("Type your message here...")

with st.sidebar:
    st.markdown("# Control Panel")

    st.selectbox(
        "RAG Mode",
        ["Generic RAG", "RAG Fusion"],
        key="rag_selection"
    )

    st.text_input(
        "Local LLM",
        LLM_MODEL,
        disabled=True
    )

    st.file_uploader(
        "Upload resumes",
        type=["csv"],
        key="uploaded_file",
        on_change=upload_file
    )

    st.button(
        "Clear conversation",
        on_click=clear_message
    )

    st.divider()
    st.markdown(info_message)

    st.divider()
    st.markdown(about_message)

    st.markdown(
        "Made by [Fardeen Akhtar](https://github.com/Fardeen627)"
    )


for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)

    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)

    else:
        with st.chat_message("AI"):
            message[0].render(*message[1:])


retriever = st.session_state.rag_pipeline

llm = ChatBot(
    model=LLM_MODEL
)


if user_query is not None and user_query != "":
    with st.chat_message("Human"):
        st.markdown(user_query)

    st.session_state.chat_history.append(
        HumanMessage(content=user_query)
    )

    with st.chat_message("AI"):
        start = time.time()

        with st.spinner("Generating answers..."):
            document_list = retriever.retrieve_docs(
                user_query,
                llm,
                st.session_state.rag_selection
            )

            query_type = retriever.meta_data["query_type"]

            st.session_state.resume_list = document_list

            stream_message = llm.generate_message_stream(
                user_query,
                document_list,
                [],
                query_type
            )

            response = st.write_stream(stream_message)

        end = time.time()

        retriever_message = chatbot_verbosity

        retriever_message.render(
            document_list,
            retriever.meta_data,
            end - start
        )

        st.session_state.chat_history.append(
            AIMessage(content=response)
        )

        st.session_state.chat_history.append(
            (
                retriever_message,
                document_list,
                retriever.meta_data,
                end - start
            )
        )