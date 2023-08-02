import os

from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import CTransformers
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

import streamlit as st

st.title("Welcome to Paper Chat App")
st.write("This is a simple app for demonstration purposes.")


# if os.path.exists("data/"):
#     st.write("Paper inside 'data' folder:")
#     files = os.listdir("data/")

#     # Print each file name

#     for file in files:
#         if file == ".DS_Store":
#             continue
#         st.write(file)

# Load PDF file from data path
loader = DirectoryLoader("data/", glob="*.pdf", loader_cls=PyPDFLoader)
documents = loader.load()
# Split text from PDF into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=50
)
texts = text_splitter.split_documents(documents)
# Load embeddings model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
)

# Build and persist FAISS vector store
vectorstore = FAISS.from_documents(texts, embeddings)
vectorstore.save_local("vectorstore/db_faiss")


llm = CTransformers(
    model="models/llama-2-7b-chat.ggmlv3.q8_0.bin",  # Location of downloaded GGML model
    model_type="llama",  # Model type Llama
    config={
        "max_new_tokens": 1000,
        "temperature": 0.01,
        "repetition_penalty": 2,
    },
)

qa_template = """
You're helping machine learning researcher to understand paper. \
Use the following pieces of information to answer the user's question. \
Context: {context}
Question: {question}

Please pay attention to the context and the question provided. \
If the given context is helpful, please provide the answer based on that context. \
If the context is not sufficient to answer the question or is not convincing enough, \
you must mention the paper title and then provide a helpful answer.
Make sure not to include citation numbers such as [15] or [54] in your response.

Please pay attention to keeping your answer short but clear. \
If you don't know the answer, just say that you don't know, don't try to make up an answer. \
Only provide the helpful answer below and nothing else.
Don't repeat your answer.
Helpful answer:
"""


def set_qa_prompt():
    prompt = PromptTemplate(
        template=qa_template, input_variables=["context", "question"]
    )
    return prompt


# Build RetrievalQA object
def build_retrieval_qa(llm, prompt, vectordb):
    dbqa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectordb.as_retriever(search_kwargs={"k": 2}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )
    return dbqa


# Instantiate QA object
def setup_dbqa():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )
    vectordb = FAISS.load_local("vectorstore/db_faiss", embeddings)
    qa_prompt = set_qa_prompt()
    dbqa = build_retrieval_qa(llm, qa_prompt, vectordb)

    return dbqa


dbqa = setup_dbqa()


# Receive input text
st.write("## Input Question")
user_input = st.text_area(
    "Enter your text here", "What is positional encoding in transformer?"
)
st.info(f"Your Question: {user_input}")


response = dbqa({"query": user_input})


st.write(f"""LLM Answer: {response["result"]}""")
source_docs = response["source_documents"]
for i, doc in enumerate(source_docs):
    st.write(f"Source Text: {doc.page_content}")
    st.write(f'Document Name: {doc.metadata["source"]}')
    st.write(f'Page Number: {doc.metadata["page"]}\n')
