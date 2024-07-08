import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.vectorstores import FAISS  
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

st.title("Gemma Model Document QnA")

llm = ChatGroq(model_name="Gemma-7b-it")

prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    <context>
    Questions: {input}
    """
)

def vector_embedding():
  if "vectors" not in st.session_state:
    st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")  # Corrected model name
    st.session_state.loader = PyPDFDirectoryLoader("./user_data")
    st.session_state.docs = st.session_state.loader.load()
    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
    st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

prompt1 = st.text_input("Enter the question you want to ask.")

if st.button("Create Vector Store"):
  vector_embedding()
  st.write("Vector Store Database is ready")

if prompt1:
  document_chain = create_stuff_documents_chain(llm, prompt)
  if "vectors" in st.session_state:
    retriever = st.session_state.vectors.as_retriever()
    retriever_chain = create_retrieval_chain(retriever, document_chain)
    response = retriever_chain.invoke({'input': prompt1})
    st.write(response['answer'])

    with st.expander("Document Similarity Search"):
      for i, doc in enumerate(response['context']):
        st.write(doc.page_content)
        st.write("-----------------------------")
  else:
    st.write("Vector store not initialized. Please create the vector store first.")
