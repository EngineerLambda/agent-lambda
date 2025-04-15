import streamlit as st
from langchain.tools import tool
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain.schema import Document
from typing import List

def create_rag_tool(documents: List[Document], embeddings_model_name: str = "models/embedding-001"):
    """
    Creates a Langchain retrieval tool from a list of Document objects.

    Args:
        documents (List[Document]): The list of documents loaded from the file.
        embeddings_model_name (str): The name of the embedding model to use.

    Returns:
        BaseRetrieverTool: The configured Langchain retrieval tool, or None if error.
    """
    if not documents:
        st.warning("No documents provided to create RAG tool.")
        return None

    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs_split = text_splitter.split_documents(documents)

        if not docs_split:
            st.warning("Document content was empty after splitting.")
            return None

        embedding = GoogleGenerativeAIEmbeddings(model=embeddings_model_name)

        vectorstore = FAISS.from_documents(docs_split, embedding)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

        retrieval_tool = create_retriever_tool(
            retriever=retriever,
            name="document_search",
            description="Use this tool *only* to answer questions about the content of the uploaded document. Pass the user's question directly as input to the tool."
        )
        print("RAG tool created successfully.")
        return retrieval_tool

    except Exception as e:
        st.error(f"Failed to create RAG tool: {e}")
        print(f"Error creating RAG tool: {e}")
        return None