
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from .config import (
    LLM_TEMPERATURE,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
)
from .retriever import retrieve_relevant_documents


def format_docs(docs: list[Document]) -> str:
    """Format retrieved documents into a single text context block"""
    return "\n\n".join(doc.page_content for doc in docs)


def build_rag_chain():
    """Build and return a LangChain RAG pipeline"""
    llm = ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=LLM_TEMPERATURE,
    )

    prompt_template = """Answer the question based only on the following context:

Context:
{context}

Question: {question}

Answer:"""

    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = prompt | llm | StrOutputParser()
    return chain


def generate_answer(query: str) -> dict:
    """Execute the complete RAG pipeline for a given query"""
    docs = retrieve_relevant_documents(query)
    context = format_docs(docs)
    
    chain = build_rag_chain()
    response = chain.invoke({"context": context, "question": query})
    
    return {
        "query": query,
        "answer": response,
        "retrieved_docs": docs,
    }

