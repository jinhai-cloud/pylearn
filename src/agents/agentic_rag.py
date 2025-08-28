from typing import List

from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_core.tools import create_retriever_tool, Tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

def create_chunk() -> List[Document]:
    urls = [
        "https://lilianweng.github.io/posts/2024-11-28-reward-hacking/",
        "https://lilianweng.github.io/posts/2024-07-07-hallucination/",
        "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/",
    ]
    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=100, chunk_overlap=50
    )
    doc_splits = text_splitter.split_documents(docs_list)
    return doc_splits

def create_retriever(doc_splits: List[Document]) -> Tool:
    vectorstore = InMemoryVectorStore.from_documents(
        documents=doc_splits, embedding=OpenAIEmbeddings()
    )
    retriever = vectorstore.as_retriever()
    retriever_tool = create_retriever_tool(
        retriever,
        "retrieve_blog_posts",
        "Search and return information about Lilian Weng blog posts.",
    )
    return retriever_tool

if __name__ == '__main__':
    load_dotenv()
    docs = create_chunk()
    retriever_tool = create_retriever(docs)
    print(retriever_tool.invoke({"query": "types of reward hacking"}))