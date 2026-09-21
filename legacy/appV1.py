import os
from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_classic.retrievers.multi_query import MultiQueryRetriever



from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

import discord
from discord.ext import commands


def document_loader():
    """
    upload pdf files as documents

    Args:
        **Input**: None
        **Output**: Documents
    """
    print("-"*5,"Start loading", "-"*5)
    loader = DirectoryLoader("N26",
                             loader_cls = PyMuPDFLoader,
                             show_progress= True,
                             exclude=["*Zone.Identifier"])
    documents = loader.load()
    for doc in documents:
        doc.metadata["bank"] = "N26"   # tag every chunk with its bank
    print("-"*5,"End loading", "-"*5)
    return documents

def text_splitter(documents):
    """
    Splitting the documents
    Args:
        **Input**: documents
        **Output**: Chunks
    """
    splitter = RecursiveCharacterTextSplitter( separators = ['\n\n', '\n'], chunk_size = 500, chunk_overlap = 500)
    return splitter.split_documents(documents)


def vector_database(chunks):
    embeddings = HuggingFaceEmbeddings(model_kwargs={"device": "cpu"})
    return Chroma.from_documents(chunks, embeddings)

prompt_template = """
You are an assistant for online bank account fees and terms. Use the following pieces of context to answer the question at the end.

Guidelines:
- Only use information from the provided context. If the answer isn't in the context, say you don't have that information in the documents provided, don't try to make up an answer.
- Every fee, limit, or rule you state must be attributed to the specific bank it applies to (e.g. "N26 charges...", "According to Revolut..."). Never state a fact without naming which bank's document it came from.
- Be precise with numbers: fees, percentages, limits, and thresholds must match the context exactly.
- Format your answer in clean markdown: use headers, bullet points, or tables where they make the answer easier to scan, especially when comparing banks or plans.
- When a fee depends on the plan or on conditions, clearly state which plan or condition each figure applies to.
- If asked about a bank not present in the context, say so clearly rather than guessing or mixing it up with another bank.
- Maintain a professional, neutral tone throughout.
- Never guess at a fee, percentage, or limit.

Context:
{context}

Question: {question}

Answer (in markdown):
"""
QA_PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

query_prompt_template = """
You are an AI language model assistant. Your task is to generate as many different versions of the given user query as possible to retrieve relevant documents from a vector database. 
By generating multiple perspectives on the user query, your goal is to help the user overcome some of the limitations of distance-based similarity search. 
Provide these alternative queries separated by newlines.
Original query: {question}
"""
QUERY_PROMPT = PromptTemplate(template=query_prompt_template, input_variables=["question"])


def build_vectordb():
    documents = document_loader()
    chunks = text_splitter(documents)
    return vector_database(chunks)
api_key=os.getenv("GROQ_API_KEY")
def get_llm():
    return ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0.5,
    )

vectordb = build_vectordb()
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer",
)
retriever1 = MultiQueryRetriever.from_llm(llm = get_llm(), retriever = vectordb.as_retriever(), prompt= QUERY_PROMPT)
qa = ConversationalRetrievalChain.from_llm( 
    llm=get_llm(),
    chain_type="stuff",
    retriever=retriever1, #vectordb.as_retriever(search_kwargs={"k": 6}),
    memory=memory,
    combine_docs_chain_kwargs={"prompt": QA_PROMPT},
    return_source_documents=True)

def retriever_qa(query):
    response = qa.invoke({"question": query})
    answer = response["answer"]
    sources = response["source_documents"]
    if sources:
        sources_md = "\n\n---\n**Sources**\n"
        seen = set()
        for doc in sources:
            bank = doc.metadata.get("bank", "Unknown bank")
            filename = doc.metadata.get("source", "unknown file")
            page = doc.metadata.get("page", None)
            page_label = f", page {page + 1}" if page is not None else ""
            key = (bank, filename, page)
            if key in seen:
                continue
            seen.add(key)
            sources_md += f"- **{bank}** — {filename}{page_label}\n"
        answer += sources_md
    return answer

if __name__ == "__main__":
    while True:
        x = input()
        answer = retriever_qa(x)
        print(answer)