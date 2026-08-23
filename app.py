import os
from dotenv import load_dotenv
import gradio as gr
from pypdf import PdfReader

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

load_dotenv()

# ---- Prompt template ----
prompt_template = """
You are a Revolut fees and terms assistant. Use the following pieces of context to answer the question at the end.

Guidelines:
- Only use information from the provided context. If the answer isn't in the context, say you don't have that information in the document provided, don't try to make up an answer.
- Be precise with numbers: fees, percentages, limits, and thresholds must match the context exactly.
- Format your answer in clean markdown: use headers, bullet points, or tables where they make the answer easier to scan.
- When a fee depends on the plan (Standard, Plus, Premium, Metal, Ultra) or on conditions, clearly state which plan or condition each figure applies to.
- If a question falls outside what this document covers, respond professionally: "I don't have that information in the document provided. I'd recommend checking directly with Revolut's official terms page."
- Maintain a professional, neutral tone throughout.
- Never guess at a fee, percentage, or limit.

Context:
{context}

Question: {question}

Answer (in markdown):
"""
QA_PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# ---- functions ----
def document_loader(filename):
    reader = PdfReader(filename)
    contents = "".join(page.extract_text() for page in reader.pages)
    with open("internal_policy.txt", "w", encoding="utf-8") as f:
        f.write(contents)
    return TextLoader("internal_policy.txt").load()

def text_splitter(documents):
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=150, separator="\n")
    return splitter.split_documents(documents)

def vector_database(chunks):
    embeddings = HuggingFaceEmbeddings(model_kwargs={"device": "cpu"})
    return Chroma.from_documents(chunks, embeddings)

def build_vectordb(file):
    documents = document_loader(file)
    chunks = text_splitter(documents)
    return vector_database(chunks)
api_key=os.getenv("GROQ_API_KEY")
def get_llm():
    return ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0.5,
    )

# ---- built ONCE, not per-call ----
vectordb = build_vectordb("internal_policy.pdf")
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer",
)
qa = ConversationalRetrievalChain.from_llm(
    llm=get_llm(),
    chain_type="stuff",
    retriever=vectordb.as_retriever(search_kwargs={"k": 6}),
    memory=memory,
    combine_docs_chain_kwargs={"prompt": QA_PROMPT},
    return_source_documents=True,
)

def retriever_qa(query):
    response = qa.invoke({"question": query})
    return response["answer"]

# ---- Gradio Interface ----
demo = gr.Interface(
    fn=retriever_qa,
    inputs=[gr.Textbox(placeholder="Write a message...")],
    outputs=gr.Markdown(container=True, max_height=400),
    examples=[
        "What ATM withdrawal fees apply on the Standard plan?",
        "What's the exchange fee if I go over my monthly limit?",
        "How much does a replacement card cost?",
    ],
    title="Revolut Fees Assistant",
    description="A RAG-powered assistant for Revolut's personal account terms and fees. Ask about card fees, ATM withdrawals, currency exchange limits, or international payments, and get answers grounded in the actual fee document.",
    flagging_mode="auto",
)

demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
