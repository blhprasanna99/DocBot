from fastapi import FastAPI
from dotenv import load_dotenv
from pydantic import BaseModel
import os
from anthropic import Anthropic
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from fastapi.responses import FileResponse

load_dotenv()

client = Anthropic()
MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")

CHROMA_PATH = os.getenv("CHROMA_PATH",".chroma")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2")
TOP_K = int(os.getenv("TOP_K", "4"))

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
vectorstore = Chroma(
    persist_directory=CHROMA_PATH,
    embedding_function=embeddings,
)

class ChatRequest(BaseModel):
    message: str

app = FastAPI(title="DocBot", version="0.1.0")

@app.get("/health")
def health():
    return {"ok":True}

@app.get("/")
def index():
    return FileResponse("static/index.html")

@app.post("/chat")
def chat(req: ChatRequest):

    docs = vectorstore.similarity_search(req.message, k=TOP_K)

    context = "\n\n---\n\n".join(
        f"[Source: {d.metadata.get('source', 'unknown')}]\n{d.page_content}"
        for d in docs
    )

    sources = sorted({d.metadata.get("source", "unknown") for d in docs})
 
    system_prompt = (
        "You are DocBot, an assistant that answers questions about internal "
        "company documentation. Answer using ONLY the provided context. "
        "If the answers isn't in the context, say you don't know. Be concise."
    )

    user_prompt = f"Context:\n\n{context}\nQuestion: {req.message}"


    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return {
        "reply": response.content[0].text,
        "sources": sources,
        }
