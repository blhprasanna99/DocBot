from fastapi import FastAPI
from dotenv import load_dotenv
from pydantic import BaseModel
import os
from anthropic import Anthropic

load_dotenv()

client = Anthropic()
MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")

class ChatRequest(BaseModel):
    message: str

app = FastAPI(title="DocBot", version="0.1.0")

@app.get("/health")
def health():
    return {"ok":True}

@app.post("/chat")
def chat(req: ChatRequest):
    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": req.message}],
    )
    return {"reply": response.content[0].text}
