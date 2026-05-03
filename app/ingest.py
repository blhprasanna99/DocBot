import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()

DOCS_PATH = Path(os.getenv("DOCS_PATH", "docs/posthog/contents/handbook"))
CHROMA_PATH = os.getenv("CHROMA_DB", ".chroma")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

print(f"Loading docs from: {DOCS_PATH}")
print(f"Persisting Chroma to: {CHROMA_PATH}")
print(f"Embedding model: {EMBEDDING_MODEL}")

loader = DirectoryLoader(
    str(DOCS_PATH),
    glob="**/*.md",
    loader_cls=TextLoader,
    show_progress=True
)
documents = loader.load()
print(f"Loaded {len(documents)} documents")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " ", ""],
)
chunks = splitter.split_documents(documents)
print(f"Split into {len(chunks)} chunks")

print("Loading embeddings model )downloads ~90MB on first run)...")
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

print(f"Embedding {len(chunks)} chunks and writing to Chroma...")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=CHROMA_PATH,
)
print(f"Done. Chroma persisted to {CHROMA_PATH}")