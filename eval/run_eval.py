import os
import time
import yaml
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()

CHROMA_PATH = os.getenv("CHROMA_PATH", ".chroma")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2")
TOP_K = int(os.getenv("TOP_K", "8"))

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

with open("eval/questions.yaml") as f:
    questions = yaml.safe_load(f)

hits = 0
total_latency_ms = 0.0
results = []

print(f"\nRunning {len(questions)} eval questions (TOP_K={TOP_K})\n")
print(f"{'#':>3}  {'result':<6}  {'latency':>9}  question")
print("-" * 90)

for i, item in enumerate(questions, 1):
    question = item["question"]
    expected = item["expected_source"]

    start = time.perf_counter()
    docs = vectorstore.similarity_search(question, k=TOP_K)
    elapsed_ms = (time.perf_counter() - start) * 1000
    total_latency_ms += elapsed_ms

    retrieved = [d.metadata.get("source", "") for d in docs]
    hit = any(r.endswith(expected) for r in retrieved)
    if hit:
        hits += 1
    results.append((i, question, expected, hit, elapsed_ms, retrieved))

    print(f"{i:>3}  {'hit' if hit else 'miss':<6}  {elapsed_ms:>7.1f}ms  {question}")

print("-" * 90)
print(f"\nhit@{TOP_K}: {hits}/{len(questions)} = {hits / len(questions) * 100:.1f}%")
print(f"avg latency: {total_latency_ms / len(questions):.1f}ms\n")

misses = [r for r in results if not r[3]]
if misses:
    print("MISSES (what we got instead):")
    for i, q, exp, _, _, retrieved in misses:
        print(f"  [{i}] {q}")
        print(f"      expected: {exp}")
        for r in retrieved[:3]:
            print(f"      got:      {r}")
        print()
