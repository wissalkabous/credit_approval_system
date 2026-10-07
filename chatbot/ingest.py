from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb

faq_path = Path("data/bank_faq/bank_faq.txt")

text = faq_path.read_text(encoding="utf-8") #loads the entire file as a Python string.

print(text[:500])

chunks = []

lines = text.splitlines()

for i, line in enumerate(lines):
    if line.startswith("Q:"):
        question = line
        answer = lines[i + 1]
        chunks.append(question + "\n" + answer)

print(f"Number of chunks: {len(chunks)}")
print("\nFirst chunk:")
print(chunks[0])

#turn each text chunk into a vector of numbers

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")# pre-trained embedding model
embeddings = embedding_model.encode(chunks)

print(f"Number of embeddings: {len(embeddings)}")
print(f"Embedding size: {len(embeddings[0])}") # Embedding: [0.12, -0.45, 0.87, …] (384 values)

#Create the ChromaDB collection
client = chromadb.PersistentClient(path="chatbot/chroma_db") #DB will be saved on disk at the path "/chroma_db"

try:
    client.delete_collection("bank_faq")
except Exception:
    pass

collection = client.get_or_create_collection(
    name="bank_faq"
)

collection.add(
    documents=chunks,
    embeddings=embeddings.tolist(),
    ids=[f"faq_{i}" for i in range(len(chunks))] # unique IDs for each chunk (faq_0)
)

print(f"Stored documents: {collection.count()}")

# Test retrieval:
results = collection.query(
    query_texts=["What credit score do I need for a loan?"],
    n_results=2
)

print("\nRetrieved chunks:")
for result in results["documents"][0]:
    print("\n", result)