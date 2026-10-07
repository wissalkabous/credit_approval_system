import os
import chromadb
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer


load_dotenv()

client_groq = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Connect to existing ChromaDB
client = chromadb.PersistentClient(path="chatbot/chroma_db")

collection = client.get_collection("bank_faq")


# Create the retrieval function

def retrieve_context(question, n_results=3):  # FAQ chunks (default = 3).
    question_embedding = embedding_model.encode([question])[0]  # turns the question into an embedding vector.

    results = collection.query(  # searches ChromaDB for the closest embeddings (semantic similarity).
        query_embeddings=[question_embedding.tolist()],
        n_results=n_results
    )

    return results["documents"][0]


def generate_answer(question, context):
    context_text = "\n\n".join(context)

    prompt = f"""
You are a helpful banking assistant.

Answer the user's question using only the information provided
in the knowledge base below.

Knowledge base:
{context_text}

User question:
{question}

If the knowledge base does not contain enough information to answer
the question, say that you do not have enough information.
"""

    response = client_groq.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content


def ask_chatbot(question):
    context = retrieve_context(question)
    answer = generate_answer(question, context)

    return answer


# “question → embedding → ChromaDB search → retrieve FAQ chunks”

if __name__ == "__main__":
    print(f"Knowledge base loaded: {collection.count()} documents")
    print(ask_chatbot("What documents do I need for a personal loan?"))