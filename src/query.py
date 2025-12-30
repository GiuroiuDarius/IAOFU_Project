import chromadb
import requests
import os
import dotenv
dotenv.load_dotenv()
from google import genai
from sentence_transformers import CrossEncoder

# def query_chromadb(question: str, n_results: int = 3):

#     database = chromadb.PersistentClient(path="chroma_db")
#     collection = database.get_collection("test_fin")

#     results = collection.query(
#         query_texts=[question],
#         n_results=n_results
#     )

#     relevant_chunks = []
#     for doc in results.get("documents", [[]])[0]:
#         relevant_chunks.append(doc)
#     return relevant_chunks


# Load reranker model
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L6-v2')

def query_chromadb(question: str, n_results: int = 10, n_rerank: int = 4):
    database = chromadb.PersistentClient(path="chroma_db")
    collection = database.get_collection("test_fin")
    results = collection.query(
        query_texts=[question],
        n_results=n_results
    )
    chunks = results.get("documents", [[]])[0]

    if not chunks:
        return []
    pairs = [[question, chunk] for chunk in chunks]
    scores = reranker.predict(pairs)
    reranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
    top_chunks = [chunk for chunk, score in reranked[:n_rerank]]
    return top_chunks


def create_conversation(question: str, context: str):

    client = genai.Client()

    with open("src\history\historyQA.txt", "r", encoding="utf-8") as f:
        prevQuestions = f.read().strip()
    
    with open("src\history\historyQA.txt", "a", encoding="utf-8") as f:
        f.write("\nQuestion: " + question + "\n")


    messages = [
        (
            "system",
            "-You are a helpful assistant that answers questions using only the information found in the context that the user provides."
            "-If you don't find the answer in the context provided, say that, and don't provide an answer from other sources."
            "-This is the history of the conversation that you can use to give better and more suitable responses: \n" + prevQuestions + "."
            "-If the user asks you about the previous conversation between you and him, give him relevant information from that conversation, if you haven't discused that specific topic with the user"
            "tell him that."
            "-Note that the user can use in a sentence words from other languages."
            "-The user can refer to a person using only his first name, last name or both. So that means that he/she is refering at the same exact person. Keep that in mind when answering a question.",
        ),
        ("human", "Based on the following documents:\n\n" + context + " \n\n give me an answer to the question" + question),
    ]

    response = client.models.generate_content(
        model="gemini-2.5-flash", contents=f"{messages}"
    )

    with open("src/history/historyQA.txt", "a", encoding="utf-8") as f:
        f.write("Answer: \n" + response.text + "\n\n")

    
    # Retinere intrebare, raspuns + chunk-uri similare
    # Scriu intrebarea in fisier 
    with open("src\history\history+context.txt", "a", encoding="utf-8") as f:
        f.write("Question: \n" + question + "\n\n")
    with open("src\history\history+context.txt", "a", encoding="utf-8") as f:
        f.write("Answer: \n" + response.text + "\n\n")
    with open("src\history\history+context.txt", "a", encoding="utf-8") as f:
        f.write("Context: \n" + context + "\n\n")


    return response.text


