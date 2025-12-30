from query import query_chromadb, create_conversation

def main():

    print("+---------------------------------+")
    print("Asistent chat. Scrie 'exit' pentru a închide conversația.")
    print("+---------------------------------+")

    while True:
        question = input("\nQuestion: ")
        if question.strip().lower() == "exit":
            print("Conversation ended.")
            break
        relevant_chunks = query_chromadb(question)
        context = "\n".join(relevant_chunks)
        if not context.strip():
            print("No relevant context found in documents.")
            continue
        answer = create_conversation(question, context)
        print("\nAnswer:\n", answer)
        print("+---------------------------------+")

if __name__ == "__main__":
    main()