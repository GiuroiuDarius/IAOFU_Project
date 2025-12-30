import csv
from query import query_chromadb, create_conversation

INPUT_CSV = "dataset/test_dataset.csv"
OUTPUT_TXT = "dataset/rag_test_results.txt"

with open(INPUT_CSV, encoding="utf-8") as f:
    reader = csv.DictReader(f)
    results = []
    nr = 0
    for row in reader:
        if row["Corpus"] == "DCLRV":
            nr += 1
            if nr > 5:
                break
            question = row["Întrebare"]
            correct_answer = row["Răspuns"]
            relevant_chunks = query_chromadb(question)
            context = "\n".join(relevant_chunks)
            generated_answer = create_conversation(question, context)
            results.append({
                "question": question,
                "correct_answer": correct_answer,
                "generated_answer": generated_answer
            })

with open(OUTPUT_TXT, "w", encoding="utf-8") as out:
    for i, r in enumerate(results, 1):
        out.write(f"Test {i}\n")
        out.write(f"Question: {r['question']}\n")
        out.write(f"Correct Answer: {r['correct_answer']}\n")
        out.write(f"Generated Answer: {r['generated_answer']}\n")
        out.write("-"*40 + "\n")
