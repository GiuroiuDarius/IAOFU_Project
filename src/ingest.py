from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import chromadb
import re
from uuid import uuid4
import matplotlib.pyplot as plt
from langchain_community.document_loaders import PyPDFLoader
from sentence_transformers import SentenceTransformer

# ChromaDB 0.4.16+ requires embedding_function to be a class with __call__(self, input)
class SentenceTransformerEmbeddingFunction:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
    def __call__(self, input):
        return self.model.encode(input).tolist()


# Extragere text de pe site
def extract_text(pdfs: list):
   
    textChunks = []
    for pdf in pdfs:
        loader = PyPDFLoader(pdf)
        pages = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1024,
            chunk_overlap = 400,
        )

        for page in pages:
            clean_text = re.sub(r'\s+', ' ', page.page_content).strip()
            textChunks.extend(splitter.split_text(clean_text))

    return textChunks

# Pentru a creea embedding-uri
def embedding_text(textChunks: list):
    # Load SentenceTransformer model
    embedding_function = SentenceTransformerEmbeddingFunction()

    database = chromadb.PersistentClient(path="chroma_db")
    collection = database.create_collection(
        "test_fin",
            embedding_function=embedding_function
    )

    documents = []
    for chunk in textChunks:
        document = Document(
            page_content=chunk
        )
        documents.append(document.page_content)

    print(f"Chunk-uri de adăugat: {len(documents)}")
    uuids = [str(uuid4()) for _ in range(len(textChunks))]
    collection.add(documents=documents, ids=uuids)

    # Numarul de documente din colecție
    try:
        count = collection.count()
        print(f"Chunk-uri salvate în colecție: {count}")
    except Exception as e:
        print(f"Eroare la numărarea chunk-urilor: {e}")

    return collection




def main():
    pdfs = ["dataset\dictionar1.pdf", "dataset\dictionar2.pdf", "dataset\dictionar3.pdf"]
    text_chunks = extract_text(pdfs=pdfs)

    # for chunk in text_chunks:
        # print("\n\nChunk\n\n" + chunk)
    embedding_text(text_chunks)
        

if __name__ == "__main__":
    main()