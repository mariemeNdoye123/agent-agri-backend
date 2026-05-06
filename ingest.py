import os
import shutil
from pathlib import Path
import json
from typing import List


from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# ----------CONFIGURATION----------
DATA_PATH = Path("data")            # dossier contenant les pdf
PERSIST_DIRECTORY = "chroma_db"    # dossier Chroma persistant
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
# ----------------------------

def load_documents(data_path: Path):
    """Charge les PDF et ajoute les métadonnées (source_file, page_number)"""
    documents = []
    if not data_path.exists():
        raise FileNotFoundError(f"Le dossier {data_path} n'existe pas.")

    pdf_files = sorted([p for p in data_path.iterdir() if p.suffix.lower() == ".pdf"])

    for pdf in pdf_files:
        loader = PyPDFLoader(str(pdf))
        docs = loader.load()
        for i, d in enumerate(docs):
            d.metadata["source_file"] = pdf.name
            d.metadata["page_number"] = d.metadata.get("page", i+1)
        documents.extend(docs)
    return documents

def split_documents(documents: List[Document]) -> List[Document]:
    """
    Découpe chaque document en chunks et préserve les métadonnées.
    Chaque chunk aura :
      - "source" : nom du fichier PDF
      - "page"   : numéro de page original
      - "chunk_index" : index du chunk dans le document
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    all_chunks = []

    for doc in documents:
        # Récupère le nom du fichier et le numéro de page du document parent
        source_file = doc.metadata.get("source_file", "inconnu.pdf")
        page_number = doc.metadata.get("page_number", 0)

        # Découpe le texte du document en chunks
        chunks_text = splitter.split_text(doc.page_content)

        for i, chunk_text in enumerate(chunks_text):
            chunk_doc = Document(
                page_content=chunk_text,
                metadata={
                    "source": source_file,
                    "page": page_number,
                    "chunk_index": i
                }
            )
            all_chunks.append(chunk_doc)

    print(f"Total chunks créés : {len(all_chunks)}")
    return all_chunks

def create_vector_store(chunks, persist_directory: str, reset: bool = False):
    """Crée ou met à jour la base Chroma avec les chunks"""
    if reset and os.path.exists(persist_directory):
        print(f"Suppression du dossier existant {persist_directory} (reset)...")
        shutil.rmtree(persist_directory)

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    print("Création ou ajout à la base Chroma ...")
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )

    print(f"Base vectorielle créée/persistée dans: {persist_directory} ({len(chunks)} chunks)")


    return vectordb

if __name__ == "__main__":
    print("=== Début de l'ingestion ===")
    docs = load_documents(DATA_PATH)
    print(f"Documents chargés : {len(docs)}")
    if not docs:
        raise SystemExit("Aucun PDF trouvé dans ./data/")

    chunks = split_documents(docs)
    print(f"Texte découpé en chunks : {len(chunks)}")

    create_vector_store(chunks, PERSIST_DIRECTORY, reset=True)
    print("=== Ingestion terminée ===")
