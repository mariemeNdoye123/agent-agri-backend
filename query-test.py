#!/usr/bin/env python3
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Configurations
PERSIST_DIRECTORY = "chroma_db"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

if __name__ == "__main__":
    # Charger le modèle d’embeddings
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # Charger la base vectorielle persistée 
    vectordb = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings
    )

    # Définir une requête utilisateur
    query = "Quelle est l'importance du mil au Sénégal?"

    # Interroger Chroma → recherche sémantique des 3 passages les plus proches
    results = vectordb.similarity_search(query, k=3)
    
    # Affichage des résultats
    print("\n=== Résultats de la recherche ===")
    if results:
        for i, doc in enumerate(results, start=1):
            print(f"\n Résultat {i}:")
            print(doc.page_content[:500] + "...")# Afficher un extrait du texte (500 premiers caractères)
            print(f"Source: {doc.metadata.get('source_file')} | Page: {doc.metadata.get('page_number')}")
    else:
        print("Aucun résultat pertinent trouvé. Veuillez vérifier votre corpus.")
