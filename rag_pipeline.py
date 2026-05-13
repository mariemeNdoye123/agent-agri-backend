from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from typing import List, Any

# --- Configuration ---
PERSIST_DIRECTORY = "chroma_db"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
LLM_MODEL = "mistral" 

qa_chain = None

# --- Étape 1 : Traduction de l'entrée (Wolof -> Français) ---

def translate_to_french(text: str) -> str:
    """Traduit la question si elle est en Wolof pour permettre la recherche RAG."""
    llm = Ollama(model=LLM_MODEL, temperature=0) # Rigueur maximale
    
    prompt = (
        "Tu es un traducteur expert Wolof-Français spécialisé en agriculture sénégalaise. "
        "Ta tâche est de traduire la question de l'utilisateur en Français pour un système de recherche de documents.\n\n"
        
        "CONSIGNE 1 (LEXIQUE STRICT) : Tu dois impérativement respecter ces correspondances :\n"
        "- dugub = mil\n"
        "- mboq = maïs\n"
        "- ceeb = riz\n"
        "- gerté = arachide\n"
        "- ñébé = niébé\n"
        "- bay / baye = cultiver / planter\n\n"
        
        "CONSIGNE 2 (FRANÇAIS) : Si la question est déjà en Français, renvoie-la EXACTEMENT telle quelle, "
        "sans changer un seul mot, ni la ponctuation.\n\n"
        
        "CONSIGNE 3 (FORMAT) : Réponds UNIQUEMENT par la traduction, sans aucun commentaire.\n\n"
        
        "Exemples :\n"
        "Question : 'Naka lañuy baye dugub?' -> Traduction : Comment cultiver le mil ?\n"
        "Question : 'Naka lañuy baye mboq?' -> Traduction : Comment cultiver le maïs ?\n"
        "Question : 'Quelles sont les maladies du riz ?' -> Traduction : Quelles sont les maladies du riz ?\n\n"
        
        f"Question : '{text}' -> Traduction :"
    )
    
    try:
        response = llm.invoke(prompt)
        return response.strip().replace('"', '')
    except:
        return text

# --- Initialisation du Pipeline ---

def initialize_rag_pipeline():
    global qa_chain
    print("Initialisation du pipeline RAG (Questions Wolof/FR -> Réponses FR)...")

    # 1. Ressources
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectordb = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    llm = Ollama(model=LLM_MODEL)

    # 2. Prompt Système (Impose le Français pour la réponse)
    SYSTEM_PROMPT = (
        "Tu es un expert agricole sénégalais. Réponds à la question de l'utilisateur "
        "en utilisant UNIQUEMENT les informations des documents fournis ci-dessous. "
        "TA RÉPONSE DOIT ÊTRE EN FRANÇAIS, claire et précise.\n\n"
        "Documents contextuels :\n{context}"
    )
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{input}")
    ])

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # 3. Logique de traitement (Simplifiée)
    def bilingual_input_logic(query: str):
        # A. On traduit la question vers le français (Transparent pour l'utilisateur)
        query_fr = translate_to_french(query)
        print(f"DEBUG: Question reçue: {query} | Question traitée: {query_fr}")
        
        # B. Recherche RAG (en Français)
        docs = retriever.invoke(query_fr)
        context = format_docs(docs)
        
        # C. Génération de la réponse technique (Toujours en Français)
        chain = prompt_template | llm
        answer_fr = chain.invoke({"context": context, "input": query_fr})
        
        # On retourne directement la réponse en français
        return {
            "result": answer_fr,
            "source_documents": docs
        }

    # Création du runnable pour main.py
    qa_chain = RunnableLambda(bilingual_input_logic)
    
    print("Pipeline prêt : Entrée bilingue acceptée, sortie en Français garantie.")
    return qa_chain