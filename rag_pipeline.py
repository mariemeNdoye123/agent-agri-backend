from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from typing import List, Any

PERSIST_DIRECTORY = "chroma_db"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
LLM_MODEL = "mistral"

qa_chain = None

def format_docs(docs: List[Any]) -> str:
    """Formate une liste de documents en une seule chaîne de contexte pour le prompt."""
    return "\n\n".join(getattr(doc, "page_content", str(doc)) for doc in docs)

def initialize_rag_pipeline():
    global qa_chain
    print("Initialisation du pipeline RAG...")

    # Embeddings + base de données vectorielle
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectordb = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})

    # LLM
    llm = Ollama(model=LLM_MODEL)

    # Prompt système
    SYSTEM_PROMPT = (
        "Tu es un expert agricole sénégalais. "
        "LA RÉPONSE DOIT IMPÉRATIVEMENT ÊTRE EN FRANÇAIS. "
        "Utilise uniquement les informations présentes dans les documents fournis ci-dessous. "
        "Si la réponse n’est pas contenue dans ces documents, réponds : "
        "'Je n’ai pas de réponse à cette question avec les informations disponibles.'\n\n"
        "Documents contextuels :\n{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{input}")
    ])

    # Chaîne RAG native
    rag_chain_native = RunnableParallel({
        "context": retriever | format_docs,
        "source_documents": retriever,
        "input": RunnablePassthrough()
    })

    # Chaîne de génération de réponse
    response_chain = prompt | llm | StrOutputParser()

    # Chaîne finale combinant réponse et sources
    qa_chain = (
        rag_chain_native
        | RunnableParallel({
            "result": response_chain,
            "source_documents": (lambda x: x.get("source_documents", []))
        })
    )

    print("Pipeline RAG initialisé avec succès.")
    return qa_chain


