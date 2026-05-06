import json
import os
import shutil
import csv
import tempfile
from pathlib import Path
from typing import List, Dict, Any

# Importe les fonctions d'ingestion de votre fichier existant
import ingest

# Import des modèles pour l'accès aux données
from models import User, Historique
from sqlalchemy.orm import Session
from sqlalchemy import func

# Dossier où les PDF doivent être stockés pour l'ingestion
DATA_PATH = Path("data")
META_FILE = DATA_PATH / "documents_meta.json" 

# GESTION DES DOCUMENTS (AVEC ARCHIVAGE)
def load_metadata() -> Dict[str, Any]:
    """Charge le fichier JSON contenant les états des documents (is_active)."""
    if META_FILE.exists():
        with open(META_FILE, "r") as f:
            return json.load(f)
    return {}


def save_metadata(metadata: Dict[str, Any]):
    """Sauvegarde les métadonnées (is_active) dans un fichier JSON."""
    with open(META_FILE, "w") as f:
        json.dump(metadata, f, indent=4)


def get_corpus_documents(active_only: bool = True) -> List[Dict[str, Any]]:
    """
    Liste les fichiers PDF du dossier DATA_PATH.
    Si active_only=True → renvoie uniquement les fichiers actifs.
    Si active_only=False → renvoie uniquement les fichiers archivés.
    """
    if not DATA_PATH.exists():
        return []
    
    metadata = load_metadata()
    documents = []

    for p in DATA_PATH.iterdir():
        if p.suffix.lower() != ".pdf":
            continue
        
        is_active = metadata.get(p.name, {}).get("is_active", True)

        if active_only and not is_active:
            continue
        if not active_only and is_active:
            continue
        
        documents.append({
            "filename": p.name,
            "is_active": is_active
        })

    return sorted(documents, key=lambda d: d["filename"])


def add_document_to_corpus(file_content: bytes, filename: str):
    """
    Sauvegarde un fichier PDF dans le dossier DATA_PATH et l’active par défaut.
    """
    if not DATA_PATH.exists():
        DATA_PATH.mkdir()

    save_path = DATA_PATH / filename
    with open(save_path, "wb") as f:
        f.write(file_content)

    # Ajouter aux métadonnées avec is_active=True
    metadata = load_metadata()
    metadata[filename] = {"is_active": True}
    save_metadata(metadata)

    return save_path.resolve()


def toggle_document_status(filename: str) -> bool:
    """
    Inverse l’état (actif ↔ archivé) d’un document sans le supprimer.
    Retourne le nouvel état (True = actif, False = archivé).
    """
    file_path = DATA_PATH / filename
    if not file_path.exists():
        raise FileNotFoundError(f"Le document {filename} n'existe pas dans {DATA_PATH}")

    metadata = load_metadata()
    current_state = metadata.get(filename, {}).get("is_active", True)
    metadata[filename] = {"is_active": not current_state}
    save_metadata(metadata)

    return not current_state


#Ingestion
def run_full_ingestion(reset: bool = False, progress_callback=None):
    """
    Exécute la logique complète d'ingestion (charge, splitte, crée/met à jour la DB).
    Accepte un progress_callback(message, percentage) pour mettre à jour l'état.
    """

    def report_progress(message: str, percentage: int):
        """Fonction interne pour gérer l'appel du callback si fourni."""
        if progress_callback:
            progress_callback(message, percentage)
    try:
        # 1. Charger les PDF
        report_progress("Chargement des documents du corpus...", 10)
        docs = ingest.load_documents(ingest.DATA_PATH)
        if not docs:
            # Même s'il n'y a rien à faire, on met à jour le statut
            report_progress("Aucun PDF trouvé pour l'ingestion.", 100) 
            return {"status": "warning", "message": "Aucun PDF trouvé pour l'ingestion."}

        # 2. Découper en chunks
        report_progress("Extraction des textes et découpage en morceaux (chunks)...", 30)
        chunks = ingest.split_documents(docs)
        
        if not chunks:
            report_progress("Aucun contenu textuel n'a pu être extrait des documents.", 100)
            return {"status": "warning", "message": "Aucun contenu textuel n'a pu être extrait."}

        # 3. Préparation finale des données / Vectorisation
        report_progress(f"Vectorisation de {len(chunks)} morceaux de texte...", 60)
        # Note: Cette étape englobe la "préparation/normalisation" et le début de la vectorisation.

        # 4. Création et mise a jour de la base vectorielle
        report_progress("Stockage dans la base vectorielle et indexation...", 80)
        ingest.create_vector_store(chunks, ingest.PERSIST_DIRECTORY, reset=reset)

        return {"status": "success", "message": f"Ingestion complète terminée. {len(chunks)} morceaux créés."}

    except FileNotFoundError as e:
        report_progress(f"Erreur de fichier: {str(e)}", 100)
        return {"status": "error", "message": str(e)}
    except Exception as e:
        report_progress(f"Erreur critique lors de l'ingestion: {e}", 100)
        return {"status": "error", "message": f"Erreur critique lors de l'ingestion: {e}"}


# ----------------------------------------------------------------------
# FONCTIONS D'EXPORT ET DE STATISTIQUES (ADMIN/CHERCHEUR)
# ----------------------------------------------------------------------

def export_global_history_csv(db: Session) -> str:
    """
    Récupère l'historique complet de tous les utilisateurs et le sauve dans un fichier CSV temporaire.
    :return: Chemin d'accès au fichier CSV temporaire.
    """
    # Jointure pour récupérer l'email de l'utilisateur
    history_data = db.query(Historique, User.email).join(User, Historique.user_id == User.id).all()

    # Création d'un fichier temporaire pour le CSV
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv', encoding='utf-8') as tmpfile:
        writer = csv.writer(tmpfile)

        # Écriture de l'en-tête
        writer.writerow(["ID", "Date", "Utilisateur Email", "Question", "Reponse", "Sources"])

        # Écriture des données
        for entry, email in history_data:
            writer.writerow([
                entry.id,
                entry.date_question.isoformat(),
                email,
                entry.question.replace('\n', ' '),
                entry.reponse.replace('\n', ' '),
                entry.sources
            ])
        
        return tmpfile.name # Retourne le chemin du fichier temporaire

def get_dashboard_metrics(db: Session) -> Dict[str, Any]:
    """
    Calcule et retourne les métriques globales pour le tableau de bord.
    """
    total_users = db.query(User).filter(User.is_active == True).count()

    total_questions = db.query(Historique).count()
    
    # Comptage des utilisateurs par rôle
    users_by_role = db.query(User.role, func.count(User.role)).group_by(User.role).all()
    roles_dict = {role: count for role, count in users_by_role}
    
    # Documents dans le corpus (fichiers PDF sur le disque)
    documents_in_corpus = get_corpus_documents()
    total_documents = len(documents_in_corpus)
    
    # Exemple de calcul de l'activité récente (questions posées dans les dernières 24h)
    from datetime import datetime, timedelta
    one_day_ago = datetime.now() - timedelta(days=1)
    recent_questions = db.query(Historique).filter(Historique.date_question >= one_day_ago).count()
    
    return {
        "total_users": total_users,
        "total_questions": total_questions,
        "users_by_role": roles_dict,
        "documents_in_corpus": documents_in_corpus,
        "total_documents": total_documents,
        "recent_questions_24h": recent_questions
    }
