import time
import datetime
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Dict, Any

# Importation de vos dépendances
from admin_utils import run_full_ingestion 
from rag_pipeline import initialize_rag_pipeline

# Importation de la base de données et des modèles
from sqlalchemy.orm import Session
from db import SessionLocal 
from models import Job, User # Assurez-vous que Job et User sont importés

# Dépendance à maintenir : l'Executor pour le travail asynchrone
executor = ThreadPoolExecutor(max_workers=1)

# Traçage des IDs des jobs actifs en mémoire pour éviter le double lancement.
active_job_ids: Dict[str, int] = {} 


# ----------------------------------------------------------------------
# 1. Gestion de la Session DB pour le Thread
# ----------------------------------------------------------------------

def get_db_session_for_thread() -> Session:
    """Crée et retourne une session DB distincte pour le thread de fond."""
    print(f"[{datetime.datetime.now()}] [Job] Tentative de création de session DB pour le thread...")
    try:
        db = SessionLocal() 
        return db
    except Exception as e:
        print(f"[{datetime.datetime.now()}] [Job ERREUR CRITIQUE] Erreur lors de la création de la session DB pour le thread: {e}")
        # Re-lever l'exception si la DB est cruciale
        raise

# ----------------------------------------------------------------------
# 2. Tâche de fond (Background Task)
# ----------------------------------------------------------------------

def background_ingestion_task(job_db_id: int, update_chain_func, reset: bool = False):
    """
    Fonction bloquante exécutée dans un thread séparé.
    Mise à jour directe de la base de données.
    """
    
    print(f"[{datetime.datetime.now()}] [Job {job_db_id}] Tâche de fond démarrée.")

    try:
        # 1. Connexion DB et récupération de l'entrée Job
        db = get_db_session_for_thread()
        job_entry = db.query(Job).filter(Job.id == job_db_id).first()
        
        if not job_entry:
            print(f"[{datetime.datetime.now()}] [Job {job_db_id}] Erreur: Entrée Job non trouvée.")
            db.close()
            return 
            
        print(f"[{datetime.datetime.now()}] [Job {job_db_id}] Entrée DB récupérée. Statut initial: {job_entry.status}")
        
        # --- Fonction de Callback ---
        def update_progress(message: str, percent: int, status: str = "EN_COURS"):
            """Met à jour l'état du job dans la base de données."""
            print(f"[{datetime.datetime.now()}] [Job {job_db_id} - {percent}%] Mise à jour du statut: {message}")
            try:
                # Vérifier si l'annulation a été demandée
                db.refresh(job_entry)
                if job_entry.status == "CANCELLÉ":
                    # Si c'est annulé, ne plus faire de mise à jour et sortir
                    print(f"[{datetime.datetime.now()}] [Job {job_db_id}] Annulation détectée. Arrêt du thread.")
                    raise InterruptedError("Annulation demandée.")

                job_entry.progress = percent
                job_entry.message = message
                job_entry.status = status
                db.commit() # Persistance immédiate
            except InterruptedError:
                # Si l'annulation est demandée, arrêter le thread
                raise
            except Exception as commit_error:
                db.rollback()
                print(f"[{datetime.datetime.now()}] [Job {job_db_id} ERREUR COMMIT] Échec du commit: {commit_error}")
            
        # Le statut EN_COURS est mis à jour ici
        update_progress("Démarrage de la tâche de fond...", 0, status="EN_COURS")
        
        # --- DÉBUT DE L'INGESTION ---
        print(f"[{datetime.datetime.now()}] [Job {job_db_id}] Appel de run_full_ingestion(reset={reset})...")
        
        # L'ingestion doit utiliser le progress_callback pour vérifier le statut CANCELLÉ
        ingestion_result = run_full_ingestion(reset=reset, progress_callback=update_progress)
        print(f"[{datetime.datetime.now()}] [Job {job_db_id}] run_full_ingestion terminé.")
        
        # Vérifier si l'annulation a été demandée pendant l'ingestion
        db.refresh(job_entry)
        if job_entry.status == "CANCELLÉ":
             job_entry.message = "Tâche annulée par l'administrateur pendant l'exécution."
             db.commit()
             print(f"[{datetime.datetime.now()}] [Job {job_db_id}] Tâche CANCELLÉE détectée à la fin de l'ingestion.")
             return # Fin normale si annulée

        # 2. Étape de post-traitement (90%)
        update_progress("Réinitialisation du pipeline RAG...", 90)
        
        new_qa_chain = initialize_rag_pipeline()
        update_chain_func(new_qa_chain)

        # 3. Étape Finale (100%)
        job_entry.progress = 100
        job_entry.status = "TERMINÉ"
        job_entry.message = "Ingestion complète terminée avec succès" 
        job_entry.end_time = datetime.datetime.utcnow()
        db.commit() 
        print(f"[{datetime.datetime.now()}] [Job {job_db_id}] Tâche TERMINÉE avec succès.")

    except InterruptedError:
        # Gère l'interruption demandée via le statut CANCELLÉ
        print(f"[{datetime.datetime.now()}] [Job {job_db_id}] Tâche arrêtée à la demande de l'administrateur.")
        # Le statut CANCELLÉ a déjà été mis à jour par stop_ingestion_job
        return
    except Exception as e:
       # Gestion de l'échec
        error_message = f"Erreur critique: {type(e).__name__}: {str(e)}"
        print(f"[{datetime.datetime.now()}] [Job {job_db_id} ÉCHEC] {error_message}")
        
        if 'job_entry' in locals() and job_entry:
             # Utiliser la session existante si elle est accessible
            job_entry.status = "ÉCHEC"
            job_entry.message = error_message
            job_entry.progress = 100 
            job_entry.end_time = datetime.datetime.utcnow()
            try:
                db.commit() 
            except Exception as final_commit_error:
                print(f"[{datetime.datetime.now()}] [Job {job_db_id} ERREUR FINALE] Échec du commit après erreur: {final_commit_error}")


    finally:
        # Nettoyage et suppression de la référence du job actif
        if 'job_entry' in locals() and job_entry and job_entry.job_id in active_job_ids:
            del active_job_ids[job_entry.job_id]
        if 'db' in locals() and db:
            db.close() # Fermeture de la session du thread
            print(f"[{datetime.datetime.now()}] [Job {job_db_id}] Session DB fermée.")


# ----------------------------------------------------------------------
# 3. Fonctions d'API (Accès à la DB via Session FastAPI)
# ----------------------------------------------------------------------
def start_ingestion_job(db: Session, user: User, reset: bool = False, update_chain_func=None) -> str:
    """
    Crée l'entrée dans la DB, l'associe à l'utilisateur, et soumet la tâche.
    """
    job_uid = f"ingestion_{time.time()}"
    
    # 1. Créer et enregistrer le job dans la DB (y compris l'ID utilisateur)
    new_job = Job(
        job_id=job_uid,
        job_type="INGESTION_CORPUS",
        status="EN_QUEUE",
        progress=0,
        message="En file d'attente...",
        launched_by_user_id=user.id 
    )
    db.add(new_job)
    db.commit()
    db.refresh(new_job)

    # 2. Soumettre la fonction bloquante à l'executor
    executor.submit(background_ingestion_task, new_job.id, update_chain_func, reset)
    
    # 3. Conserver la trace en mémoire (utile en production pour le traçage)
    active_job_ids[job_uid] = new_job.id
    
    return job_uid

def stop_ingestion_job(db: Session, job_id: str):
    """
    Tente d'annuler un job en cours en mettant à jour son statut à CANCELLÉ.
    La boucle d'ingestion doit vérifier ce statut pour s'arrêter.
    """
    job = db.query(Job).filter(Job.job_id == job_id).first()
    if not job:
        raise ValueError("Job non trouvé.")

    if job.status in ["TERMINÉ", "ÉCHEC", "CANCELLÉ"]:
        return {"message": f"Le job {job.job_id} est déjà {job.status}."}

    # Mise à jour du statut pour l'annulation
    job.status = "CANCELLÉ" 
    job.message = "Annulation demandée par l'administrateur..."
    job.end_time = datetime.datetime.utcnow() 
    db.commit()
    
    return {"message": f"Demande d'annulation du job {job.job_id} envoyée. Le statut sera mis à jour dès que le thread s'arrête."}

def delete_job(db: Session, job_id: str):
    """Supprime un job de l'historique."""
    job = db.query(Job).filter(Job.job_id == job_id).first()
    if not job:
        raise ValueError("Job non trouvé.")
        
    # Empêcher la suppression d'un job en cours
    if job.status == 'EN_COURS':
         raise ValueError("Impossible de supprimer un job en cours d'exécution. Veuillez l'arrêter d'abord.")
         
    db.delete(job)
    db.commit()
    return {"message": f"Job {job_id} supprimé de l'historique."}

def _job_to_dict(job: Job) -> Dict[str, Any]:
    """Convertit l'objet Job DB en dictionnaire pour le frontend."""
    return {
        "job_id": job.job_id,
        "type": job.job_type,
        "status": job.status,
        "progress": job.progress,
        "message": job.message,
        "start_time": job.start_time.timestamp() if job.start_time else None,
        "end_time": job.end_time.timestamp() if job.end_time else None,
        "launched_by_email": job.launched_by.email if job.launched_by else "N/A" 
    }


def get_job_status(db: Session, job_id: str):
    """Récupère le statut depuis la DB."""
    job = db.query(Job).filter(Job.job_id == job_id).first()
    if not job:
        return None
        
    return _job_to_dict(job)


def get_all_jobs(db: Session):
    """Récupère l'historique complet depuis la DB."""
    # Jointure avec l'utilisateur pour obtenir l'email
    jobs = db.query(Job).order_by(Job.start_time.desc()).all()
    
    return [_job_to_dict(job) for job in jobs]
