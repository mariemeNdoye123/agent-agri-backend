# main.py
# ================= IMPORTATIONS =================
import json
import os
from fastapi import FastAPI, Depends, HTTPException, File, UploadFile
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.responses import RedirectResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import Optional, List 
from datetime import date, timedelta
from pathlib import Path

# ================= COMPOSANTS DU PROJET =================
from db import engine, Base, get_db
from models import User, Historique, Chat
from schemas import (
    RegisterModel, Question, HistoriqueResponse, UpdateProfileModel,
    UserUpdate, ChatResponse,FeedbackModel, FeedbackResolveModel,FeedbackResponse
)
from security import get_password_hash, verify_password, create_access_token, get_current_user, require_roles
from rag_pipeline import initialize_rag_pipeline, qa_chain
from admin_utils import (
    get_corpus_documents, add_document_to_corpus, run_full_ingestion,
    export_global_history_csv, get_dashboard_metrics, toggle_document_status
)
from jobs_service import start_ingestion_job, get_job_status, get_all_jobs, stop_ingestion_job, delete_job



# ================= INITIALISATION DE LA BASE DE DONNÉES =================
# Création des tables SQLAlchemy si elles n'existent pas
Base.metadata.create_all(bind=engine)

# Chemin vers le dossier de stockage des documents
DATA_PATH = Path("data")

# ================= INITIALISATION DE L’APPLICATION FASTAPI =================
app = FastAPI(
    title="Agent Agricole RAG API",
    description="API d'aide à la décision agricole basée sur la recherche augmentée (RAG) et Mistral."
)

# ================= CORS =================
# Autoriser le front Angular (localhost:4200) à communiquer avec cette API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================= INITIALISATION DU PIPELINE RAG =================
print("Initialisation du pipeline RAG...")
try:
    qa_chain = initialize_rag_pipeline()
    print("Pipeline RAG initialisé avec succès.")
except Exception as e:
    print(f"Erreur FATALE lors de l'initialisation du pipeline RAG: {e}")
    qa_chain = None

def update_main_qa_chain(new_chain):
    """Met à jour l'instance globale de QA chain depuis un job en arrière-plan."""
    global qa_chain
    qa_chain = new_chain
    print("Pipeline RAG mis à jour par le thread de travail.")


# ================= ROUTES =================

# --- Route d'accueil ---
@app.get("/")
def home():
    # Redirige vers la documentation interactive de FastAPI
    return RedirectResponse(url="/docs")

# ------------------ AUTHENTIFICATION & PROFIL ------------------

@app.post("/register", tags=["Authentification & Profil"])
def register(user: RegisterModel, db: Session = Depends(get_db)):
    """
    Inscription d'un nouvel utilisateur.
    - Vérifie si l'email existe déjà.
    - Hash le mot de passe.
    - Ajoute l'utilisateur à la base de données.
    """
    if db.query(User).filter(User.email == user.email).first():
        raise HTTPException(status_code=400, detail="Un compte existe déjà avec cet email")
    hashed_pw = get_password_hash(user.mot_de_passe)
    new_user = User(nom=user.nom, prenom=user.prenom, email=user.email, mot_de_passe=hashed_pw, role=user.role)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return {"message": f"Utilisateur {user.prenom} {user.nom} créé avec succès"}

@app.post("/login", tags=["Authentification & Profil"])
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    """
    Connexion utilisateur :
    - Vérifie l'email et le mot de passe.
    - Vérifie si l'utilisateur est actif.
    - Génère un token JWT pour l'authentification.
    """
    db_user = db.query(User).filter(User.email == form_data.username).first()

    # Vérifie si l'utilisateur existe
    if not db_user or not verify_password(form_data.password, db_user.mot_de_passe):
        raise HTTPException(status_code=400, detail="Identifiants invalides")

    # Vérifie si l'utilisateur est actif
    if not db_user.is_active:
        raise HTTPException(status_code=403, detail="Compte inactif. Veuillez contacter un administrateur.")

    # Génère le token si tout est OK
    access_token = create_access_token(data={"sub": db_user.email}, expires_delta=timedelta(days=30))
    return {"access_token": access_token, "token_type": "bearer", "expire_in_days": 30}


@app.get("/profil", tags=["Authentification & Profil"])
def profil(utilisateur: User = Depends(get_current_user)):
    """
    Récupère le profil de l'utilisateur connecté via JWT.
    """
    if not utilisateur:
        raise HTTPException(status_code=401, detail="Non authentifié")
    return {
        "id": utilisateur.id,
        "nom": utilisateur.nom,
        "prenom": utilisateur.prenom,
        "email": utilisateur.email,
        "role": utilisateur.role
    }

@app.put("/update_profil", tags=["Authentification & Profil"])
def update_profil(data: UpdateProfileModel, utilisateur: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """
    Met à jour le profil de l'utilisateur connecté.
    - Nom, prénom et mot de passe peuvent être modifiés.
    """
    if not utilisateur:
        raise HTTPException(status_code=401, detail="Non authentifié")
    if data.nom: utilisateur.nom = data.nom
    if data.prenom: utilisateur.prenom = data.prenom
    if data.mot_de_passe: utilisateur.mot_de_passe = get_password_hash(data.mot_de_passe)
    db.commit()
    db.refresh(utilisateur)
    return {"message": "Profil mis à jour avec succès.", "profil": {"nom": utilisateur.nom, "prenom": utilisateur.prenom, "email": utilisateur.email, "role": utilisateur.role}}


# ------------------ RAG & INTERACTION ------------------

@app.post("/ask", tags=["RAG & Interaction"])
def ask_question(
    question: Question,
    chat_id: Optional[int] = None,
    db: Session = Depends(get_db),
    utilisateur: Optional[User] = Depends(get_current_user)
):
    """Pose une question au pipeline RAG et renvoie la réponse + sources."""

    if not qa_chain:
        raise HTTPException(status_code=503, detail="Le service RAG n'est pas encore initialisé.")

    # Appel pipeline
    result = qa_chain.invoke(question.query)
    answer_text = result.get("result", "Erreur lors de la génération de la réponse")
    

    # Récupération sécurisée des sources
    raw_sources = result.get("source_documents", [])
    sources = []
    for doc in raw_sources:
        metadata = getattr(doc, "metadata", {})
        sources.append({
            "file": metadata.get("source") or metadata.get("source_file") or "inconnu",
            "page": metadata.get("page") or metadata.get("page_number") or 0
        })

    response_data = {"question": question.query, "answer": answer_text, "sources": sources}

    # Historisation si utilisateur connecté
    if utilisateur:
        if chat_id:
            chat = db.query(Chat).filter(Chat.id == chat_id, Chat.user_id == utilisateur.id).first()
            if not chat:
                raise HTTPException(status_code=404, detail="Discussion introuvable")
        else:
            chat = Chat(titre=question.query[:50], user_id=utilisateur.id)
            db.add(chat)
            db.commit()
            db.refresh(chat)

        histo = Historique(
            question=question.query,
            reponse=answer_text,
            sources=json.dumps(sources),
            chat_id=chat.id,
            user_id=utilisateur.id,
            # is_helpful est null par défaut, pas besoin de le définir ici
        )
        db.add(histo)
        db.commit()
        db.refresh(histo)

        response_data.update({
            "chat_id": chat.id, 
            "chat_title": chat.titre,
            "historique_id": histo.id 
        })

    return response_data

@app.get("/historique", response_model=List[ChatResponse], tags=["RAG & Interaction"])
def historique(utilisateur: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """
    Retourne toutes les discussions et l'historique des messages d'un utilisateur connecté.
    """
    if not utilisateur:
        raise HTTPException(status_code=401, detail="Utilisateur non authentifié ou token invalide")
    chats = db.query(Chat).filter(Chat.user_id == utilisateur.id).all()
    
    response_list = []
    for chat in chats:
        messages_response = []
        for m in chat.messages:
            messages_response.append(
                HistoriqueResponse(
                    question=m.question, 
                    reponse=m.reponse, 
                    sources=json.loads(m.sources or "[]"), 
                    date=m.date_question.isoformat(),
                    id=m.id,
                    is_helpful=m.is_helpful # <-- LIGNE CRUCIALE MAINTENANT DISPONIBLE
                )
            )
        
        response_list.append(
            ChatResponse(
                id=chat.id,
                titre=chat.titre,
                date_creation=chat.date_creation.isoformat(),
                messages=messages_response
            )
        )
        
    return response_list


@app.delete("/delete_chat/{chat_id}", tags=["RAG & Interaction"])
def delete_chat(chat_id: int, utilisateur: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """
    Supprime une discussion et tous ses messages pour l'utilisateur connecté.
    """
    if not utilisateur:
        raise HTTPException(status_code=401, detail="Utilisateur non authentifié")
    chat = db.query(Chat).filter(Chat.id == chat_id, Chat.user_id == utilisateur.id).first()
    if not chat:
        raise HTTPException(status_code=404, detail="Discussion introuvable ou non autorisée")
    db.query(Historique).filter(Historique.chat_id == chat.id).delete()
    db.delete(chat)
    db.commit()
    return {"message": f"Discussion '{chat.titre}' supprimée avec succès."}

@app.post("/feedback", tags=["RAG & Interaction"])
def submit_feedback(
    feedback_data: FeedbackModel,
    db: Session = Depends(get_db),
    utilisateur: User = Depends(get_current_user)
):
    """
    Enregistre le feedback (pouce levé/baissé) pour une réponse spécifique dans l'historique.
    """
    if not utilisateur:
        raise HTTPException(status_code=401, detail="Non authentifié")

    # Récupérer l'entrée Historique
    historique_entry = db.query(Historique).filter(
        Historique.id == feedback_data.historique_id,
        Historique.user_id == utilisateur.id # l'utilisateur doit posséder le message
    ).first()

    if not historique_entry:
        raise HTTPException(status_code=404, detail="Message d'historique introuvable ou non autorisé.")

    # Mise à jour de l'entrée avec le feedback
    historique_entry.is_helpful = feedback_data.is_helpful
    
    db.commit()

    # Log interne (amélioration du modèle)
    if not feedback_data.is_helpful:
        print(" FEEDBACK NÉGATIF REÇU (Mauvaise Réponse) ")
        print(f"Question : {historique_entry.question}")
        print(f"Réponse : {historique_entry.reponse[:100]}...")
        # L'administrateur peut maintenant consulter cette entrée pour :
        # - Créer une paire Q/R corrigée pour le Fine-Tuning du LLM.
        # - Injecter un nouveau document corrigé dans le Corpus RAG.
        
    return {"message": "Feedback enregistré avec succès.", "is_helpful": feedback_data.is_helpful}


# ================= ADMINISTRATION =================

# --- Gestion des utilisateurs ---
@app.get("/admin/list_users", tags=["Administration : Utilisateurs"])
def liste_utilisateurs(db: Session = Depends(get_db), admin: User = Depends(require_roles("admin"))):
    """
    Liste uniquement les utilisateurs actifs (admin only).
    """
    utilisateurs_actifs = db.query(User).filter(User.is_active == True).all()
    return [
        {
            "id": u.id,
            "nom": u.nom,
            "prenom": u.prenom,
            "email": u.email,
            "role": u.role,
            "is_active": u.is_active
        }
        for u in utilisateurs_actifs
    ]

@app.get("/admin/list_archived_users", tags=["Administration : Utilisateurs"])
def liste_utilisateurs_archives(db: Session = Depends(get_db), admin: User = Depends(require_roles("admin"))):
    """
    Liste les utilisateurs désactivés (archivés).
    """
    utilisateurs_archives = db.query(User).filter(User.is_active == False).all()
    return [
        {
            "id": u.id,
            "nom": u.nom,
            "prenom": u.prenom,
            "email": u.email,
            "role": u.role,
            "is_active": u.is_active
        }
        for u in utilisateurs_archives
    ]

@app.patch("/admin/archive_user/{user_id}", tags=["Administration : Utilisateurs"])
def archive_user(user_id: int, db: Session = Depends(get_db), admin: User = Depends(require_roles("admin"))):
    utilisateur = db.query(User).filter(User.id == user_id).first()
    if not utilisateur:
        raise HTTPException(status_code=404, detail="Utilisateur non trouvé")

    if not utilisateur.is_active:
        raise HTTPException(status_code=400, detail="Utilisateur déjà désactivé")

    utilisateur.is_active = False
    db.commit()
    db.refresh(utilisateur)

    return {"message": f"Utilisateur {utilisateur.email} archivé avec succès."}


@app.patch("/admin/unarchive_user/{user_id}", tags=["Administration : Utilisateurs"])
def unarchive_user(user_id: int, db: Session = Depends(get_db), admin: User = Depends(require_roles("admin"))):
    utilisateur = db.query(User).filter(User.id == user_id).first()
    if not utilisateur:
        raise HTTPException(status_code=404, detail="Utilisateur non trouvé")

    if utilisateur.is_active:
        raise HTTPException(status_code=400, detail="Utilisateur déjà actif")

    utilisateur.is_active = True
    db.commit()
    db.refresh(utilisateur)

    return {"message": f"Utilisateur {utilisateur.email} restauré avec succès."}


@app.get("/admin/user/{user_id}", tags=["Administration : Utilisateurs"])
def get_user_details(user_id: int, db: Session = Depends(get_db), admin: User = Depends(require_roles("admin"))):
    """
    Récupère les détails d'un utilisateur par son ID (admin only).
    """
    utilisateur = db.query(User).filter(User.id == user_id).first()
    if not utilisateur:
        raise HTTPException(status_code=404, detail="Utilisateur introuvable")
    return {
        "id": utilisateur.id,
        "nom": utilisateur.nom,
        "prenom": utilisateur.prenom,
        "email": utilisateur.email,
        "role": utilisateur.role,
        "is_active": utilisateur.is_active
    }

@app.put("/admin/edit_users_profil/{user_id}", tags=["Administration : Utilisateurs"])
def modifier_utilisateur(user_id: int, update_data: UserUpdate, db: Session = Depends(get_db), admin: User = Depends(require_roles("admin"))):
    """
    Met à jour les informations d'un utilisateur spécifique.
    """
    utilisateur = db.query(User).filter(User.id == user_id).first()
    if not utilisateur:
        raise HTTPException(status_code=404, detail="Utilisateur introuvable")
    if update_data.nom:
        utilisateur.nom = update_data.nom
    if update_data.prenom:
        utilisateur.prenom = update_data.prenom
    if update_data.email:
        utilisateur.email = update_data.email
    if update_data.role:
        utilisateur.role = update_data.role
    if update_data.mot_de_passe:
        utilisateur.mot_de_passe = get_password_hash(update_data.mot_de_passe)
    db.commit()
    db.refresh(utilisateur)
    return {"message": "Utilisateur mis à jour avec succès", "utilisateur": utilisateur.email}



# --- Dashboard admin ---
@app.get("/admin/dashboard", tags=["Administration"])
def tableau_de_bord(admin: User = Depends(require_roles("admin")), db: Session = Depends(get_db)):
    """
    Retourne les métriques principales pour l'administration.
    """
    stats = get_dashboard_metrics(db)
    return {
        "total_utilisateurs": stats["total_users"],
        "total_questions_posees": stats["total_questions"],
        "utilisateurs_par_role": stats["users_by_role"],
        "documents_disponibles": stats["documents_in_corpus"],
        "nombre_total_documents": stats["total_documents"],
        "questions_dernieres_24h": stats["recent_questions_24h"]
    }


@app.get("/admin/recent_activity", tags=["Administration"])
def recent_activity(admin: User = Depends(require_roles("admin")), db: Session = Depends(get_db)):
    """
    Retourne le nombre de questions posées pour les 7 derniers jours pour créer un graphique.
    """
    today = date.today()
    dates_to_check = [(today - timedelta(days=i)) for i in range(6, -1, -1)]
    start_date = dates_to_check[0]
    query = db.query(func.date(Historique.date_question).label("day"), func.count(Historique.id).label("total")).filter(func.date(Historique.date_question) >= start_date).group_by(func.date(Historique.date_question)).all()
    count_by_day_str = {q.day: q.total for q in query}
    return [{"name": d.isoformat(), "value": count_by_day_str.get(d.isoformat(), 0)} for d in dates_to_check]

#--- Gestion des feedbacks----
@app.get("/admin/feedback/negative", response_model=List[FeedbackResponse],  tags=["Administration : Feedback"])
def get_negative_feedback(db: Session = Depends(get_db), admin: User = Depends(require_roles("admin"))):
    """
    Retourne tous les messages d'historique jugés NON utiles (is_helpful = False)
    pour que l'équipe puisse les analyser et corriger.
    """
    feedback_negatif = (
        db.query(Historique)
        .filter(Historique.is_helpful == False)
        .filter(Historique.resolved == False)  
        .all()
    )
    return feedback_negatif

@app.post("/admin/feedback/resolve/{feedback_id}", response_model=FeedbackResponse,  tags=["Administration : Feedback"])
def resolve_feedback(
    feedback_id: int,
    payload: FeedbackResolveModel,
    db: Session = Depends(get_db),
    admin: User = Depends(require_roles("admin"))
):
    feedback = db.query(Historique).filter(Historique.id == feedback_id).first()
    if not feedback:
        raise HTTPException(status_code=404, detail="Feedback non trouvé")

    feedback.error_type = payload.error_type
    feedback.action_taken = payload.action_taken
    feedback.comment = payload.comment
    feedback.resolved = True

    db.commit()
    db.refresh(feedback)

    return feedback

@app.get("/admin/feedback/resolved", response_model=List[FeedbackResponse],  tags=["Administration : Feedback"])
def get_resolved_feedback(db: Session = Depends(get_db), admin: User = Depends(require_roles("admin"))):
    return db.query(Historique).filter(Historique.resolved == True).all()


@app.get("/admin/feedback/{feedback_id}", response_model=FeedbackResponse, tags=["Administration : Feedback"])
def get_feedback_by_id(
    feedback_id: int,
    db: Session = Depends(get_db),
    admin: User = Depends(require_roles("admin"))
):
    """
    Retourne un feedback spécifique pour affichage/traitement
    """
    feedback = db.query(Historique).filter(Historique.id == feedback_id).first()
    if not feedback:
        raise HTTPException(status_code=404, detail="Feedback non trouvé")
    
    return feedback


# --- Gestion du corpus ---
@app.get("/admin/corpus/list", tags=["Administration : Corpus"])
def liste_documents_corpus(admin: User = Depends(require_roles("admin"))):
    """Liste tous les documents actifs dans le corpus."""
    return {"documents": get_corpus_documents(active_only=True)}

@app.get("/admin/corpus/archived", tags=["Administration : Corpus"])
def liste_documents_archives(admin: User = Depends(require_roles("admin"))):
    """Liste tous les documents archivés dans le corpus."""
    return {"documents": get_corpus_documents(active_only=False)}

@app.post("/admin/corpus/upload", tags=["Administration : Corpus"])
async def upload_document(file: UploadFile = File(...), admin: User = Depends(require_roles("admin"))):
    """
    Upload d’un document PDF vers le corpus.
    - Vérifie l’extension du fichier.
    - Sauvegarde dans le corpus.
    """
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Seuls les fichiers PDF sont acceptés.")
    file_content = await file.read()
    save_path = add_document_to_corpus(file_content, file.filename)
    return {"message": f"Fichier {file.filename} téléchargé. Veuillez relancer l'ingestion manuellement.", "path": str(save_path)}

@app.post("/admin/corpus/toggle/{filename}", tags=["Administration : Corpus"])
def toggle_document(filename: str, admin: User = Depends(require_roles("admin"))):
    """Active ou archive un document existant dans le corpus."""
    try:
        new_status = toggle_document_status(filename)
        return {"message": f"Le document '{filename}' a été {'activé' if new_status else 'archivé'} avec succès.", "is_active": new_status}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Fichier {filename} introuvable dans le corpus.")

@app.post("/admin/corpus/ingest/full", tags=["Administration : Corpus"])
def relancer_ingestion_async(db: Session = Depends(get_db), admin: User = Depends(require_roles("admin"))):
    """Relance l’ingestion complète du corpus en tâche de fond."""
    job_id = start_ingestion_job(db=db, user=admin, reset=True, update_chain_func=update_main_qa_chain)
    return {"message": "Ingestion lancée en tâche de fond. Consultez /admin/jobs pour le statut.", "job_id": job_id, "status_endpoint": f"/admin/jobs/{job_id}"}

# --- Gestion des jobs ---
@app.get("/admin/jobs", tags=["Administration : Jobs"])
def list_jobs(db: Session = Depends(get_db), admin: User = Depends(require_roles("admin"))):
    """Liste tous les jobs en base (historique)."""
    return get_all_jobs(db=db)

@app.get("/admin/jobs/{job_id}", tags=["Administration : Jobs"])
def get_job_info(job_id: str, db: Session = Depends(get_db), admin: User = Depends(require_roles("admin"))):
    """Retourne le statut d’un job spécifique par son ID."""
    status = get_job_status(db=db, job_id=job_id)
    if not status: raise HTTPException(status_code=404, detail="Job ID introuvable.")
    return status

@app.post("/admin/jobs/{job_id}/stop", tags=["Administration : Jobs"])
def stop_job(job_id: str, db: Session = Depends(get_db), admin: User = Depends(require_roles("admin"))):
    """Arrête un job en cours ou en queue."""
    try: return stop_ingestion_job(db=db, job_id=job_id)
    except ValueError as e: raise HTTPException(status_code=404, detail=str(e))
    except Exception as e: raise HTTPException(status_code=500, detail=f"Erreur lors de l'annulation du job : {str(e)}")

@app.delete("/admin/jobs/{job_id}", tags=["Administration : Jobs"])
def delete_job_endpoint(job_id: str, db: Session = Depends(get_db), admin: User = Depends(require_roles("admin"))):
    """Supprime un job de l’historique."""
    try: return delete_job(db=db, job_id=job_id)
    except ValueError as e: raise HTTPException(status_code=400, detail=str(e))
    except Exception as e: raise HTTPException(status_code=500, detail=f"Erreur lors de la suppression du job : {str(e)}")


# --- Chercheur & Rapports ---
@app.get("/chercheur/documents", tags=["Chercheur & Rapports"])
def consulter_documents(chercheur: User = Depends(require_roles("chercheur", "admin"))):
    """Récupère tous les documents du corpus pour le chercheur."""
    return {"documents": get_corpus_documents()}

@app.get("/export", tags=["Chercheur & Rapports"], dependencies=[Depends(require_roles("chercheur", "admin"))])
def exporter_donnees(db: Session = Depends(get_db)):
    """Exporte l'historique global en CSV."""
    csv_path = export_global_history_csv(db)
    if not csv_path or not os.path.exists(csv_path):
        raise HTTPException(status_code=500, detail="Le fichier d'export n'a pas pu être généré.")
