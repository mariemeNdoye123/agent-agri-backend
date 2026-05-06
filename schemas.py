# Fichier: schemas.py
from pydantic import BaseModel, EmailStr
from typing import List, Dict, Optional,Union 
from datetime import datetime

# --- Modèles d'authentification ---
class RegisterModel(BaseModel):
    """Schéma pour l'inscription d'un nouvel utilisateur."""
    nom: str
    prenom: str
    email: EmailStr
    mot_de_passe: str
    role: str = "agriculteur"

class UserInfo(BaseModel):
    """Schéma pour les informations de base de l'utilisateur."""
    email: EmailStr
    nom: str
    prenom: str
    role: str

# --- Modèles RAG et Historique ---
class Question(BaseModel):
    """Schéma pour la question de l'utilisateur."""
    query: str

class FeedbackModel(BaseModel):
    """Schéma pour l'envoi du feedback d'un utilisateur sur une réponse."""
    historique_id: int # ID de l'entrée Historique à noter (celle qui contient la Q/R)
    is_helpful: bool  # True pour OK, False pour NOK
   
class FeedbackResolveModel(BaseModel):
    """Schéma pour résoudre un feedback."""
    error_type: str
    action_taken: str
    comment: Optional[str] = None

class FeedbackResponse(BaseModel):
    id: int
    question: str
    reponse: str
    is_helpful: bool
    error_type: Optional[str] = None
    action_taken: Optional[str] = None
    comment: Optional[str] = None  # comment admin
    resolved: bool
    user_id: Optional[int] = None
    chat_id: Optional[int] = None
    date_question: Optional[datetime] = None
    class Config:
        orm_mode = True


class Source(BaseModel):
    """Schéma pour un document source dans la réponse."""
    file: str
    page: int

class RAGResponse(BaseModel):
    """Schéma de la réponse complète de l'agent RAG."""
    question: str
    answer: str
    sources: List[Dict]

class HistoriqueEntry(BaseModel):
    """Schéma d'une entrée de l'historique de l'utilisateur."""
    question: str
    reponse: str
    sources: Optional[Union[str, List[str]]] = None
    date: str

class HistoriqueResponse(BaseModel):
    id: int
    question: str
    reponse: str
    sources: List[dict] = []
    date: datetime
    is_helpful: Optional[bool] = None

    class Config:
        orm_mode = True

class ChatResponse(BaseModel):
    id: int
    titre: Optional[str]
    date_creation: datetime
    messages: List[HistoriqueResponse] = []

    class Config:
        orm_mode = True


class UpdateProfileModel(BaseModel):
    """Schéma pour la mise à jour du profil utilisateur."""
    nom: Optional[str] = None
    prenom: Optional[str] = None
    mot_de_passe: Optional[str] = None

class UserUpdate(BaseModel):
    """Schéma utilisé pour la modification complète d’un utilisateur (par un admin)."""
    nom: Optional[str] = None
    prenom: Optional[str] = None
    email: Optional[EmailStr] = None
    mot_de_passe: Optional[str] = None
    role: Optional[str] = None


class DocumentCreate(BaseModel):
    """Schéma pour l’ajout d’un document."""
    titre: str
    description: Optional[str] = None

class DocumentStatusUpdate(BaseModel):
    """Schéma utilisé pour basculer l'état actif/inactif d'un document."""
    is_active: bool

class DocumentInfo(BaseModel):
    """Schéma pour afficher les informations d’un document."""
    id: int
    titre: str
    description: Optional[str]
    fichier: str
    is_active: bool
    uploader_id: int
    uploader: str

    class Config:
        orm_mode = True

    
