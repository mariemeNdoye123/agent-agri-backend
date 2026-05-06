from sqlalchemy import Column, Integer, String, ForeignKey, Text, DateTime,Boolean, Float
from sqlalchemy.orm import relationship
from datetime import datetime
from db import Base  # Importe Base depuis le fichier db.py

# Table Utilisateur
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    nom = Column(String)
    prenom = Column(String)
    email = Column(String, unique=True, index=True, nullable=False)    
    mot_de_passe = Column(String)
    role = Column(String, default="agriculteur")
    is_active = Column(Boolean, default=True) 
    
    # 1. Relation User (un) vers Chat (plusieurs discussions)
    chats = relationship("Chat", back_populates="user")
    # 2. Relation User (un) vers Historique (tous les messages de l'utilisateur, pour l'historique global)
    historiques = relationship("Historique", back_populates="user")

# Table Chat (une discussion qui regroupe plusieurs messages)
class Chat(Base):
    __tablename__ = "chats"

    id = Column(Integer, primary_key=True, index=True)
    titre = Column(String, nullable=True)
    date_creation = Column(DateTime, default=datetime.utcnow)

    # Clé étrangère vers l'utilisateur (propriétaire du chat)
    user_id = Column(Integer, ForeignKey("users.id"))
    user = relationship("User", back_populates="chats") 

    # Relation vers l'Historique (messages dans ce chat)
    messages = relationship("Historique", back_populates="chat") 

# Table Historique (questions/réponses liées à un chat et un utilisateur)
class Historique(Base):
    __tablename__ = "historiques"
    id = Column(Integer, primary_key=True, index=True)
    question = Column(Text)
    reponse = Column(Text)
    sources = Column(Text) 
    date_question = Column(DateTime, default=datetime.utcnow)
    #feedbacks
    is_helpful = Column(Boolean, default=None, nullable=True) 
    error_type = Column(String, nullable=True)
    action_taken = Column(String, nullable=True)
    comment = Column(String, nullable=True)
    resolved = Column(Boolean, default=False)
    
    # Clé étrangère vers le Chat
    chat_id = Column(Integer, ForeignKey("chats.id"))
    chat = relationship("Chat", back_populates="messages") 

    # Clé étrangère vers l'utilisateur (pour la traçabilité)
    user_id = Column(Integer, ForeignKey("users.id"))
    user = relationship("User", back_populates="historiques")

# Table Document (corpus documentaire)
class Document(Base):
    __tablename__ = "documents"
    id = Column(Integer, primary_key=True, index=True)
    titre = Column(String, nullable=False)
    description = Column(Text)
    fichier = Column(String, nullable=False)
    type_fichier = Column(String, default="pdf") 
    uploader_id = Column(Integer, ForeignKey("users.id"))
    uploader = relationship("User")
    is_active = Column(Boolean, default=True)

# Table Job (Historique des tâches de fond)
class Job(Base):
    """Modèle pour l'enregistrement persistant des tâches de fond (Jobs)."""
    __tablename__ = "jobs"

    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(String, unique=True, index=True, nullable=False) # L'ID généré par time.time()
    job_type = Column(String, nullable=False) # Ex: 'INGESTION_CORPUS'
    status = Column(String, nullable=False, default="EN_QUEUE") # Ex: 'EN_QUEUE', 'EN_COURS', 'TERMINÉ', 'ÉCHEC'
    
    # Progress peut être un Float pour plus de précision si nécessaire, mais Integer est souvent suffisant pour le %
    progress = Column(Integer, default=0) 
    
    message = Column(Text) # Message de statut actuel (plus long qu'un simple String)
    
    start_time = Column(DateTime, default=datetime.utcnow)
    end_time = Column(DateTime, nullable=True)
    
    # Pour stocker les détails du résultat ou de l'erreur (si besoin)
    result_data = Column(Text, nullable=True)

    #relation de l'utilisateur qui a lancé la tâche (bonne pratique)
    launched_by_user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    launched_by = relationship("User", foreign_keys=[launched_by_user_id])
