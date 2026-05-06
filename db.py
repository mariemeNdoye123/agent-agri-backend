from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.declarative import declarative_base

# Configuration de la base de données SQLite
DATABASE_URL = "sqlite:///./users.db"

# Déclaration de la base (nécessaire pour les modèles)
Base = declarative_base()

# Création de l'engine (le point de connexion)
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

# Création du SessionLocal pour les requêtes (le "Dialogue" avec la BDD)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    """Dépendance FastAPI pour ouvrir et fermer une session de base de données."""
    db = SessionLocal()
    try:
        # Fournit la session à la route FastAPI
        yield db
    finally:
        # Ferme la session après utilisation (même en cas d'erreur)
        db.close()

