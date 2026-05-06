# security.py
from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from typing import Optional
from db import get_db
from models import User

# ================== CONFIGURATION SÉCURITÉ ==================
SECRET_KEY = "SECRET_KEY_CHANGE_THIS"  # Changer en vrai secret en prod
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_DAYS = 30  # Durée de validité du token (30 jours)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login", auto_error=False)
# ============================================================

# === UTILITAIRES DE HASHAGE ===
def get_password_hash(password: str) -> str:
    """Hashage sécurisé du mot de passe."""
   
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Vérifie qu’un mot de passe correspond au hash stocké."""
    
    return pwd_context.verify(plain_password, hashed_password)


# === GESTION DES TOKENS JWT ===
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Crée un token JWT avec une date d’expiration."""
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(days=ACCESS_TOKEN_EXPIRE_DAYS))
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def get_current_user(db: Session = Depends(get_db), token: Optional[str] = Depends(oauth2_scheme)) -> Optional[User]:
    """
    Retourne l'utilisateur si connecté, sinon None
    """
    if not token:
        return None

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if not email:
            return None
        user = db.query(User).filter(User.email == email).first()
        return user
    except JWTError:
        return None


# --- CONTRÔLE DES RÔLES ---
def require_roles(*roles_autorises):
    """
    Décorateur FastAPI pour restreindre l'accès à certaines routes en fonction du rôle utilisateur.
    """

    def verifier_roles(utilisateur: User = Depends(get_current_user)):
        if not utilisateur:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Utilisateur non authentifié"
            )
        if utilisateur.role not in roles_autorises:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Accès refusé : rôle '{utilisateur.role}' non autorisé pour cette action."
            )
        return utilisateur

    return verifier_roles
