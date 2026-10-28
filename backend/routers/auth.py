from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
import jwt
from pydantic import BaseModel
from backend.database import get_db
from backend.models.user import User
from backend.services.auth_service import get_password_hash, verify_password, create_access_token
from backend.config import settings

router = APIRouter()
security = HTTPBearer()

class AuthRequest(BaseModel):
    phone_or_email: str
    password: str

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security), db: Session = Depends(get_db)):
    token = credentials.credentials
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid token")
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return user

@router.post("/register", status_code=status.HTTP_201_CREATED)
def register(request: AuthRequest, db: Session = Depends(get_db)):
    existing = db.query(User).filter(User.phone_or_email == request.phone_or_email).first()
    if existing:
        raise HTTPException(status_code=400, detail="User already exists")
    
    user = User(
        phone_or_email=request.phone_or_email,
        password_hash=get_password_hash(request.password)
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    
    token = create_access_token(data={"sub": user.id})
    return {"id": user.id, "token": token}

@router.post("/login")
def login(request: AuthRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.phone_or_email == request.phone_or_email).first()
    if not user or not verify_password(request.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    token = create_access_token(data={"sub": user.id})
    return {"token": token, "user": {"id": user.id, "phone_or_email": user.phone_or_email, "is_admin": user.is_admin}}

@router.get("/me")
def get_me(current_user: User = Depends(get_current_user)):
    return {
        "id": current_user.id,
        "phone_or_email": current_user.phone_or_email,
        "is_admin": current_user.is_admin
    }
