"""
FastAPI backend for a lightweight Spotify-clone.

Includes:
- JWT authentication (signup/login)
- Song catalog + search
- Playlist CRUD (per-user)
- Basic streaming endpoint (file response)
- Admin track upload

Environment variables (from .env):
- POSTGRES_URL, POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DB, POSTGRES_PORT
- ALLOWED_ORIGINS, ALLOWED_HEADERS, ALLOWED_METHODS
- JWT_SECRET (recommended), JWT_EXPIRES_MINUTES (optional)
- ADMIN_EMAILS (optional; comma-separated emails with admin privileges)
- MEDIA_DIR (optional; directory to store uploaded audio files)
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Annotated, Any, Optional

import jwt
from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    Response,
    UploadFile,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    func,
    select,
)
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship
from passlib.context import CryptContext

openapi_tags = [
    {"name": "Health", "description": "Health and diagnostics endpoints."},
    {"name": "Auth", "description": "User authentication and identity endpoints."},
    {"name": "Songs", "description": "Song catalog endpoints (browse, details, search)."},
    {"name": "Playlists", "description": "User playlist management endpoints."},
    {"name": "Streaming", "description": "Audio streaming endpoints."},
    {"name": "Admin", "description": "Administrative endpoints (track upload)."},
]


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    return os.getenv(name, default)


def _require_env(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return v


JWT_SECRET = _env("JWT_SECRET", "dev-insecure-secret-change-me")
JWT_EXPIRES_MINUTES = int(_env("JWT_EXPIRES_MINUTES", "120"))
ADMIN_EMAILS = {
    e.strip().lower()
    for e in (_env("ADMIN_EMAILS", "") or "").split(",")
    if e.strip()
}
MEDIA_DIR = Path(_env("MEDIA_DIR", str(Path(__file__).resolve().parent / "media")))
MEDIA_DIR.mkdir(parents=True, exist_ok=True)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def _get_db_url() -> str:
    """
    Prefer POSTGRES_URL as provided by the environment; fallback to constructed URL.
    """
    url = _env("POSTGRES_URL")
    if url:
        # POSTGRES_URL in this environment already includes DB name and port.
        return url.strip('"')
    user = _require_env("POSTGRES_USER")
    pw = _require_env("POSTGRES_PASSWORD")
    host = "localhost"
    port = _require_env("POSTGRES_PORT")
    db = _require_env("POSTGRES_DB")
    return f"postgresql+psycopg2://{user}:{pw}@{host}:{port}/{db}"


engine = create_engine(_get_db_url(), pool_pre_ping=True)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    display_name: Mapped[str] = mapped_column(String(120), default="")
    is_admin: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    playlists: Mapped[list["Playlist"]] = relationship(back_populates="owner")


class Song(Base):
    __tablename__ = "songs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    title: Mapped[str] = mapped_column(String(200), index=True)
    artist: Mapped[str] = mapped_column(String(200), index=True)
    album: Mapped[str] = mapped_column(String(200), default="")
    duration_ms: Mapped[int] = mapped_column(Integer, default=0)
    # Path to file on disk within MEDIA_DIR
    file_name: Mapped[str] = mapped_column(String(255), unique=True)
    mime_type: Mapped[str] = mapped_column(String(120), default="audio/mpeg")
    cover_url: Mapped[str] = mapped_column(String(500), default="")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )


class Playlist(Base):
    __tablename__ = "playlists"
    __table_args__ = (UniqueConstraint("owner_id", "name", name="uq_playlist_owner_name"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    owner_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    name: Mapped[str] = mapped_column(String(200))
    description: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    owner: Mapped[User] = relationship(back_populates="playlists")
    items: Mapped[list["PlaylistItem"]] = relationship(
        back_populates="playlist", cascade="all, delete-orphan"
    )


class PlaylistItem(Base):
    __tablename__ = "playlist_items"
    __table_args__ = (UniqueConstraint("playlist_id", "song_id", name="uq_playlist_song"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    playlist_id: Mapped[int] = mapped_column(ForeignKey("playlists.id"), index=True)
    song_id: Mapped[int] = mapped_column(ForeignKey("songs.id"), index=True)
    added_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    playlist: Mapped[Playlist] = relationship(back_populates="items")
    song: Mapped[Song] = relationship()


def init_db() -> None:
    Base.metadata.create_all(bind=engine)


# ---- API models ----
class HealthResponse(BaseModel):
    message: str = Field(..., description="Health status message.")


class TokenResponse(BaseModel):
    access_token: str = Field(..., description="JWT access token.")
    token_type: str = Field("bearer", description="Token type.")


class SignupRequest(BaseModel):
    email: EmailStr = Field(..., description="User email.")
    password: str = Field(..., min_length=8, description="User password (min 8 chars).")
    display_name: str = Field("", description="Optional display name.")


class LoginRequest(BaseModel):
    email: EmailStr = Field(..., description="User email.")
    password: str = Field(..., description="User password.")


class UserMeResponse(BaseModel):
    id: int = Field(..., description="User id.")
    email: EmailStr = Field(..., description="User email.")
    display_name: str = Field(..., description="Display name.")
    is_admin: bool = Field(..., description="Whether user is admin.")


class SongResponse(BaseModel):
    id: int = Field(..., description="Song id.")
    title: str = Field(..., description="Song title.")
    artist: str = Field(..., description="Artist name.")
    album: str = Field("", description="Album name.")
    duration_ms: int = Field(0, description="Duration in ms.")
    cover_url: str = Field("", description="Cover URL (optional).")
    stream_url: str = Field(..., description="URL to stream this song.")


class PlaylistResponse(BaseModel):
    id: int = Field(..., description="Playlist id.")
    name: str = Field(..., description="Playlist name.")
    description: str = Field("", description="Playlist description.")
    song_count: int = Field(..., description="Number of songs in playlist.")


class PlaylistDetailResponse(BaseModel):
    id: int = Field(..., description="Playlist id.")
    name: str = Field(..., description="Playlist name.")
    description: str = Field("", description="Playlist description.")
    songs: list[SongResponse] = Field(default_factory=list, description="Songs in playlist.")


class PlaylistCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=200, description="Playlist name.")
    description: str = Field("", description="Playlist description.")


class PlaylistUpdateRequest(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=200, description="New name.")
    description: Optional[str] = Field(None, description="New description.")


class AddSongToPlaylistRequest(BaseModel):
    song_id: int = Field(..., description="Song id to add.")


# ---- Auth helpers ----
def _hash_password(password: str) -> str:
    return pwd_context.hash(password)


def _verify_password(password: str, password_hash: str) -> bool:
    return pwd_context.verify(password, password_hash)


def _create_access_token(user: User) -> str:
    now = datetime.now(timezone.utc)
    payload: dict[str, Any] = {
        "sub": str(user.id),
        "email": user.email,
        "is_admin": user.is_admin,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=JWT_EXPIRES_MINUTES)).timestamp()),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")


def _decode_token(token: str) -> dict[str, Any]:
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
    except jwt.ExpiredSignatureError as e:
        raise HTTPException(status_code=401, detail="Token expired") from e
    except jwt.PyJWTError as e:
        raise HTTPException(status_code=401, detail="Invalid token") from e


def _get_bearer_token(request: Request) -> str:
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    return auth[len("Bearer ") :].strip()


def get_db_session() -> Session:
    with Session(engine) as session:
        yield session


DbDep = Annotated[Session, Depends(get_db_session)]


def get_current_user(request: Request, db: DbDep) -> User:
    token = _get_bearer_token(request)
    payload = _decode_token(token)
    user_id = int(payload.get("sub", "0"))
    user = db.get(User, user_id)
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user


CurrentUserDep = Annotated[User, Depends(get_current_user)]


def require_admin(user: CurrentUserDep) -> User:
    if user.is_admin:
        return user
    # allow config-based admins by email
    if user.email.lower() in ADMIN_EMAILS:
        return user
    raise HTTPException(status_code=403, detail="Admin access required")


AdminDep = Annotated[User, Depends(require_admin)]


# ---- App ----
app = FastAPI(
    title="MusicStream API",
    description="Backend API for a Spotify-clone: auth, songs, playlists, search, and streaming.",
    version="1.0.0",
    openapi_tags=openapi_tags,
)

allowed_origins = [o.strip() for o in (_env("ALLOWED_ORIGINS", "*") or "*").split(",")]
allowed_headers = [h.strip() for h in (_env("ALLOWED_HEADERS", "*") or "*").split(",")]
allowed_methods = [m.strip() for m in (_env("ALLOWED_METHODS", "*") or "*").split(",")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins if allowed_origins != ["*"] else ["*"],
    allow_credentials=True,
    allow_methods=allowed_methods if allowed_methods != ["*"] else ["*"],
    allow_headers=allowed_headers if allowed_headers != ["*"] else ["*"],
)

init_db()


@app.get(
    "/",
    tags=["Health"],
    summary="Health check",
    description="Basic health check endpoint.",
    response_model=HealthResponse,
)
# PUBLIC_INTERFACE
def health_check() -> HealthResponse:
    """Return service health."""
    return HealthResponse(message="Healthy")


@app.get(
    "/healthz",
    tags=["Health"],
    summary="Health check (k8s-style)",
    description="Health check endpoint used by frontend template / infra.",
    response_model=HealthResponse,
)
# PUBLIC_INTERFACE
def healthz() -> HealthResponse:
    """Return service health."""
    return HealthResponse(message="OK")


@app.post(
    "/auth/signup",
    tags=["Auth"],
    summary="Create user account",
    description="Create a new user and return an access token.",
    response_model=TokenResponse,
    status_code=201,
)
# PUBLIC_INTERFACE
def signup(payload: SignupRequest, db: DbDep) -> TokenResponse:
    """Create a user account."""
    user = User(
        email=payload.email.lower(),
        password_hash=_hash_password(payload.password),
        display_name=payload.display_name or payload.email.split("@")[0],
        is_admin=(payload.email.lower() in ADMIN_EMAILS),
    )
    db.add(user)
    try:
        db.commit()
    except IntegrityError as e:
        db.rollback()
        raise HTTPException(status_code=409, detail="Email already registered") from e
    db.refresh(user)
    return TokenResponse(access_token=_create_access_token(user))


@app.post(
    "/auth/login",
    tags=["Auth"],
    summary="Login",
    description="Login with email/password and return an access token.",
    response_model=TokenResponse,
)
# PUBLIC_INTERFACE
def login(payload: LoginRequest, db: DbDep) -> TokenResponse:
    """Authenticate user with email/password."""
    user = db.scalar(select(User).where(User.email == payload.email.lower()))
    if not user or not _verify_password(payload.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return TokenResponse(access_token=_create_access_token(user))


@app.get(
    "/auth/me",
    tags=["Auth"],
    summary="Get current user",
    description="Return the currently authenticated user.",
    response_model=UserMeResponse,
)
# PUBLIC_INTERFACE
def me(user: CurrentUserDep) -> UserMeResponse:
    """Return current user profile."""
    return UserMeResponse(
        id=user.id,
        email=user.email,
        display_name=user.display_name,
        is_admin=user.is_admin or (user.email.lower() in ADMIN_EMAILS),
    )


@app.get(
    "/songs",
    tags=["Songs"],
    summary="List songs",
    description="List songs in reverse chronological order.",
    response_model=list[SongResponse],
)
# PUBLIC_INTERFACE
def list_songs(request: Request, db: DbDep, q: Optional[str] = None) -> list[SongResponse]:
    """List songs; optional query `q` filters title/artist/album."""
    stmt = select(Song).order_by(Song.created_at.desc())
    if q:
        like = f"%{q.strip()}%"
        stmt = stmt.where(
            (Song.title.ilike(like)) | (Song.artist.ilike(like)) | (Song.album.ilike(like))
        ).order_by(Song.created_at.desc())
    songs = db.scalars(stmt).all()
    base = str(request.base_url).rstrip("/")
    return [
        SongResponse(
            id=s.id,
            title=s.title,
            artist=s.artist,
            album=s.album,
            duration_ms=s.duration_ms,
            cover_url=s.cover_url,
            stream_url=f"{base}/stream/{s.id}",
        )
        for s in songs
    ]


@app.get(
    "/songs/{song_id}",
    tags=["Songs"],
    summary="Get song",
    description="Get song metadata by id.",
    response_model=SongResponse,
)
# PUBLIC_INTERFACE
def get_song(song_id: int, request: Request, db: DbDep) -> SongResponse:
    """Get a single song."""
    s = db.get(Song, song_id)
    if not s:
        raise HTTPException(status_code=404, detail="Song not found")
    base = str(request.base_url).rstrip("/")
    return SongResponse(
        id=s.id,
        title=s.title,
        artist=s.artist,
        album=s.album,
        duration_ms=s.duration_ms,
        cover_url=s.cover_url,
        stream_url=f"{base}/stream/{s.id}",
    )


@app.get(
    "/stream/{song_id}",
    tags=["Streaming"],
    summary="Stream audio",
    description="Returns audio file for the given song id.",
    responses={
        200: {"description": "Audio stream"},
        404: {"description": "Song not found"},
    },
)
# PUBLIC_INTERFACE
def stream_song(song_id: int, db: DbDep) -> Response:
    """Stream an audio file for a song (simple file response)."""
    s = db.get(Song, song_id)
    if not s:
        raise HTTPException(status_code=404, detail="Song not found")
    file_path = MEDIA_DIR / s.file_name
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Audio file missing on server")
    return FileResponse(path=str(file_path), media_type=s.mime_type, filename=s.file_name)


@app.get(
    "/playlists",
    tags=["Playlists"],
    summary="List my playlists",
    description="List playlists owned by the current user.",
    response_model=list[PlaylistResponse],
)
# PUBLIC_INTERFACE
def list_playlists(user: CurrentUserDep, db: DbDep) -> list[PlaylistResponse]:
    """List current user's playlists."""
    playlists = db.scalars(select(Playlist).where(Playlist.owner_id == user.id)).all()
    out: list[PlaylistResponse] = []
    for p in playlists:
        count = db.scalar(select(func.count()).select_from(PlaylistItem).where(PlaylistItem.playlist_id == p.id))
        out.append(
            PlaylistResponse(
                id=p.id,
                name=p.name,
                description=p.description or "",
                song_count=int(count or 0),
            )
        )
    return out


@app.post(
    "/playlists",
    tags=["Playlists"],
    summary="Create playlist",
    description="Create a new playlist for the current user.",
    response_model=PlaylistResponse,
    status_code=201,
)
# PUBLIC_INTERFACE
def create_playlist(user: CurrentUserDep, payload: PlaylistCreateRequest, db: DbDep) -> PlaylistResponse:
    """Create a playlist."""
    p = Playlist(owner_id=user.id, name=payload.name.strip(), description=payload.description or "")
    db.add(p)
    try:
        db.commit()
    except IntegrityError as e:
        db.rollback()
        raise HTTPException(status_code=409, detail="Playlist name already exists") from e
    db.refresh(p)
    return PlaylistResponse(id=p.id, name=p.name, description=p.description or "", song_count=0)


@app.get(
    "/playlists/{playlist_id}",
    tags=["Playlists"],
    summary="Get playlist",
    description="Get playlist details including songs.",
    response_model=PlaylistDetailResponse,
)
# PUBLIC_INTERFACE
def get_playlist(playlist_id: int, request: Request, user: CurrentUserDep, db: DbDep) -> PlaylistDetailResponse:
    """Get playlist details."""
    p = db.get(Playlist, playlist_id)
    if not p or p.owner_id != user.id:
        raise HTTPException(status_code=404, detail="Playlist not found")
    base = str(request.base_url).rstrip("/")
    items = db.scalars(
        select(PlaylistItem).where(PlaylistItem.playlist_id == playlist_id).order_by(PlaylistItem.added_at.desc())
    ).all()
    songs: list[SongResponse] = []
    for it in items:
        s = it.song
        songs.append(
            SongResponse(
                id=s.id,
                title=s.title,
                artist=s.artist,
                album=s.album,
                duration_ms=s.duration_ms,
                cover_url=s.cover_url,
                stream_url=f"{base}/stream/{s.id}",
            )
        )
    return PlaylistDetailResponse(id=p.id, name=p.name, description=p.description or "", songs=songs)


@app.patch(
    "/playlists/{playlist_id}",
    tags=["Playlists"],
    summary="Update playlist",
    description="Update playlist name/description.",
    response_model=PlaylistResponse,
)
# PUBLIC_INTERFACE
def update_playlist(
    playlist_id: int, user: CurrentUserDep, payload: PlaylistUpdateRequest, db: DbDep
) -> PlaylistResponse:
    """Update a playlist."""
    p = db.get(Playlist, playlist_id)
    if not p or p.owner_id != user.id:
        raise HTTPException(status_code=404, detail="Playlist not found")

    if payload.name is not None:
        p.name = payload.name.strip()
    if payload.description is not None:
        p.description = payload.description

    try:
        db.commit()
    except IntegrityError as e:
        db.rollback()
        raise HTTPException(status_code=409, detail="Playlist name already exists") from e

    count = db.scalar(select(func.count()).select_from(PlaylistItem).where(PlaylistItem.playlist_id == p.id))
    return PlaylistResponse(id=p.id, name=p.name, description=p.description or "", song_count=int(count or 0))


@app.delete(
    "/playlists/{playlist_id}",
    tags=["Playlists"],
    summary="Delete playlist",
    description="Delete a playlist and its items.",
    status_code=204,
)
# PUBLIC_INTERFACE
def delete_playlist(playlist_id: int, user: CurrentUserDep, db: DbDep) -> Response:
    """Delete a playlist."""
    p = db.get(Playlist, playlist_id)
    if not p or p.owner_id != user.id:
        raise HTTPException(status_code=404, detail="Playlist not found")
    db.delete(p)
    db.commit()
    return Response(status_code=204)


@app.post(
    "/playlists/{playlist_id}/songs",
    tags=["Playlists"],
    summary="Add song to playlist",
    description="Add a song to a playlist.",
    status_code=201,
)
# PUBLIC_INTERFACE
def add_song_to_playlist(
    playlist_id: int, user: CurrentUserDep, payload: AddSongToPlaylistRequest, db: DbDep
) -> dict[str, Any]:
    """Add a song to a playlist."""
    p = db.get(Playlist, playlist_id)
    if not p or p.owner_id != user.id:
        raise HTTPException(status_code=404, detail="Playlist not found")
    s = db.get(Song, payload.song_id)
    if not s:
        raise HTTPException(status_code=404, detail="Song not found")

    it = PlaylistItem(playlist_id=playlist_id, song_id=s.id)
    db.add(it)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        # already exists
        return {"status": "exists"}
    return {"status": "added"}


@app.delete(
    "/playlists/{playlist_id}/songs/{song_id}",
    tags=["Playlists"],
    summary="Remove song from playlist",
    description="Remove a song from a playlist.",
    status_code=204,
)
# PUBLIC_INTERFACE
def remove_song_from_playlist(playlist_id: int, song_id: int, user: CurrentUserDep, db: DbDep) -> Response:
    """Remove a song from a playlist."""
    p = db.get(Playlist, playlist_id)
    if not p or p.owner_id != user.id:
        raise HTTPException(status_code=404, detail="Playlist not found")

    item = db.scalar(
        select(PlaylistItem).where(PlaylistItem.playlist_id == playlist_id, PlaylistItem.song_id == song_id)
    )
    if not item:
        raise HTTPException(status_code=404, detail="Song not in playlist")
    db.delete(item)
    db.commit()
    return Response(status_code=204)


@app.post(
    "/admin/upload",
    tags=["Admin"],
    summary="Upload a new track",
    description="Admin-only: upload an audio file and create a song record.",
    response_model=SongResponse,
)
# PUBLIC_INTERFACE
async def admin_upload(
    request: Request,
    _: AdminDep,
    db: DbDep,
    title: Annotated[str, Form(..., description="Track title")],
    artist: Annotated[str, Form(..., description="Artist")],
    album: Annotated[str, Form("", description="Album")],
    duration_ms: Annotated[int, Form(0, description="Duration in milliseconds")],
    cover_url: Annotated[str, Form("", description="Cover URL")],
    file: Annotated[UploadFile, File(..., description="Audio file")],
) -> SongResponse:
    """Upload a track file to server storage and create song metadata."""
    # Store file under a unique name
    timestamp = int(datetime.now(timezone.utc).timestamp())
    safe_name = "".join([c for c in (file.filename or "track") if c.isalnum() or c in (".", "_", "-")])
    stored_name = f"{timestamp}_{safe_name}"
    dest = MEDIA_DIR / stored_name

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file")
    dest.write_bytes(content)

    s = Song(
        title=title.strip(),
        artist=artist.strip(),
        album=album.strip() if album else "",
        duration_ms=duration_ms or 0,
        file_name=stored_name,
        mime_type=file.content_type or "audio/mpeg",
        cover_url=cover_url or "",
    )
    db.add(s)
    db.commit()
    db.refresh(s)

    base = str(request.base_url).rstrip("/")
    return SongResponse(
        id=s.id,
        title=s.title,
        artist=s.artist,
        album=s.album,
        duration_ms=s.duration_ms,
        cover_url=s.cover_url,
        stream_url=f"{base}/stream/{s.id}",
    )
