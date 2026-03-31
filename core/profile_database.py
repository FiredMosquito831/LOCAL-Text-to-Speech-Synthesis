"""
Speaker Profile Database
=========================
SQLite-based storage for voice profiles with embeddings.
"""

import logging
import json
import sqlite3
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import base64
import shutil

logger = logging.getLogger(__name__)


@dataclass
class SpeakerProfile:
    """Voice profile data structure."""
    
    id: str
    name: str
    tags: List[str]
    description: str
    
    # Audio samples
    reference_samples: List[Dict[str, Any]]  # [{path, transcript, duration_s}]
    
    # Computed data
    embedding: Optional[str] = None  # Base64 encoded tensor
    quality_score: float = 0.0
    
    # Metadata
    created_at: str = ""
    updated_at: str = ""
    backend_compatibility: List[str] = None
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        if not self.updated_at:
            self.updated_at = self.created_at
        if self.backend_compatibility is None:
            self.backend_compatibility = ["f5-tts", "gpt-sovits", "fish-speech"]
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> "SpeakerProfile":
        return cls(**data)
    
    def to_vsp(self) -> bytes:
        """Export profile as .vsp (VoiceSynth Profile) bytes."""
        return json.dumps(self.to_dict(), indent=2).encode('utf-8')
    
    @classmethod
    def from_vsp(cls, data: bytes) -> "SpeakerProfile":
        """Import profile from .vsp bytes."""
        return cls.from_dict(json.loads(data.decode('utf-8')))


class ProfileDatabase:
    """
    SQLite database for speaker profiles.
    
    Features:
    - Profile CRUD operations
    - Search by name/tags
    - Embedding storage
    - Sample management
    """
    
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = Path(db_path or "~/.config/voicesynth/profiles.db").expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._init_db()
    
    def _init_db(self):
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS profiles (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    tags TEXT,
                    description TEXT,
                    embedding TEXT,
                    quality_score REAL DEFAULT 0.0,
                    created_at TEXT,
                    updated_at TEXT,
                    backend_compatibility TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS samples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    profile_id TEXT NOT NULL,
                    audio_path TEXT NOT NULL,
                    transcript TEXT,
                    duration_s REAL,
                    FOREIGN KEY (profile_id) REFERENCES profiles(id)
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_profile_name ON profiles(name)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_samples_profile ON samples(profile_id)
            """)
            
            conn.commit()
        
        logger.info(f"Profile database initialized at {self.db_path}")
    
    def save_profile(self, profile: SpeakerProfile) -> bool:
        """Save or update a profile."""
        try:
            profile.updated_at = datetime.now().isoformat()
            
            with sqlite3.connect(self.db_path) as conn:
                # Check if exists
                existing = conn.execute(
                    "SELECT id FROM profiles WHERE id = ?",
                    (profile.id,)
                ).fetchone()
                
                if existing:
                    # Update
                    conn.execute("""
                        UPDATE profiles SET
                            name = ?,
                            tags = ?,
                            description = ?,
                            embedding = ?,
                            quality_score = ?,
                            updated_at = ?,
                            backend_compatibility = ?
                        WHERE id = ?
                    """, (
                        profile.name,
                        json.dumps(profile.tags),
                        profile.description,
                        profile.embedding,
                        profile.quality_score,
                        profile.updated_at,
                        json.dumps(profile.backend_compatibility),
                        profile.id
                    ))
                else:
                    # Insert
                    conn.execute("""
                        INSERT INTO profiles (
                            id, name, tags, description, embedding,
                            quality_score, created_at, updated_at, backend_compatibility
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        profile.id,
                        profile.name,
                        json.dumps(profile.tags),
                        profile.description,
                        profile.embedding,
                        profile.quality_score,
                        profile.created_at,
                        profile.updated_at,
                        json.dumps(profile.backend_compatibility)
                    ))
                
                # Update samples
                conn.execute("DELETE FROM samples WHERE profile_id = ?", (profile.id,))
                for sample in profile.reference_samples:
                    conn.execute("""
                        INSERT INTO samples (profile_id, audio_path, transcript, duration_s)
                        VALUES (?, ?, ?, ?)
                    """, (
                        profile.id,
                        sample.get("path", ""),
                        sample.get("transcript", ""),
                        sample.get("duration_s", 0)
                    ))
                
                conn.commit()
            
            logger.info(f"Saved profile: {profile.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save profile: {e}")
            return False
    
    def get_profile(self, profile_id: str) -> Optional[SpeakerProfile]:
        """Get a profile by ID."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                
                row = conn.execute(
                    "SELECT * FROM profiles WHERE id = ?",
                    (profile_id,)
                ).fetchone()
                
                if not row:
                    return None
                
                # Get samples
                samples = conn.execute(
                    "SELECT audio_path, transcript, duration_s FROM samples WHERE profile_id = ?",
                    (profile_id,)
                ).fetchall()
                
                return SpeakerProfile(
                    id=row["id"],
                    name=row["name"],
                    tags=json.loads(row["tags"] or "[]"),
                    description=row["description"] or "",
                    embedding=row["embedding"],
                    quality_score=row["quality_score"] or 0.0,
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                    backend_compatibility=json.loads(row["backend_compatibility"] or "[]"),
                    reference_samples=[
                        {"path": s["audio_path"], "transcript": s["transcript"], "duration_s": s["duration_s"]}
                        for s in samples
                    ]
                )
                
        except Exception as e:
            logger.error(f"Failed to get profile: {e}")
            return None
    
    def list_profiles(self, search: str = "", tags: List[str] = None) -> List[SpeakerProfile]:
        """List profiles with optional search/filter."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                
                query = "SELECT id FROM profiles WHERE 1=1"
                params = []
                
                if search:
                    query += " AND (name LIKE ? OR tags LIKE ?)"
                    params.extend([f"%{search}%", f"%{search}%"])
                
                if tags:
                    for tag in tags:
                        query += " AND tags LIKE ?"
                        params.append(f"%{tag}%")
                
                query += " ORDER BY updated_at DESC"
                
                rows = conn.execute(query, params).fetchall()
                
                profiles = []
                for row in rows:
                    profile = self.get_profile(row["id"])
                    if profile:
                        profiles.append(profile)
                
                return profiles
                
        except Exception as e:
            logger.error(f"Failed to list profiles: {e}")
            return []
    
    def delete_profile(self, profile_id: str) -> bool:
        """Delete a profile."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM samples WHERE profile_id = ?", (profile_id,))
                conn.execute("DELETE FROM profiles WHERE id = ?", (profile_id,))
                conn.commit()
            
            logger.info(f"Deleted profile: {profile_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete profile: {e}")
            return False
    
    def export_profile(self, profile_id: str, output_path: Path) -> bool:
        """Export profile as .vsp file."""
        profile = self.get_profile(profile_id)
        if not profile:
            return False
        
        try:
            output_path = Path(output_path)
            
            # Create export directory
            export_dir = output_path.parent / f"{profile.name}_export"
            export_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy audio samples
            for i, sample in enumerate(profile.reference_samples):
                src = Path(sample["path"])
                if src.exists():
                    dst = export_dir / f"sample_{i}{src.suffix}"
                    shutil.copy2(src, dst)
                    sample["path"] = str(dst.name)  # Relative path
            
            # Write profile
            with open(output_path, "wb") as f:
                f.write(profile.to_vsp())
            
            logger.info(f"Exported profile to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export profile: {e}")
            return False
    
    def import_profile(self, vsp_path: Path) -> Optional[SpeakerProfile]:
        """Import profile from .vsp file."""
        try:
            with open(vsp_path, "rb") as f:
                profile = SpeakerProfile.from_vsp(f.read())
            
            # Update paths to absolute
            vsp_dir = vsp_path.parent
            for sample in profile.reference_samples:
                sample_path = vsp_dir / sample["path"]
                if sample_path.exists():
                    sample["path"] = str(sample_path)
            
            # Save to database
            self.save_profile(profile)
            
            logger.info(f"Imported profile: {profile.name}")
            return profile
            
        except Exception as e:
            logger.error(f"Failed to import profile: {e}")
            return None


class QualityScorer:
    """
    Score the quality of voice reference samples.
    
    Metrics:
    - SNR (Signal-to-Noise Ratio)
    - Clarity (spectral analysis)
    - Duration appropriateness
    - Consistency across samples
    """
    
    def __init__(self):
        self._sr = 24000
    
    def score_sample(self, audio_path: Path) -> Dict[str, float]:
        """Score a single audio sample."""
        try:
            import numpy as np
            import soundfile as sf
            
            # Load audio
            audio, sr = sf.read(str(audio_path))
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)  # Mono
            
            # Duration score (3-30 seconds ideal)
            duration = len(audio) / sr
            if 3 <= duration <= 30:
                duration_score = 1.0
            elif duration < 3:
                duration_score = duration / 3
            else:
                duration_score = max(0.5, 1 - (duration - 30) / 60)
            
            # RMS energy (avoid silent/clipped audio)
            rms = np.sqrt(np.mean(audio**2))
            if 0.01 < rms < 0.5:
                energy_score = 1.0
            elif rms <= 0.01:
                energy_score = rms / 0.01
            else:
                energy_score = max(0.5, 1 - (rms - 0.5))
            
            # Peak detection (avoid clipping)
            peak = np.max(np.abs(audio))
            if peak > 0.99:
                clipping_score = 0.5
            else:
                clipping_score = 1.0
            
            # Silence ratio
            silence_threshold = 0.01
            silence_ratio = np.mean(np.abs(audio) < silence_threshold)
            silence_score = max(0, 1 - silence_ratio * 2)
            
            # Overall score
            overall = (
                duration_score * 0.25 +
                energy_score * 0.25 +
                clipping_score * 0.25 +
                silence_score * 0.25
            )
            
            return {
                "duration_score": duration_score,
                "energy_score": energy_score,
                "clipping_score": clipping_score,
                "silence_score": silence_score,
                "overall": overall,
                "duration_s": duration
            }
            
        except Exception as e:
            logger.error(f"Failed to score sample: {e}")
            return {"overall": 0.5, "error": str(e)}
    
    def score_profile(self, profile: SpeakerProfile) -> float:
        """Score an entire profile based on all samples."""
        if not profile.reference_samples:
            return 0.0
        
        scores = []
        for sample in profile.reference_samples:
            path = Path(sample.get("path", ""))
            if path.exists():
                result = self.score_sample(path)
                scores.append(result.get("overall", 0.5))
        
        if not scores:
            return 0.0
        
        # Average score with bonus for multiple samples
        avg_score = sum(scores) / len(scores)
        multi_sample_bonus = min(0.1, len(scores) * 0.02)
        
        return min(1.0, avg_score + multi_sample_bonus)


class EmbeddingFusion:
    """
    Fuse embeddings from multiple voice samples.
    
    Strategies:
    - Average pooling
    - Weighted average based on quality
    - Attention-based fusion
    """
    
    def fuse_embeddings(
        self,
        embeddings: List[Any],
        weights: Optional[List[float]] = None,
        strategy: str = "weighted_average"
    ) -> Any:
        """
        Fuse multiple embeddings into one.
        
        Args:
            embeddings: List of embedding tensors
            weights: Optional weights for each embedding
            strategy: Fusion strategy
        """
        try:
            import torch
            import numpy as np
            
            if not embeddings:
                return None
            
            # Convert to tensors if needed
            tensors = []
            for emb in embeddings:
                if isinstance(emb, np.ndarray):
                    tensors.append(torch.from_numpy(emb))
                elif isinstance(emb, torch.Tensor):
                    tensors.append(emb)
            
            if not tensors:
                return None
            
            # Stack along new dimension
            stacked = torch.stack(tensors)
            
            if strategy == "average":
                return stacked.mean(dim=0)
            
            elif strategy == "weighted_average":
                if weights is None:
                    weights = [1.0 / len(tensors)] * len(tensors)
                
                weights = torch.tensor(weights).view(-1, 1)
                weights = weights / weights.sum()
                
                return (stacked * weights).sum(dim=0)
            
            elif strategy == "max":
                return stacked.max(dim=0).values
            
            else:
                return stacked.mean(dim=0)
                
        except Exception as e:
            logger.error(f"Embedding fusion failed: {e}")
            return embeddings[0] if embeddings else None
    
    def encode_embedding(self, tensor) -> str:
        """Encode embedding tensor as base64 string for storage."""
        try:
            import torch
            import numpy as np
            
            if isinstance(tensor, torch.Tensor):
                array = tensor.cpu().numpy()
            else:
                array = np.array(tensor)
            
            return base64.b64encode(array.tobytes()).decode('utf-8')
            
        except Exception as e:
            logger.error(f"Failed to encode embedding: {e}")
            return ""
    
    def decode_embedding(self, encoded: str, shape: tuple = None) -> Any:
        """Decode base64 embedding back to tensor."""
        try:
            import torch
            import numpy as np
            
            data = base64.b64decode(encoded.encode('utf-8'))
            array = np.frombuffer(data, dtype=np.float32)
            
            if shape:
                array = array.reshape(shape)
            
            return torch.from_numpy(array)
            
        except Exception as e:
            logger.error(f"Failed to decode embedding: {e}")
            return None


# Global database instance
_profile_db: Optional[ProfileDatabase] = None


def get_profile_database() -> ProfileDatabase:
    """Get the global profile database instance."""
    global _profile_db
    if _profile_db is None:
        _profile_db = ProfileDatabase()
    return _profile_db
