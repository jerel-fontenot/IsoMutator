"""
ALGORITHM SUMMARY:
This module implements a Cryptographic Response Cache using SQLite and SHA-256.
By hashing the Target AI's responses, it transforms complex, multi-dimensional 
tensor mathematics into an O(1) dictionary lookup. This allows isolated OS 
processes to instantly share evaluation results without recalculating them.

TECHNOLOGY QUIRKS:
- Thread Safety: SQLite allows multiple readers, but concurrent writes can lock.
  We use a short timeout and graceful exception handling so a locked database 
  just results in a cache miss, never a pipeline crash.
"""

import sqlite3
import hashlib

from isomutator.core.log_manager import LogManager
from isomutator.core.config import settings


class ResponseCache:
    """
    A high-speed, multi-process safe cache for AI evaluation results.
    """
    def __init__(self, db_path: str | None = None):
        self.logger = LogManager.get_logger("isomutator.cache")
        self.db_path = db_path or str(settings.cache_db)
        # timeout=5.0 allows processes to wait briefly if the DB is actively writing
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=5.0)
        self._initialize_db()

    def _initialize_db(self):
        """Creates the cache table if it doesn't already exist."""
        try:
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS evaluations (
                    hash_key TEXT PRIMARY KEY,
                    is_flagged BOOLEAN,
                    confidence REAL
                )
            ''')
            self.conn.commit()
        except Exception as e:
            self.logger.error(f"Failed to initialize SQLite cache: {e}")

    def _hash_response(self, text: str) -> str:
        """Generates a cryptographic SHA-256 hash of the text."""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()

    def store_evaluation(self, response: str, is_flagged: bool, confidence: float):
        """Saves the NLP evaluation results to the database."""
        if not response:
            return
            
        hash_key = self._hash_response(response)
        try:
            self.conn.execute('''
                INSERT OR REPLACE INTO evaluations (hash_key, is_flagged, confidence)
                VALUES (?, ?, ?)
            ''', (hash_key, is_flagged, confidence))
            self.conn.commit()
            self.logger.trace(f"Stored evaluation in cache for hash {hash_key[:8]}.")
        except sqlite3.OperationalError as e:
            self.logger.debug(f"SQLite lock preventing cache write. Failing gracefully: {e}")
        except Exception as e:
            self.logger.error(f"Failed to store evaluation in cache: {e}")

    def get_evaluation(self, response: str) -> dict | None:
        """Retrieves a cached NLP evaluation if it exists."""
        if not response:
            return None
            
        hash_key = self._hash_response(response)
        try:
            cursor = self.conn.execute('''
                SELECT is_flagged, confidence FROM evaluations WHERE hash_key = ?
            ''', (hash_key,))
            row = cursor.fetchone()
            
            if row:
                self.logger.trace(f"Cache HIT for hash {hash_key[:8]}.")
                return {"is_flagged": bool(row[0]), "confidence": float(row[1])}
            
            self.logger.trace(f"Cache MISS for hash {hash_key[:8]}.")
            return None
        except sqlite3.OperationalError as e:
            self.logger.debug(f"SQLite lock preventing cache read. Failing gracefully: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Failed to retrieve evaluation from cache: {e}")
            return None

    def close(self):
        """Closes the database connection safely."""
        try:
            self.conn.close()
        except Exception:
            pass