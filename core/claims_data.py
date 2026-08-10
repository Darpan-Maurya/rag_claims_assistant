from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from core.config import settings


class ClaimsDataStore:
    """Loads structured claims for deterministic analytics independently of RAG."""

    def __init__(
        self,
        metadata_path: Path | None = None,
        processed_data_path: Path | None = None,
    ) -> None:
        self.metadata_path = metadata_path or settings.metadata_path
        self.processed_data_path = processed_data_path or settings.processed_data_path
        self._df: Optional[pd.DataFrame] = None
        self._source_path: Optional[Path] = None
        self._last_error: Optional[str] = None
        self.reload()

    def reload(self) -> bool:
        self._df = None
        self._source_path = None
        self._last_error = None

        errors: list[str] = []
        for path in (self.metadata_path, self.processed_data_path):
            if not path.exists():
                errors.append(f"missing:{path}")
                continue
            try:
                df = pd.read_parquet(path)
                if df.empty:
                    errors.append(f"empty:{path}")
                    continue
                self._df = df.reset_index(drop=True)
                self._source_path = path
                return True
            except Exception as exc:
                errors.append(f"unreadable:{path}:{type(exc).__name__}")

        self._last_error = "; ".join(errors) or "No claims data source is configured."
        return False

    def dataframe(self) -> pd.DataFrame:
        if self._df is None:
            raise RuntimeError(self._last_error or "Claims data is not ready.")
        return self._df

    def readiness(self) -> Dict[str, Any]:
        return {
            "ready": self._df is not None,
            "rows": int(len(self._df)) if self._df is not None else 0,
            "source": str(self._source_path) if self._source_path else None,
            "reason": self._last_error,
        }
