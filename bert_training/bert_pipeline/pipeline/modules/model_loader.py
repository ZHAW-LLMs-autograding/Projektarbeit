from __future__ import annotations

import fnmatch
import logging
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, List, Protocol, runtime_checkable

from pydantic import BaseModel
from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.errors import HfHubHTTPError

from pipeline.config import GlobalConfig, build_effective_config, load_global_config

logger = logging.getLogger("pipeline.model_loader")
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
logger.addHandler(_handler)
logger.setLevel(logging.INFO)

# ============================================================
# Errors
# ============================================================
class ModelLoaderError(Exception): ...
class ModelNotFound(ModelLoaderError): ...

# ============================================================
# Config
# ============================================================
class ModelLoaderCfg(BaseModel):
    enabled: bool = True
    model_id: str
    revision: Optional[str] = None
    allow_patterns: Optional[List[str]] = None
    ignore_patterns: Optional[List[str]] = None
    skip_if_present: bool = False
    required_files: List[str] = []
    force_download: bool = False
    smoke_test: bool = False
    class Config:
        extra = "forbid"

def _effective_model_loader_cfg(global_cfg: GlobalConfig) -> ModelLoaderCfg:
    merged: Dict[str, Any] = build_effective_config(global_cfg, "model_loader")
    model_id = merged.get("model_id")
    if not model_id or str(model_id).strip() in {"${DEFAULT}", "", "default"}:
        raise ValueError("model_loader.model_id is not set. Define it in global_config.json under modules.model_loader.model_id")
    return ModelLoaderCfg(**merged)

# ============================================================
# DTOs
# ============================================================
@dataclass(frozen=True)
class ModelArtifact:
    local_dir: Path
    provider: str
    model_ref: str
    revision: Optional[str]
    files: Optional[List[str]] = None

@dataclass(frozen=True)
class ModelPlan:
    provider: str
    model_ref: str
    revision: Optional[str]
    target_dir: Path

# ============================================================
# Provider
# ============================================================
class _HFProvider:
    def __init__(self):
        self.api = HfApi()
    def name(self) -> str:
        return "huggingface"
    @staticmethod
    def _strip(model_ref: str) -> str:
        return model_ref[5:] if model_ref.startswith("hf://") else model_ref
    @staticmethod
    def sanitize_id(model_ref: str) -> str:
        safe = model_ref.replace("/", "__").replace(":", "_")
        return re.sub(r"[^a-zA-Z0-9_.\\-]+", "_", safe)
    def ensure_exists(self, model_ref: str, *, revision: Optional[str], token: Optional[str]) -> None:
        repo_id = self._strip(model_ref)
        try:
            self.api.model_info(repo_id, revision=revision, token=token)
        except HfHubHTTPError as e:
            if getattr(e, "response", None) and e.response.status_code == 404:
                raise ModelNotFound(f"[HF] '{repo_id}' (rev={revision}) not found.") from e
            raise ModelLoaderError(f"[HF] API error: {e}") from e
        except Exception as e:
            raise ModelLoaderError(f"[HF] Unexpected check error: {e}") from e
    def download(self, model_ref: str, *, revision: Optional[str], token: Optional[str],
                 target_dir: Path, allow_patterns: Optional[List[str]],
                 ignore_patterns: Optional[List[str]]) -> Path:
        repo_id = self._strip(model_ref)
        try:
            local = snapshot_download(
                repo_id=repo_id,
                revision=revision,
                token=token,
                local_dir=str(target_dir),
                local_dir_use_symlinks=False,
                allow_patterns=allow_patterns,
                ignore_patterns=ignore_patterns,
            )
            return Path(local)
        except HfHubHTTPError as e:
            if getattr(e, "response", None) and e.response.status_code == 401:
                raise ModelLoaderError("[HF] Unauthorized: missing/invalid HF token.") from e
            raise ModelLoaderError(f"[HF] Download failed: {e}") from e
        except Exception as e:
            raise ModelLoaderError(f"[HF] Unexpected download error: {e}") from e

# ============================================================
# Service API
# ============================================================
@runtime_checkable
class ModelLoaderServiceAPI(Protocol):
    def name(self) -> str: ...
    def plan(self) -> ModelPlan: ...
    def check_remote(self) -> None: ...
    def download(self) -> ModelArtifact: ...
    def local_dir(self) -> Optional[Path]: ...
    def metadata(self) -> Dict[str, Any]: ...

# ============================================================
# Service
# ============================================================
class ModelLoaderService(ModelLoaderServiceAPI):
    """
    Zielpfad: <artifacts_root>/<models_subdir>/<provider>__<model_id>
    """
    def __init__(self, global_cfg: GlobalConfig, run_dirs: Dict[str, Path]):
        self.g = global_cfg
        self.c: ModelLoaderCfg = _effective_model_loader_cfg(self.g)
        self._provider = _HFProvider()
        self._token = os.getenv(self.g.secrets.hf_token_env or "HF_TOKEN")

        artifacts_root = Path(self.g.paths.artifacts_root)         # bereits absolut (load_global_config)
        models_root = artifacts_root / self.g.paths.models_subdir  # models_subdir bleibt RELATIV
        safe = self._provider.sanitize_id(self.c.model_id)
        self._target = (models_root / f"{self._provider.name()}__{safe}").resolve()
        self._target.mkdir(parents=True, exist_ok=True)
        self._local_dir: Optional[Path] = None

        logger.info(f"[DEBUG] artifacts_root={artifacts_root}")
        logger.info(f"[DEBUG] models_root={models_root.resolve()}")
        logger.info(f"[DEBUG] target_dir={self._target}")

    def name(self) -> str:
        return "model_loader"

    def plan(self) -> ModelPlan:
        return ModelPlan(self._provider.name(), self.c.model_id, self.c.revision, self._target)

    def _derive_patterns(self) -> Dict[str, Optional[List[str]]]:
        """
        Forme allow/ignore je nach smoke_test.
        """
        allow = list(self.c.allow_patterns or [])
        ignore = list(self.c.ignore_patterns or [])

        if self.c.smoke_test:
            # Harte Auswahl nur Meta/Tokenizer/Configs – KEINE Weights
            allow = [
                "*.json",
                "config.*",
                "generation_config.*",
                "special_tokens_map.*",
                "tokenizer.*",
                "*.model",
                "merges.txt",
                "vocab.*"
            ]
            # Sicherstellen, dass safetensors ausgeschlossen sind
            ignore = list(set(ignore + ["*.safetensors", "*.bin", "*.pt"]))

        return {"allow": allow or None, "ignore": ignore or None}

    def check_remote(self) -> None:
        self._provider.ensure_exists(self.c.model_id, revision=self.c.revision, token=self._token)

    def _all_required_present(self) -> bool:
        if not self.c.required_files or not self._target.exists():
            return False
        existing = [str(p.relative_to(self._target)) for p in self._target.rglob("*") if p.is_file()]
        for pattern in self.c.required_files:
            if not any(fnmatch.fnmatchcase(path, pattern) for path in existing):
                return False
        return True

    def download(self) -> ModelArtifact:
        self.check_remote()

        patterns = self._derive_patterns()
        allow_patterns = patterns["allow"]
        ignore_patterns = patterns["ignore"]

        # Bei smoke_test ignorieren wir required_files, um den Skip nicht zu erzwingen
        can_skip = (not self.c.smoke_test) and self.c.skip_if_present and self._all_required_present() and not self.c.force_download
        if can_skip:
            logger.info(f"⏭️  Skip download: required files already present in {self._target}")
            self._local_dir = self._target
        else:
            logger.info(f"[{self._provider.name()}] Download '{self.c.model_id}' (rev={self.c.revision}) -> {self._target}")
            logger.info(f"[DEBUG] allow_patterns={allow_patterns} ignore_patterns={ignore_patterns}")
            local = self._provider.download(
                self.c.model_id,
                revision=self.c.revision,
                token=self._token,
                target_dir=self._target,
                allow_patterns=allow_patterns,
                ignore_patterns=ignore_patterns,
            )
            self._local_dir = Path(local)

        files = [str(p.relative_to(self._local_dir)) for p in self._local_dir.rglob("*") if p.is_file()]
        logger.info(f"✅ Model available at: {self._local_dir}")
        return ModelArtifact(self._local_dir, self._provider.name(), self.c.model_id, self.c.revision, files)

    def local_dir(self) -> Optional[Path]:
        return self._local_dir

    def metadata(self) -> Dict[str, Any]:
        return {
            "provider": self._provider.name(),
            "model_ref": self.c.model_id,
            "revision": self.c.revision,
            "target_dir": str(self._target),
            "token_env": self.g.secrets.hf_token_env,
            "smoke_test": self.c.smoke_test,
        }

# ============================================================
# Factory + CLI
# ============================================================
def build_model_loader(global_cfg: GlobalConfig, run_dirs: Dict[str, Path]) -> ModelLoaderServiceAPI:
    return ModelLoaderService(global_cfg, run_dirs)

def _make_run_id(mode: str) -> str:
    import datetime, subprocess
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    if mode == "timestamp": return ts
    if mode == "git":
        proc = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
        sha = proc.stdout.strip() if proc.returncode == 0 else "nogit"
        return sha
    proc = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
    sha = proc.stdout.strip() if proc.returncode == 0 else "nogit"
    return f"{ts}-{sha}"

def main():
    from argparse import ArgumentParser
    p = ArgumentParser()
    p.add_argument("--global-config", required=True)
    p.add_argument("--run-id", default=None)
    args = p.parse_args()

    g = load_global_config(Path(args.global_config))
    run_id = args.run_id or _make_run_id(g.run_id_mode)
    _ = g.resolve_run_dirs(run_id)

    svc = build_model_loader(g, run_dirs={})
    logger.info(f"Plan: {svc.plan()}")
    try:
        artifact = svc.download()
        print(f"✅ Model available at: {artifact.local_dir}")
    except ModelLoaderError as e:
        logger.error(e)
        sys.exit(2)

if __name__ == "__main__":
    main()
