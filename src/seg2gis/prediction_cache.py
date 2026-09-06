import hashlib
import importlib.metadata
import json
import os
import re
import tempfile
import zipfile
from pathlib import Path

import numpy as np


CACHE_SCHEMA_VERSION = 1
_FINGERPRINT_CHUNK_SIZE = 1024 * 1024
_INFERENCE_SOURCE_FILES = ("gis_utils.py", "models.py", "transforms.py")
_INFERENCE_DISTRIBUTIONS = (
    "albumentations",
    "numpy",
    "opencv-python",
    "segmentation-models-pytorch",
    "torch",
    "torchvision",
)


def safe_path_part(value):
    value = str(value).strip()
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return value.strip("._") or "unnamed"


def resolve_prediction_cache_dir(
    repo_root,
    base_dir,
    run_name,
    split,
    tile_size,
    stride,
):
    if base_dir is None:
        return None

    cache_name = (
        f"{safe_path_part(run_name)}_{safe_path_part(split)}"
        f"_tile{int(tile_size)}_stride{int(stride)}"
    )
    return Path(repo_root) / base_dir / cache_name


def cache_file_for_image(cache_dir, image_path):
    return Path(cache_dir) / f"{safe_path_part(Path(image_path).stem)}.npz"


def fingerprint_file(path):
    resolved_path = Path(path).resolve(strict=True)

    if not resolved_path.is_file():
        raise FileNotFoundError(f"Cannot fingerprint non-file path: {resolved_path}")

    digest = hashlib.sha256()
    with resolved_path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(_FINGERPRINT_CHUNK_SIZE), b""):
            digest.update(chunk)

    return {
        "path": str(resolved_path),
        "size_bytes": resolved_path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def fingerprint_inference_sources():
    source_dir = Path(__file__).resolve().parent
    return {
        file_name: fingerprint_file(source_dir / file_name)["sha256"]
        for file_name in _INFERENCE_SOURCE_FILES
    }


def inference_runtime_versions():
    versions = {}

    for distribution in _INFERENCE_DISTRIBUTIONS:
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None

    return versions


class PredictionCache:
    def __init__(
        self,
        checkpoint_path,
        architecture,
        encoder,
        tile_size,
        stride,
        device,
        refresh=False,
    ):
        self.checkpoint_fingerprint = fingerprint_file(checkpoint_path)
        self.architecture = str(architecture)
        self.encoder = str(encoder)
        self.tile_size = int(tile_size)
        self.stride = int(stride)
        self.device = str(device)
        self.refresh = bool(refresh)
        self.inference_sources = fingerprint_inference_sources()
        self.runtime_versions = inference_runtime_versions()

    def metadata_for(self, image_path, mask_path):
        return {
            "schema_version": CACHE_SCHEMA_VERSION,
            "checkpoint": self.checkpoint_fingerprint,
            "model": {
                "architecture": self.architecture,
                "encoder": self.encoder,
            },
            "image": fingerprint_file(image_path),
            "mask": fingerprint_file(mask_path),
            "tile_size": self.tile_size,
            "stride": self.stride,
            "device": self.device,
            "inference_sources": self.inference_sources,
            "runtime_versions": self.runtime_versions,
        }

    def load(self, cache_path, expected_metadata):
        cache_path = Path(cache_path)

        if self.refresh or not cache_path.is_file():
            return None

        try:
            with np.load(cache_path, allow_pickle=False) as data:
                stored_metadata = json.loads(str(data["cache_metadata"].item()))
                if stored_metadata != expected_metadata:
                    return None

                prob_map = data["prob_map"].astype(np.float32, copy=False)
                target_mask = data["target_mask"].astype(bool, copy=False)
        except (
            EOFError,
            OSError,
            KeyError,
            TypeError,
            ValueError,
            zipfile.BadZipFile,
        ):
            return None

        if prob_map.shape != target_mask.shape:
            return None

        return prob_map, target_mask

    def save(self, cache_path, metadata, prob_map, target_mask):
        cache_path = Path(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = None

        try:
            with tempfile.NamedTemporaryFile(
                dir=cache_path.parent,
                prefix=f".{cache_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
                np.savez_compressed(
                    temporary_file,
                    cache_metadata=json.dumps(metadata, sort_keys=True),
                    prob_map=prob_map.astype(np.float32, copy=False),
                    target_mask=target_mask.astype(np.uint8, copy=False),
                )

            os.replace(temporary_path, cache_path)
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()
