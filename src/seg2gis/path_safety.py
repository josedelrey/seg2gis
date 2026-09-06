from pathlib import Path


def resolve_safe_tile_output_dir(out_dir, repo_root, protected_paths=()):
    """Resolve and validate a tile output directory before destructive reset."""
    if out_dir is None or not str(out_dir).strip():
        raise ValueError("Tile output directory cannot be empty.")

    repo_root = Path(repo_root).resolve()
    data_root = (repo_root / "data").resolve()
    output_path = Path(out_dir).resolve()

    if output_path == data_root or data_root not in output_path.parents:
        raise ValueError(
            "Unsafe tile output directory. It must be a dedicated subdirectory "
            f"inside the repository data directory: {data_root}"
        )

    for protected_path in protected_paths:
        if protected_path is None or not str(protected_path).strip():
            continue

        protected_path = Path(protected_path).resolve()
        paths_overlap = (
            output_path == protected_path
            or output_path in protected_path.parents
            or protected_path in output_path.parents
        )

        if paths_overlap:
            raise ValueError(
                "Unsafe tile output directory. It overlaps an input directory: "
                f"output={output_path}, input={protected_path}"
            )

    return output_path
