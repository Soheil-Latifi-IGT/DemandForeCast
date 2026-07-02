from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any


def project_root_from_config(config_path: str | Path) -> Path:
    """Find the repository root that owns a pipeline config."""
    path = Path(config_path).expanduser().resolve()
    for parent in [path.parent, *path.parents]:
        has_config_dir = (parent / "config").is_dir() or (parent / "configs").is_dir()
        if has_config_dir and (parent / "src").is_dir():
            return parent
    return path.parent


def resolve_project_path(value: str | Path, project_root: str | Path) -> Path:
    """Resolve a configured path against the repository root."""
    path = Path(value).expanduser()
    root = Path(project_root).expanduser().resolve()
    if not path.is_absolute():
        return (root / path).resolve()
    if path.exists():
        return path

    parts = list(path.parts)
    root_indexes = [
        index
        for index, part in enumerate(parts)
        if part.casefold() == root.name.casefold()
    ]
    if root_indexes:
        return root.joinpath(*parts[root_indexes[-1] + 1 :])
    return path


def resolve_config_path_fields(
    config: dict[str, Any],
    config_path: str | Path,
    keys: Iterable[str],
) -> dict[str, Any]:
    """Return a shallow config copy with selected path fields resolved."""
    resolved = dict(config)
    project_root = project_root_from_config(config_path)
    for key in keys:
        value = resolved.get(key)
        if value is not None:
            resolved[key] = str(resolve_project_path(value, project_root))
    return resolved
