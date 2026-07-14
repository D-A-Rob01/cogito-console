from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOCAL_DATA_DIRECTORIES = ("data", "artifacts", "traces")


@dataclass(frozen=True, slots=True)
class PurgedFile:
    path: str
    bytes: int


def purge_local_data(*, root: Path = PROJECT_ROOT, dry_run: bool = False) -> list[PurgedFile]:
    """Remove only project-local generated data and return an exact manifest."""
    resolved_root = root.resolve()
    removed: list[PurgedFile] = []
    for directory_name in LOCAL_DATA_DIRECTORIES:
        directory = (resolved_root / directory_name).resolve()
        if not directory.is_relative_to(resolved_root):
            raise RuntimeError(f"refusing to purge outside project root: {directory}")
        if not directory.exists():
            continue
        for candidate in sorted(directory.rglob("*"), reverse=True):
            if candidate.is_dir() and not candidate.is_symlink():
                if not dry_run:
                    try:
                        candidate.rmdir()
                    except OSError:
                        pass
                continue
            size = candidate.lstat().st_size
            removed.append(
                PurgedFile(
                    path=candidate.relative_to(resolved_root).as_posix(),
                    bytes=size,
                )
            )
            if not dry_run:
                candidate.unlink()
    return sorted(removed, key=lambda item: item.path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Delete Cogito Console's project-local databases, traces, and artifacts."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be removed without deleting it.",
    )
    args = parser.parse_args()
    manifest = purge_local_data(dry_run=args.dry_run)
    action = "would remove" if args.dry_run else "removed"
    for item in manifest:
        print(f"{action}: {item.path} ({item.bytes} bytes)")
    print(f"{action}: {len(manifest)} files ({sum(item.bytes for item in manifest)} bytes total)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
