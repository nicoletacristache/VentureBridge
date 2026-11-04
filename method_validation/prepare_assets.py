"""Restore and verify archived method-validation inputs and predictions."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import stat
import zipfile

ROOT = Path(__file__).resolve().parent


def restore(archive: Path, manifest: dict[str, dict], prefix: str) -> int:
    restored = 0
    with zipfile.ZipFile(archive) as bundle:
        for entry in bundle.infolist():
            if entry.is_dir():
                continue
            rel = PurePosixPath(entry.filename)
            mode = entry.external_attr >> 16
            if rel.is_absolute() or ".." in rel.parts or not rel.parts or rel.parts[0] != prefix or stat.S_ISLNK(mode):
                raise ValueError("Unsafe archive member")
            key = "method_validation/" + rel.as_posix()
            expected = manifest.get(key)
            if expected is None:
                raise ValueError("Archive member absent from source manifest: " + key)
            payload = bundle.read(entry)
            if len(payload) != expected["bytes"] or hashlib.sha256(payload).hexdigest() != expected["sha256"]:
                raise ValueError("Archive checksum mismatch: " + key)
            dest = ROOT.joinpath(*rel.parts)
            resolved = dest.resolve()
            if not resolved.is_relative_to(ROOT.resolve()):
                raise ValueError("Extraction target escapes the release directory")
            if dest.exists():
                if dest.is_symlink() or dest.read_bytes() != payload:
                    raise FileExistsError("Refusing to overwrite a different local file: " + key)
            else:
                dest.parent.mkdir(parents=True, exist_ok=True)
                with dest.open("xb") as out:
                    out.write(payload)
            restored += 1
    expected_count = sum(1 for key, value in manifest.items() if value.get("stored_in") == str(archive.relative_to(ROOT)))
    if restored != expected_count:
        raise ValueError("Archive member count mismatch")
    return restored


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-predictions", action="store_true")
    args = parser.parse_args()
    source = json.loads((ROOT / "source_manifest.json").read_text())
    manifest = {record["original_path"]: record for record in source["files"]}
    counts = {"data_files": restore(ROOT / "datasets/benchmark_data.zip", manifest, "data")}
    if args.with_predictions:
        counts["prediction_files"] = restore(ROOT / "reference_outputs/frozen_predictions.zip", manifest, "predictions")
    print(json.dumps({"verified": True, **counts}, sort_keys=True))


if __name__ == "__main__":
    main()
