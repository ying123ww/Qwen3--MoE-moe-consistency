#!/usr/bin/env python3
"""Check indexed Git paths and blob sizes without importing ML dependencies."""
import fnmatch
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys


def git(root, *args, **kwargs):
    return subprocess.check_output(["git", *args], cwd=root, **kwargs)


def violations(path, policy):
    parts = PurePosixPath(path).parts
    name = parts[-1]
    if any(part in policy["forbidden_directories"] or part.endswith(".egg-info")
           for part in parts[:-1]):
        return "local tooling, cache, build or runtime directory"
    if path in policy["forbidden_files"]:
        return "machine-specific or generated state"
    if any(path.startswith(prefix) for prefix in policy["forbidden_prefixes"]):
        return "runtime artifacts or downloaded resources"
    model = policy["model_config_directory"]
    if path.startswith(model) and path[len(model):] not in policy["allowed_model_files"]:
        return "downloaded model asset; keep only configuration and provenance"
    if name not in policy["allowed_names"]:
        if any(fnmatch.fnmatchcase(name, pattern)
               for pattern in policy["forbidden_name_globs"]):
            return "generated binary, log, archive or local credential file"
    return None


def main():
    root = Path(git(None, "rev-parse", "--show-toplevel").decode().strip())
    policy = json.loads((root / ".repo-policy.json").read_text())
    entries = []
    errors = []
    raw = git(root, "ls-files", "--stage", "-z")
    for row in raw.split(b"\0"):
        if not row:
            continue
        metadata, filename = row.split(b"\t", 1)
        mode, sha, stage = metadata.decode().split()
        path = filename.decode("utf-8", errors="surrogateescape")
        entries.append((path, sha))
        if stage != "0":
            errors.append((path, "unresolved merge conflict"))
        if mode == "160000":
            errors.append((path, "gitlink; dependencies must contain actual source files"))
        reason = violations(path, policy)
        if reason:
            errors.append((path, reason))
    hashes = sorted({sha for _, sha in entries})
    sizes = {}
    if hashes:
        result = git(root, "cat-file", "--batch-check",
                     input=("".join(sha + "\n" for sha in hashes)).encode())
        for row in result.decode().splitlines():
            fields = row.split()
            if len(fields) != 3 or fields[1] != "blob":
                print("Cannot inspect Git object: " + row, file=sys.stderr)
                return 1
            sizes[fields[0]] = int(fields[2])
    limit = policy["max_file_bytes"]
    for path, sha in entries:
        if sizes[sha] > limit:
            errors.append((path, "blob exceeds %.0f MiB limit" % (limit / 1048576)))
    if errors:
        print("Repository hygiene failed: %d violations" % len(errors), file=sys.stderr)
        for path, reason in errors[:30]:
            print("  %s: %s" % (path, reason), file=sys.stderr)
        if len(errors) > 30:
            print("  ... %d more" % (len(errors) - 30), file=sys.stderr)
        return 1
    print("Repository hygiene passed: %d files, no forbidden paths or oversized blobs."
          % len(entries))
    return 0


if __name__ == "__main__":
    sys.exit(main())
