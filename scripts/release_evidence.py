"""Bind release artifacts, installed payloads and executed deterministic proof."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path, PurePosixPath
import re
import sys
import stat
import tarfile
import xml.etree.ElementTree as ET
import zipfile

SCENARIOS = (
    "lifecycle",
    "recovery-retention",
    "ownership",
    "configured-hosts",
    "storage-upgrade",
)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def junit(path, *, zero_skips=False):
    cases = list(ET.parse(path).getroot().iter("testcase"))
    result = {
        "tests": len(cases),
        **{
            key: sum(c.find(tag) is not None for c in cases)
            for key, tag in (
                ("skipped", "skipped"),
                ("failures", "failure"),
                ("errors", "error"),
            )
        },
    }
    result["xfailed"] = sum(
        c.find("skipped") is not None
        and c.find("skipped").get("type") == "pytest.xfail"
        for c in cases
    )
    result["skipped"] -= result["xfailed"]
    if (
        not cases
        or result["failures"]
        or result["errors"]
        or (zero_skips and (result["skipped"] or result["xfailed"]))
    ):
        raise ValueError(f"incomplete or failing proof: {path.name}: {result}")
    result["junit_sha256"] = sha256(path)
    return result


def installed_payloads(wheel):
    match = re.fullmatch(r"recoleva-([A-Za-z0-9_.!+]+)-py3-none-any\.whl", wheel.name)
    if not match:
        raise ValueError("unreviewed wheel filename")
    metadata = f"recoleva-{match.group(1)}.dist-info"
    with zipfile.ZipFile(wheel) as archive:
        if any(stat.S_ISLNK(i.external_attr >> 16) for i in archive.infolist()):
            raise ValueError("symlink wheel payload")
        names = [i.filename for i in archive.infolist() if not i.is_dir()]
        if len(names) != len(set(names)):
            raise ValueError("duplicate wheel files")
        for name in names:
            parsed = PurePosixPath(name)
            if (
                str(parsed) != name
                or parsed.is_absolute()
                or ".." in parsed.parts
                or any(p.endswith(".data") for p in parsed.parts)
                or any(p.endswith(".dist-info") for p in parsed.parts[1:])
                or "\\" in name
            ):
                raise ValueError("unsafe wheel path")
            if not name.startswith("kai_memory/") and (
                len(parsed.parts) < 2
                or parsed.parts[0] != metadata
                or any(p.endswith(".dist-info") for p in parsed.parts[1:])
            ):
                raise ValueError("unreviewed installation payload")
        expected = {
            n: hashlib.sha256(archive.read(n)).hexdigest()
            for n in names
            if n.startswith("kai_memory/")
        }
    if not expected or "kai_memory/__init__.py" not in expected:
        raise ValueError("missing SDK payload")
    forbidden = (
        "eval/",
        "benchmark_dataset.py",
        "benchmark_io.py",
        "dedup_metric.py",
        "elephant_scorer.py",
        "gold_eval.py",
        "gold_judge.py",
    )
    if any(n.removeprefix("kai_memory/").startswith(forbidden) for n in expected):
        raise ValueError("evaluation in SDK wheel")
    import kai_memory

    package = Path(kai_memory.__file__).resolve().parent
    if "site-packages" not in package.parts or not package.is_relative_to(
        Path(sys.prefix).resolve()
    ):
        raise ValueError("SDK must load from this interpreter's site-packages")
    actual = {
        "kai_memory/" + p.relative_to(package).as_posix(): sha256(p)
        for p in package.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts
    }
    if actual != expected:
        raise ValueError("installed payload differs from exact wheel")
    return expected


def verify_archive_origin(wheel):
    raw = importlib.metadata.distribution("recoleva").read_text("direct_url.json")
    if not raw:
        raise ValueError("installed archive provenance missing")
    origin = json.loads(raw)
    archive = origin.get("archive_info", {})
    actual = archive.get("hashes", {}).get("sha256")
    if actual is None and archive.get("hash", "").startswith("sha256="):
        actual = archive["hash"].removeprefix("sha256=")
    if actual != sha256(wheel):
        raise ValueError("installed archive hash differs from proved wheel")
    return {"archive_sha256": actual, "verified": True}


def build_evidence(dist, reports, source_commit):
    if not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        raise ValueError("source commit must be a full Git SHA")
    (wheel,) = dist.glob("*.whl")
    (sdist,) = dist.glob("*.tar.gz")
    version = importlib.metadata.version("recoleva")
    if (
        wheel.name != f"recoleva-{version}-py3-none-any.whl"
        or sdist.name != f"recoleva-{version}.tar.gz"
    ):
        raise ValueError("artifact names do not match installed version")
    scenarios = {}
    for name in SCENARIOS:
        path = reports / f"{name}.json"
        data = json.loads(path.read_text())
        if not isinstance(data, dict) or not data or data.get("passed") is False:
            raise ValueError(f"invalid scenario receipt: {name}")
        if (
            data.get("blocked_loaded")
            or data.get("blocked_attempts")
            or data.get("import_failures")
        ):
            raise ValueError(f"private import attempted: {name}")
        scenarios[path.name] = sha256(path)
    dependencies = sorted(
        (
            {
                "name": d.metadata["Name"],
                "version": d.version,
                "license": d.metadata.get("License-Expression")
                or d.metadata.get("License"),
                "license_classifiers": [
                    c
                    for c in d.metadata.get_all("Classifier", [])
                    if c.startswith("License ::")
                ],
            }
            for d in importlib.metadata.distributions()
        ),
        key=lambda d: d["name"].lower(),
    )
    return {
        "format": "recoleva-release/v1",
        "distribution": "recoleva",
        "version": version,
        "source_commit": source_commit,
        "python_version": platform.python_version(),
        "ci_run_url": (
            f"https://github.com/{os.environ['GITHUB_REPOSITORY']}/actions/runs/{os.environ['GITHUB_RUN_ID']}"
            if os.environ.get("GITHUB_REPOSITORY") and os.environ.get("GITHUB_RUN_ID")
            else None
        ),
        "artifacts": {
            kind: {"filename": p.name, "sha256": sha256(p)}
            for kind, p in (("wheel", wheel), ("sdist", sdist))
        },
        "installed_payloads": installed_payloads(wheel),
        "installed_archive": verify_archive_origin(wheel),
        "tests": {
            "portable": junit(reports / "unit.xml"),
            "postgres": junit(reports / "postgres.xml", zero_skips=True),
        },
        "scenarios": scenarios,
        "dependencies": dependencies,
        "provider_calls": 0,
        "claims": "deterministic contract proof; model quality and learning efficacy unmeasured",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist", type=Path, required=True)
    parser.add_argument("--reports", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    evidence = build_evidence(args.dist, args.reports, args.source_commit)
    receipt_archive = args.reports / "proof-receipts.tar.gz"
    with tarfile.open(receipt_archive, "w:gz") as archive:
        for name in [*(f"{s}.json" for s in SCENARIOS), "unit.xml", "postgres.xml"]:
            archive.add(args.reports / name, arcname=name)
    evidence["proof_receipts"] = {
        "filename": receipt_archive.name,
        "sha256": sha256(receipt_archive),
    }
    (args.reports / "release.json").write_text(json.dumps(evidence, indent=2) + "\n")
    artifacts = [args.dist / v["filename"] for v in evidence["artifacts"].values()] + [
        receipt_archive
    ]
    (args.reports / "SHA256SUMS").write_text(
        "".join(f"{sha256(p)}  {p.name}\n" for p in artifacts)
    )


if __name__ == "__main__":
    main()
