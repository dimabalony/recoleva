"""Release receipts cannot turn missing, skipped or wrong installed proof green."""

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import zipfile

import pytest

spec = importlib.util.spec_from_file_location(
    "release_evidence", Path(__file__).parents[1] / "scripts/release_evidence.py"
)
evidence = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evidence)


@pytest.mark.parametrize(
    "case",
    [
        "",
        "<testcase><skipped /></testcase>",
        "<testcase><failure /></testcase>",
        "<testcase><error /></testcase>",
    ],
)
def test_postgres_receipt_refuses_empty_skipped_or_failing_cases(tmp_path, case):
    path = tmp_path / "postgres.xml"
    path.write_text(f"<testsuite>{case}</testsuite>")
    with pytest.raises(ValueError, match="incomplete or failing"):
        evidence.junit(path, zero_skips=True)


def test_receipt_counts_executed_cases_instead_of_summary_attributes(tmp_path):
    path = tmp_path / "postgres.xml"
    path.write_text('<testsuite tests="999"><testcase /></testsuite>')
    assert evidence.junit(path, zero_skips=True)["tests"] == 1


@pytest.mark.parametrize("mutation", ["changed", "extra", "missing"])
def test_installed_payload_inventory_refuses_byte_or_ownership_drift(
    tmp_path, monkeypatch, mutation
):
    package = tmp_path / "site-packages/kai_memory"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("API=2\n")
    monkeypatch.setitem(
        sys.modules,
        "kai_memory",
        SimpleNamespace(__file__=str(package / "__init__.py")),
    )
    monkeypatch.setattr(sys, "prefix", str(tmp_path))
    wheel = tmp_path / "recoleva-0.3.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("kai_memory/__init__.py", "API=2\n")
    assert evidence.installed_payloads(wheel)
    if mutation == "changed":
        (package / "__init__.py").write_text("API=9\n")
    elif mutation == "extra":
        (package / "unexpected.py").write_text("")
    else:
        (package / "__init__.py").unlink()
    with pytest.raises(ValueError, match="differs"):
        evidence.installed_payloads(wheel)


@pytest.mark.parametrize(
    "name",
    [
        "./kai_memory/__init__.py",
        "kai_memory/../evil.py",
        "kai_memory/eval/__init__.py",
    ],
)
def test_wheel_paths_and_evaluation_are_rejected(tmp_path, name):
    wheel = tmp_path / "recoleva-0.3.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("kai_memory/__init__.py", "")
        archive.writestr(name, "")
    with pytest.raises(ValueError, match="unsafe|evaluation"):
        evidence.installed_payloads(wheel)


def test_expected_failure_is_distinct_and_cannot_qualify_postgres(tmp_path):
    path = tmp_path / "unit.xml"
    path.write_text(
        '<testsuite><testcase /><testcase><skipped type="pytest.xfail" /></testcase></testsuite>'
    )
    result = evidence.junit(path)
    assert (result["tests"], result["skipped"], result["xfailed"]) == (2, 0, 1)
    with pytest.raises(ValueError):
        evidence.junit(path, zero_skips=True)


@pytest.mark.parametrize(
    "name",
    [
        "other.dist-info/entry_points.txt",
        "kai_memory/other.dist-info/entry_points.txt",
        "recoleva-0.3.1.dist-info/entry_points.txt",
        "nested/recoleva-0.3.0.dist-info/entry_points.txt",
        "recoleva-0.3.0.dist-info/other.dist-info/entry_points.txt",
        "recoleva.data/purelib/plugin.py",
    ],
)
def test_foreign_or_indirect_wheel_payload_is_rejected(tmp_path, name):
    wheel = tmp_path / "recoleva-0.3.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("kai_memory/__init__.py", "")
        archive.writestr(name, "injected")
    with pytest.raises(ValueError, match="unreviewed installation payload|unsafe"):
        evidence.installed_payloads(wheel)


@pytest.mark.parametrize("kind", ["missing", "wrong", "matching"])
def test_archive_origin_must_match_exact_wheel_hash(tmp_path, monkeypatch, kind):
    import json

    wheel = tmp_path / "recoleva-0.3.0-py3-none-any.whl"
    wheel.write_bytes(b"proved wheel")
    raw = (
        None
        if kind == "missing"
        else json.dumps(
            {
                "archive_info": {
                    "hashes": {
                        "sha256": evidence.sha256(wheel)
                        if kind == "matching"
                        else "0" * 64
                    }
                }
            }
        )
    )
    monkeypatch.setattr(
        evidence.importlib.metadata,
        "distribution",
        lambda name: SimpleNamespace(read_text=lambda path: raw),
    )
    if kind == "matching":
        assert evidence.verify_archive_origin(wheel)["verified"]
    else:
        with pytest.raises(ValueError, match="archive"):
            evidence.verify_archive_origin(wheel)


@pytest.mark.parametrize("kind", ["duplicate", "symlink"])
def test_wheel_cannot_hide_duplicate_or_link_payload(tmp_path, kind):
    import stat

    wheel = tmp_path / "recoleva-0.3.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("kai_memory/__init__.py", "")
        if kind == "duplicate":
            with pytest.warns(UserWarning, match="Duplicate"):
                archive.writestr("kai_memory/__init__.py", "")
        else:
            info = zipfile.ZipInfo("kai_memory/link.py")
            info.create_system = 3
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
            archive.writestr(info, "__init__.py")
    with pytest.raises(ValueError, match="duplicate|symlink"):
        evidence.installed_payloads(wheel)
