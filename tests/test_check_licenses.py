import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location("check_licenses", Path(__file__).parent.parent / "scripts" / "check_licenses.py")
check_licenses = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check_licenses)


def test_mit_passes():
    assert check_licenses.verdict("MIT", None, [])


def test_apache_with_exception_passes():
    assert check_licenses.verdict("Apache-2.0 WITH LLVM-exception", None, [])


def test_gpl_2_fails():
    assert not check_licenses.verdict("GPL-2.0-only", None, [])


def test_proprietary_classifier_fails():
    assert not check_licenses.verdict(None, None, ["License :: Other/Proprietary License"])


def test_or_inside_a_group_cannot_hide_a_bad_licence():
    assert not check_licenses.verdict("(MIT OR Apache-2.0) AND LicenseRef-Proprietary", None, [])
    assert check_licenses.verdict("(MIT OR Apache-2.0) AND BSD-3-Clause", None, [])
    assert check_licenses.verdict("MIT OR LicenseRef-Proprietary", None, [])
