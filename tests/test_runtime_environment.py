from types import SimpleNamespace

from tools import check_runtime_environment as runtime


def test_runtime_checker_reports_import_and_version_failures(monkeypatch):
    monkeypatch.setattr(runtime.sys, "version_info", (3, 11, 0))
    monkeypatch.setattr(
        runtime.importlib,
        "import_module",
        lambda name: (
            (_ for _ in ()).throw(OSError("DLL unavailable"))
            if name == "torch"
            else SimpleNamespace(__file__=f"/fake/{name}.py")
        ),
    )
    monkeypatch.setattr(
        runtime.importlib.metadata,
        "version",
        lambda name: (
            "0.0.0"
            if name == "torch"
            else runtime.EXPECTED_DISTRIBUTIONS[name]
        ),
    )

    checks = runtime.run_checks()

    by_name = {check.name: check for check in checks}
    assert by_name["python_version"].passed is True
    assert by_name["import:torch"].passed is False
    assert "DLL unavailable" in by_name["import:torch"].detail
    assert by_name["version:torch"].passed is False


def test_runtime_checker_rejects_unsupported_python(monkeypatch):
    monkeypatch.setattr(runtime.sys, "version_info", (3, 12, 0))
    monkeypatch.setattr(
        runtime.importlib,
        "import_module",
        lambda name: SimpleNamespace(__file__=f"/fake/{name}.py"),
    )
    monkeypatch.setattr(
        runtime.importlib.metadata,
        "version",
        lambda name: runtime.EXPECTED_DISTRIBUTIONS[name],
    )

    checks = runtime.run_checks()

    assert checks[0].name == "python_version"
    assert checks[0].passed is False


def test_runtime_checker_accepts_deployed_python_310(monkeypatch):
    monkeypatch.setattr(runtime.sys, "version_info", (3, 10, 18))
    monkeypatch.setattr(
        runtime.importlib,
        "import_module",
        lambda name: SimpleNamespace(__file__=f"/fake/{name}.py"),
    )
    monkeypatch.setattr(
        runtime.importlib.metadata,
        "version",
        lambda name: runtime.EXPECTED_DISTRIBUTIONS[name],
    )

    checks = runtime.run_checks()

    assert checks[0].passed is True
