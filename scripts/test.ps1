<#
.SYNOPSIS
    Run pytest, optionally with the temp directory on a non-system drive.

.DESCRIPTION
    Several tests exercise the result-persistence layer, which refuses to write
    when the target drive has less than `min_free_disk_mb` (1024 MiB) free. On a
    developer machine whose C: drive is nearly full, those tests fail on the
    environment rather than on the code, because pytest puts its `tmp_path`
    fixtures under the system temp directory.

    Setting YOLO11_TEST_TMP moves the temp directory for this test run only. It
    does not change the production result location and does not lower the disk
    reserve — the check still runs, it just runs against a drive that has room.

    Why TEMP/TMP and not `pytest --basetemp`: tests/conftest.py adds
    `tempfile.gettempdir()` to the security path validator's allowed roots.
    Moving only pytest's basetemp puts fixture files outside every allowed root
    and trades disk-space failures for SecurityError failures. Redirecting the
    temp directory itself keeps pytest, `tempfile`, and the allow-list agreeing
    on one location.

.PARAMETER (none)
    Every argument is forwarded to pytest verbatim. This script deliberately
    declares no parameters of its own: a `param()` block would make PowerShell
    bind `-v` to the common `-Verbose` parameter instead of passing it through.

.EXAMPLE
    .\scripts\test.ps1 -q

.EXAMPLE
    $env:YOLO11_TEST_TMP = "D:\yolo11_test_tmp"
    .\scripts\test.ps1 -q

.EXAMPLE
    .\scripts\test.ps1 tests/test_pipeline_finalize_status.py -v
#>

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $PSScriptRoot

# Remember the caller's values so this stays a per-run override. Nothing is
# written to the user or machine environment, and the registry is untouched.
$previousTmp = $env:TMP
$previousTemp = $env:TEMP

$requestedTemp = $env:YOLO11_TEST_TMP
if ([string]::IsNullOrWhiteSpace($requestedTemp)) {
    Write-Host "pytest temp directory: system default ($env:TEMP)"
}
else {
    if (-not (Test-Path -LiteralPath $requestedTemp -PathType Container)) {
        # -Force here is the directory equivalent of `mkdir -p`; it creates
        # missing parents and is a no-op on an existing directory. Nothing is
        # ever deleted.
        New-Item -ItemType Directory -Path $requestedTemp -Force | Out-Null
    }
    $resolvedTemp = (Resolve-Path -LiteralPath $requestedTemp).Path
    $env:TMP = $resolvedTemp
    $env:TEMP = $resolvedTemp
    Write-Host "pytest temp directory: $resolvedTemp (from YOLO11_TEST_TMP)"
}

Push-Location $repoRoot
try {
    # Back to 'Continue' before handing over to pytest. Under 'Stop', anything
    # the test run writes to stderr (pytest warnings, Qt teardown messages)
    # becomes a terminating NativeCommandError as soon as a caller redirects
    # the stream, which aborts the run and hides pytest's real exit code.
    # pytest reports success or failure through its exit code, so that is what
    # this script propagates.
    $ErrorActionPreference = 'Continue'
    & python -m pytest @args
    $exitCode = $LASTEXITCODE
}
finally {
    Pop-Location
    $env:TMP = $previousTmp
    $env:TEMP = $previousTemp
}

exit $exitCode
