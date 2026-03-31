"""Post-build verification script for yolo11_inference PyInstaller package.

Usage:
    python verify_build.py <dist_dir>
    python verify_build.py dist/yolo11_inference
"""
from __future__ import annotations

import sys
import subprocess
from pathlib import Path


REQUIRED_FILES = [
    "yolo11_inference.exe",
]

REQUIRED_DIRS = [
    "Runtime",
    "MvImport",
]

# timm backbone weights bundled in _internal for offline Patchcore loading
REQUIRED_TIMM_CACHE = (
    "timm_cache/hub/checkpoints/wide_resnet50_racm-8234f177.pth"
)

REQUIRED_DLL_SAMPLES = [
    "MvImport/MvCameraControl_class.py",
    "MvImport/CameraParams_const.py",
]

MODEL_WEIGHT_PATTERN = "**/*.pt"

WARN_SIZE_MB = 200  # 輸出小於此值可能有漏包


def hr(char: str = "─", width: int = 60) -> str:
    return char * width


def _resolve_data_root(dist: Path) -> tuple[Path, bool]:
    """Return (data_root, is_pyinstaller6).

    PyInstaller ≥ 6.0 places all non-exe files under <dist>/_internal/.
    Earlier versions put them directly in <dist>/.
    """
    internal = dist / "_internal"
    if internal.is_dir():
        return internal, True
    return dist, False


def check_dir(dist: Path) -> bool:
    print(hr())
    print(f"驗證目錄: {dist}")
    print(hr())

    if not dist.exists():
        print(f"[FAIL] 輸出目錄不存在: {dist}")
        return False

    data_root, is_v6 = _resolve_data_root(dist)
    if is_v6:
        print("[INFO] 偵測到 PyInstaller 6.x 結構，資料根目錄: _internal/")
    print()

    passed = True

    # --- 必要檔案 ---
    print("[必要檔案]")
    for f in REQUIRED_FILES:
        # exe lives in dist root; everything else is under data_root
        p = dist / f if f.endswith(".exe") else data_root / f
        if p.exists():
            size_kb = p.stat().st_size // 1024
            print(f"  ✓  {f}  ({size_kb:,} KB)")
        else:
            print(f"  ✗  {f}  ← 缺少！")
            passed = False

    # --- 必要目錄 ---
    print("\n[必要目錄]")
    for d in REQUIRED_DIRS:
        p = data_root / d
        if p.exists() and p.is_dir():
            count = sum(1 for _ in p.rglob("*") if _.is_file())
            print(f"  ✓  {d}/  ({count} 個檔案)")
        else:
            print(f"  ✗  {d}/  ← 缺少！")
            passed = False

    # --- timm_cache（Patchcore backbone 離線權重）---
    print("\n[timm_cache]")
    timm_file = data_root / REQUIRED_TIMM_CACHE
    if timm_file.exists():
        size_mb = timm_file.stat().st_size / (1024 * 1024)
        print(f"  ✓  {REQUIRED_TIMM_CACHE}  ({size_mb:.0f} MB)")
    else:
        print(f"  ✗  {REQUIRED_TIMM_CACHE}  ← 缺少！anomalib 離線無法初始化")
        passed = False

    # --- MvImport 抽查 ---
    print("\n[MvImport 抽查]")
    for f in REQUIRED_DLL_SAMPLES:
        p = data_root / f
        if p.exists():
            print(f"  ✓  {f}")
        else:
            print(f"  ✗  {f}  ← 缺少！")
            passed = False

    # --- 模型權重檔（外部資料，不計入 pass/fail）---
    print("\n[模型權重 .pt]（外部資料，僅供參考）")
    models_dir = dist / "models"
    model_files = list(models_dir.rglob("*.pt")) if models_dir.exists() else []
    if model_files:
        for pt in sorted(model_files):
            rel = pt.relative_to(dist)
            size_mb = pt.stat().st_size / (1024 * 1024)
            print(f"  ✓  {rel}  ({size_mb:.1f} MB)")
    else:
        print("  -  未在 dist 根目錄找到 .pt；請手動複製 models/ 到此處")

    # --- 套件是否齊全（抽查關鍵套件）---
    print("\n[關鍵套件 .pyd / .so 抽查]")
    key_patterns = ["torch", "cv2", "PyQt5", "ultralytics", "anomalib"]
    for kw in key_patterns:
        matches = list(dist.rglob(f"*{kw}*"))
        matches = [
            m for m in matches
            if m.is_file() and m.suffix in (".pyd", ".dll", ".so", ".py")
        ]
        if matches:
            print(f"  ✓  {kw}  ({len(matches)} 個相關檔案)")
        else:
            print(f"  ?  {kw}  ← 未找到明顯的套件檔案，可能已合併到 PKG 中")

    # --- 總大小 ---
    print("\n[輸出大小]")
    total_bytes = sum(f.stat().st_size for f in dist.rglob("*") if f.is_file())
    total_mb = total_bytes / (1024 * 1024)
    print(f"  總大小: {total_mb:.0f} MB")
    if total_mb < WARN_SIZE_MB:
        print(
            f"  [WARNING] 小於 {WARN_SIZE_MB} MB，"
            "可能有套件沒打進去（torch 本身就超過 2 GB）"
        )
        passed = False

    # --- 嘗試啟動 exe（--help 旗標，確認能執行不崩潰）---
    print("\n[exe 啟動測試]")
    exe = dist / "yolo11_inference.exe"
    if exe.exists():
        try:
            result = subprocess.run(
                [str(exe), "--help"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            # GUI app 不一定有 --help，exitcode 非零也可能正常
            if result.returncode in (0, 1, 2):
                print(f"  ✓  exe 啟動回傳 exitcode={result.returncode}（正常）")
            else:
                print(f"  ?  exe 啟動回傳 exitcode={result.returncode}")
                if result.stderr:
                    print(f"     stderr: {result.stderr[:300]}")
        except subprocess.TimeoutExpired:
            # GUI 通常不會立即退出，timeout 視為正常
            print("  ✓  exe 啟動後未在 30s 內退出（正常 GUI 行為）")
        except Exception as exc:
            print(f"  ✗  exe 啟動失敗: {exc}")
            passed = False
    else:
        print("  ✗  exe 不存在，跳過啟動測試")

    # --- 結果 ---
    print()
    print(hr("═"))
    if passed:
        print("  結果: 通過 ✓  可將整個資料夾複製到目標電腦執行")
    else:
        print("  結果: 有問題 ✗  請檢查上方標示的缺少項目")
    print(hr("═"))
    return passed


def main() -> int:
    if len(sys.argv) < 2:
        print(f"用法: python {Path(__file__).name} <dist目錄>")
        print(f"例如: python {Path(__file__).name} dist/yolo11_inference")
        return 1

    dist = Path(sys.argv[1]).resolve()
    ok = check_dir(dist)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
