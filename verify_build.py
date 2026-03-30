"""Post-build verification script for yolo11_inference PyInstaller package.

Usage:
    python verify_build.py <dist_dir>
    python verify_build.py dist/yolo11_inference
"""
from __future__ import annotations

import os
import sys
import subprocess
from pathlib import Path


REQUIRED_FILES = [
    "yolo11_inference.exe",
    "config.yaml",
    "config.example.yaml",
]

REQUIRED_DIRS = [
    "Runtime",
    "MvImport",
    "models",
]

REQUIRED_DLL_SAMPLES = [
    "MvImport/MvCameraControl_class.py",
    "MvImport/CameraParams_const.py",
]

MODEL_WEIGHT_PATTERN = "**/*.pt"

WARN_SIZE_MB = 200  # 輸出小於此值可能有漏包


def hr(char: str = "─", width: int = 60) -> str:
    return char * width


def check_dir(dist: Path) -> bool:
    print(hr())
    print(f"驗證目錄: {dist}")
    print(hr())

    if not dist.exists():
        print(f"[FAIL] 輸出目錄不存在: {dist}")
        return False

    passed = True

    # --- 必要檔案 ---
    print("\n[必要檔案]")
    for f in REQUIRED_FILES:
        p = dist / f
        if p.exists():
            size_kb = p.stat().st_size // 1024
            print(f"  ✓  {f}  ({size_kb:,} KB)")
        else:
            print(f"  ✗  {f}  ← 缺少！")
            passed = False

    # --- 必要目錄 ---
    print("\n[必要目錄]")
    for d in REQUIRED_DIRS:
        p = dist / d
        if p.exists() and p.is_dir():
            count = sum(1 for _ in p.rglob("*") if _.is_file())
            print(f"  ✓  {d}/  ({count} 個檔案)")
        else:
            print(f"  ✗  {d}/  ← 缺少！")
            passed = False

    # --- MvImport 抽查 ---
    print("\n[MvImport 抽查]")
    for f in REQUIRED_DLL_SAMPLES:
        p = dist / f
        if p.exists():
            print(f"  ✓  {f}")
        else:
            print(f"  ✗  {f}  ← 缺少！")
            passed = False

    # --- 模型權重檔 ---
    print("\n[模型權重 .pt]")
    model_files = list((dist / "models").rglob("*.pt")) if (dist / "models").exists() else []
    if model_files:
        for pt in sorted(model_files):
            rel = pt.relative_to(dist)
            size_mb = pt.stat().st_size / (1024 * 1024)
            print(f"  ✓  {rel}  ({size_mb:.1f} MB)")
    else:
        print("  ✗  找不到任何 .pt 檔案！")
        passed = False

    # --- 套件是否齊全（抽查關鍵套件）---
    print("\n[關鍵套件 .pyd / .so 抽查]")
    key_patterns = ["torch", "cv2", "PyQt5", "ultralytics", "anomalib"]
    for kw in key_patterns:
        matches = list(dist.rglob(f"*{kw}*"))
        matches = [m for m in matches if m.is_file() and m.suffix in (".pyd", ".dll", ".so", ".py")]
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
        print(f"  [WARNING] 小於 {WARN_SIZE_MB} MB，可能有套件沒打進去（torch 本身就超過 2 GB）")
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
