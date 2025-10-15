"""簡易工具，用來尋找程式碼中既定關鍵字所在行號。"""

from pathlib import Path

entries = {
    "core/config.py": [
        "Pydantic schemas unavailable",
        "self.config_path = resolved_config",
    ],
    "GUI.py": ["if not self.detection_system:"],
}
for path, patterns in entries.items():
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    for pat in patterns:
        for idx, line in enumerate(lines, 1):
            if pat in line:
                print(f"{path}:{idx}")
                break
