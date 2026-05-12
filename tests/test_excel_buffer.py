from __future__ import annotations

from core.services.results.excel_buffer import ExcelWorkbookBuffer


class DummyLogger:
    def __init__(self) -> None:
        self.warnings: list[str] = []
        self.infos: list[str] = []

    def warning(self, message, *args):
        self.warnings.append(message % args if args else message)

    def info(self, message, *args):
        self.infos.append(message % args if args else message)

    def error(self, message, *args):
        pass


def test_excel_buffer_rebuilds_corrupt_workbook(tmp_path):
    path = tmp_path / "results.xlsx"
    path.write_text("not a real xlsx", encoding="utf-8")
    logger = DummyLogger()

    buffer = ExcelWorkbookBuffer(
        path=str(path),
        columns=["id", "status"],
        buffer_limit=10,
        logger=logger,
    )

    assert path.exists()
    assert buffer.ws.max_row == 1
    assert list(tmp_path.glob("results.xlsx.corrupt_*"))
    assert logger.warnings
    buffer.close()
