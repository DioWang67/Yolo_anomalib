import logging
import uuid


class ContextFilter(logging.Filter):
    """Ensure LogRecord has context fields to avoid KeyError in formatters."""

    def filter(self, record: logging.LogRecord) -> bool:  # pragma: no cover
        for k in ("product", "area", "infer_type", "request_id"):
            if not hasattr(record, k):
                setattr(record, k, "-")
        return True


def install_context_filter(root: logging.Logger) -> None:
    filt = ContextFilter()
    root.addFilter(filt)
    for h in root.handlers:
        h.addFilter(filt)


def context_adapter(logger: logging.Logger, product: str, area: str, infer_type: str, request_id: str | None = None) -> logging.LoggerAdapter:
    extra = {
        "product": product,
        "area": area,
        "infer_type": infer_type,
        "request_id": request_id or str(uuid.uuid4())[:8],
    }
    return logging.LoggerAdapter(logger, extra)

