import logging
import uuid


class ContextFilter(logging.Filter):
    """Ensure LogRecord has context fields to avoid KeyError in formatters."""

    def filter(self, record: logging.LogRecord) -> bool:  # pragma: no cover
        for key in ("product", "area", "infer_type", "request_id"):
            if not hasattr(record, key):
                setattr(record, key, "-")
        return True


def _has_context_filter(target: logging.Filterer) -> bool:
    return any(isinstance(flt, ContextFilter) for flt in getattr(target, "filters", []))


def install_context_filter(root: logging.Logger) -> None:
    """Attach ContextFilter to the provided logger and its current handlers."""
    if not _has_context_filter(root):
        filter_instance = ContextFilter()
        root.addFilter(filter_instance)
    else:
        # Reuse the existing instance on the root logger
        filter_instance = next(
            flt for flt in root.filters if isinstance(flt, ContextFilter)
        )
    for handler in root.handlers:
        if not _has_context_filter(handler):
            handler.addFilter(filter_instance)


def context_adapter(
    logger: logging.Logger,
    product: str,
    area: str,
    infer_type: str,
    request_id: str | None = None,
) -> logging.LoggerAdapter:
    extra = {
        "product": product,
        "area": area,
        "infer_type": infer_type,
        "request_id": request_id or str(uuid.uuid4())[:8],
    }
    return logging.LoggerAdapter(logger, extra)
