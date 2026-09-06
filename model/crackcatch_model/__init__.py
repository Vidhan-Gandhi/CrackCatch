"""CrackCatch detection & analysis package.

Implements stages 1-5 of the CrackCatch pipeline:

1. Video/Image Capture      -> :mod:`crackcatch_model.sources`
2. Frame Preprocessing      -> :mod:`crackcatch_model.preprocess`
3. AI Detection Engine      -> :mod:`crackcatch_model.detector`
4. Severity & Size          -> :mod:`crackcatch_model.severity`
5. GPS Geo-Tagging          -> :mod:`crackcatch_model.geotag`

Stages 6 (storage) and 7 (dashboard) live in ``/backend`` and ``/frontend``.
Stage 8 (retraining loop) lives in ``/model/scripts``.

The package is deliberately free of any database or web-framework import so
that the detection stack can be reused from a CLI, a notebook, an edge device
or the FastAPI service without modification.
"""

from crackcatch_model.types import (
    BoundingBox,
    Detection,
    DefectRecord,
    GeoPoint,
    Severity,
    SizeEstimate,
)

__version__ = "1.0.0"

__all__ = [
    "BoundingBox",
    "Detection",
    "DefectRecord",
    "GeoPoint",
    "Severity",
    "SizeEstimate",
    "__version__",
]
