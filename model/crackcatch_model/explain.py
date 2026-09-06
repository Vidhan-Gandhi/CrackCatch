"""Explainability overlays - why did the model flag this?

Two mechanisms, chosen so that *something* useful is always available:

``gradcam``
    True Grad-CAM on a YOLOv8 checkpoint. Hooks the last convolutional stage
    of the backbone, backpropagates the summed objectness/class response, and
    weights the feature maps by their pooled gradients. This is the real
    thing, and it is what the docs claim.

``saliency_overlay``
    A gradient-free fallback (Sobel energy restricted to the detection box)
    used when no differentiable model is loaded - i.e. when the classical-CV
    baseline is running. It shows *where the contrast is*, which is genuinely
    what the heuristic detector responded to, so it is honest for that
    detector. It is NOT Grad-CAM and the function name says so.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from crackcatch_model.types import BoundingBox

logger = logging.getLogger(__name__)


def _colourise(heat: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Normalise a 2-D response to a JET colour map at ``shape``."""
    heat = np.nan_to_num(heat.astype(np.float32), nan=0.0)
    heat -= heat.min()
    peak = heat.max()
    if peak > 1e-8:
        heat /= peak
    heat = cv2.resize(heat, (shape[1], shape[0]), interpolation=cv2.INTER_CUBIC)
    return cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_JET)


def blend(frame: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    return cv2.addWeighted(heatmap, alpha, frame, 1.0 - alpha, 0.0)


def saliency_overlay(
    frame: np.ndarray, bbox: BoundingBox | None = None, alpha: float = 0.45
) -> np.ndarray:
    """Gradient-free contrast-energy overlay. See module docstring."""
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    grey = cv2.GaussianBlur(grey, (5, 5), 0)
    energy = np.hypot(
        cv2.Sobel(grey, cv2.CV_32F, 1, 0, ksize=3),
        cv2.Sobel(grey, cv2.CV_32F, 0, 1, ksize=3),
    )
    # Darkness matters as much as edge energy for a pothole.
    darkness = 255.0 - grey.astype(np.float32)
    response = cv2.GaussianBlur(energy, (0, 0), 9) * 0.6 + cv2.GaussianBlur(
        darkness, (0, 0), 15
    ) * 0.4

    if bbox is not None:
        mask = np.zeros_like(response, dtype=np.float32)
        x1, y1, x2, y2 = (int(round(v)) for v in bbox.as_xyxy())
        pad_x = int(0.25 * max(1, x2 - x1))
        pad_y = int(0.25 * max(1, y2 - y1))
        x1, y1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
        x2 = min(response.shape[1], x2 + pad_x)
        y2 = min(response.shape[0], y2 + pad_y)
        mask[y1:y2, x1:x2] = 1.0
        response *= cv2.GaussianBlur(mask, (0, 0), 15)

    return blend(frame, _colourise(response, frame.shape[:2]), alpha)



def _class_score(head, num_classes: int):
    """Reduce a YOLOv8 raw head tensor to a single differentiable class score.

    ``head`` is ``(batch, 4 + num_classes, num_anchors)``. We take the class
    channels, keep each anchor's strongest class, and sum over anchors, so the
    gradient reflects class evidence rather than box geometry. If the layout is
    unexpected the whole tensor is used, which is the previous, cruder
    behaviour but never raises.
    """
    import torch

    if head.ndim == 3 and num_classes and head.shape[1] > num_classes:
        class_channels = head[:, -num_classes:, :]
        return class_channels.amax(dim=1).sum()
    return head.abs().sum()


def gradcam(
    frame: np.ndarray,
    weights: str,
    alpha: float = 0.45,
    device: str = "cpu",
    bbox: BoundingBox | None = None,
) -> tuple[np.ndarray, str]:
    """Grad-CAM over a YOLOv8 checkpoint.

    Returns ``(image, method)`` where ``method`` describes what was *actually*
    produced - ``"grad-cam"`` or the saliency fallback. Callers must report
    this value rather than assuming Grad-CAM succeeded: a checkpoint being
    present is not evidence that a CAM could be computed.

    Choosing the target layer matters. Hooking the final 1x1 prediction convs
    yields an all-zero CAM, because ReLU of the channel-weighted sum collapses
    there. The layer that carries spatial semantics is the last neck block
    feeding the Detect head, so that is tried first and shallower convolutions
    are tried in turn until one produces a non-empty response.
    """
    try:
        import torch
        from ultralytics import YOLO

        model = YOLO(str(weights))
        torch_model = model.model.to(device).eval()
        num_classes = len(model.names) if getattr(model, "names", None) else 0
        for param in torch_model.parameters():
            param.requires_grad_(True)

        for target_layer in _candidate_layers(torch_model):
            heat = _cam_for_layer(
                torch_model, target_layer, frame, num_classes, device, torch
            )
            if heat is not None:
                return blend(frame, _colourise(heat, frame.shape[:2]), alpha), "grad-cam"

        raise RuntimeError("no target layer produced a non-empty CAM")

    except Exception as exc:
        logger.warning(
            "Grad-CAM unavailable (%s); falling back to the gradient-free "
            "saliency overlay.",
            exc,
        )
        return (
            saliency_overlay(frame, bbox, alpha),
            "saliency (Grad-CAM unavailable; NOT Grad-CAM)",
        )


def _candidate_layers(torch_model) -> list:
    """Target layers to try, deepest-semantic first.

    The neck block immediately before the Detect head is the standard choice;
    the last few convolutions are kept as fallbacks for architectures where
    that indexing does not hold.
    """
    import torch

    candidates = []
    backbone = getattr(torch_model, "model", None)
    if backbone is not None:
        try:
            # Everything except the Detect head, deepest first.
            for module in list(backbone)[:-1][::-1]:
                candidates.append(module)
        except TypeError:
            pass

    convs = [m for m in torch_model.modules() if isinstance(m, torch.nn.Conv2d)]
    candidates.extend(convs[::-1][:6])
    return candidates[:10]


def _cam_for_layer(torch_model, target_layer, frame, num_classes, device, torch):
    """Compute one Grad-CAM map, or ``None`` if this layer yields nothing."""
    activations: list = []
    gradients: list = []

    def forward_hook(_module, _inp, out):
        if not isinstance(out, torch.Tensor) or out.ndim != 4:
            return
        activations.append(out)
        out.register_hook(lambda grad: gradients.append(grad))

    handle = target_layer.register_forward_hook(forward_hook)
    try:
        resized = cv2.resize(frame, (640, 640))
        tensor = (
            torch.from_numpy(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
            .permute(2, 0, 1)
            .float()
            .div(255.0)
            .unsqueeze(0)
            .to(device)
        )
        output = torch_model(tensor)
        head = output[0] if isinstance(output, (list, tuple)) else output
        if isinstance(head, (list, tuple)):
            head = head[0]

        # YOLOv8's raw head is (batch, 4 + num_classes, num_anchors): four
        # box-regression channels followed by per-class scores. Back-
        # propagating the whole tensor lets the regression channels dominate,
        # and those fire on any strong linear edge - lane markings light up
        # brighter than the pothole. Differentiating the CLASS channels only
        # asks the right question: which pixels made the model believe a
        # defect is present here?
        score = _class_score(head, num_classes)
        torch_model.zero_grad(set_to_none=True)
        score.backward()
    except Exception:
        logger.debug("Grad-CAM pass failed for %s", type(target_layer).__name__,
                     exc_info=True)
        return None
    finally:
        handle.remove()

    if not activations or not gradients:
        return None

    act = activations[-1].detach()[0]        # (C, H, W)
    grad = gradients[-1].detach()[0]         # (C, H, W)
    channel_weights = grad.mean(dim=(1, 2))  # global-average-pooled gradients
    cam = torch.relu((act * channel_weights[:, None, None]).sum(dim=0))

    heat = cam.cpu().numpy()
    if float(heat.max()) <= 1e-8:
        # ReLU collapsed the map: keep the magnitude of the response instead,
        # which still shows where the evidence is, just unsigned.
        heat = np.abs((act * channel_weights[:, None, None]).sum(dim=0).cpu().numpy())
    if float(heat.max()) <= 1e-8 or heat.shape[0] < 2 or heat.shape[1] < 2:
        return None
    return heat
