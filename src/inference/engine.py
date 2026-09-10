"""
ONNX Runtime GPU inference engine.

Encapsulates CUDA library pre-loading, session creation, warm-up, and
batched inference with per-call timing.
"""

from __future__ import annotations

import ctypes
import logging
import site
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def _preload_cuda_libs() -> int:
    """Pre-load NVIDIA shared libraries so ONNX Runtime finds them via dlopen.

    Returns the number of libraries successfully loaded.
    """
    sp = Path(site.getsitepackages()[0])
    cuda_lib = sp / "nvidia" / "cu13" / "lib"
    cudnn_lib = sp / "nvidia" / "cudnn" / "lib"

    candidates = [
        cuda_lib / "libcublas.so.13",
        cuda_lib / "libcublasLt.so.13",
        cuda_lib / "libcufft.so.12",
        cuda_lib / "libcurand.so.10",
        cuda_lib / "libcusolver.so.12",
        cuda_lib / "libcusparse.so.12",
        cuda_lib / "libcudart.so.13",
        cudnn_lib / "libcudnn.so.9",
    ]

    loaded = 0
    for lib_path in candidates:
        if lib_path.exists():
            try:
                ctypes.CDLL(str(lib_path), mode=ctypes.RTLD_GLOBAL)
                loaded += 1
            except OSError:
                pass
    return loaded


# Pre-load at import time (before onnxruntime is imported anywhere else)
_n_preloaded = _preload_cuda_libs()

import onnxruntime as ort  # noqa: E402


class OnnxGpuEngine:
    """Manages an ONNX Runtime inference session on GPU.

    Parameters
    ----------
    model_path:
        Path to the ``.onnx`` model file.
    provider:
        Execution provider name, e.g. ``"CUDAExecutionProvider"``.
    device_id:
        GPU device index.
    warmup_batches:
        Number of dummy inference passes to run during ``warmup()``.
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        provider: str = "CUDAExecutionProvider",
        device_id: int = 0,
        warmup_batches: int = 3,
    ) -> None:
        self.model_path = Path(model_path)
        self._warmup_count = warmup_batches

        providers: list = ["CPUExecutionProvider"]
        if provider != "CPUExecutionProvider":
            providers.insert(0, (provider, {"device_id": device_id}))

        logger.info("Loading %s for ONNX Runtime inference", self.model_path.name)
        self._session = ort.InferenceSession(str(self.model_path), providers=providers)

        self._input_meta = self._session.get_inputs()[0]
        self._output_meta = self._session.get_outputs()[0]

        active = self._session.get_providers()
        logger.info(
            "ONNX Runtime %s — providers: %s  (preloaded %d CUDA libs)",
            ort.__version__,
            active,
            _n_preloaded,
        )

    # ── Properties ──────────────────────────────────────────────────────

    @property
    def input_name(self) -> str:
        return self._input_meta.name

    @property
    def input_shape(self) -> list:
        return self._input_meta.shape

    @property
    def output_shape(self) -> list:
        return self._output_meta.shape

    @property
    def providers(self) -> list[str]:
        return self._session.get_providers()

    # ── Warm-up ─────────────────────────────────────────────────────────

    def warmup(self) -> None:
        """Run a few dummy inferences to initialise CUDA kernels."""
        shape = list(self._input_meta.shape)
        # Replace any symbolic dims with concrete values
        shape = [s if isinstance(s, int) else 1 for s in shape]
        dummy = np.random.rand(*shape).astype(np.float32)
        for _ in range(self._warmup_count):
            self._session.run(None, {self.input_name: dummy})
        logger.info("Warm-up completed (%d batches)", self._warmup_count)

    # ── Inference ───────────────────────────────────────────────────────

    def infer(self, batch: np.ndarray) -> tuple[np.ndarray, float]:
        """Run inference on a pre-built batch.

        Parameters
        ----------
        batch:
            NumPy array of shape ``(B, C, H, W)`` float32.

        Returns
        -------
        tuple of (detections, inference_time_s)
            ``detections`` has the model's native output shape (e.g.
            ``(B, 300, 6)`` for end2end YOLO).
            ``inference_time_s`` is wall-clock seconds for this call.
        """
        t0 = time.perf_counter()
        outputs = self._session.run(None, {self.input_name: batch})
        t1 = time.perf_counter()
        return outputs[0], t1 - t0
