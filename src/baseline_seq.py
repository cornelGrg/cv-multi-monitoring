"""
Baseline Sequenziale — cv-multi-monitoring
============================================
Pipeline di inferenza sequenziale (non ottimizzata) su 4 stream video.

Per ogni step del ciclo:
  1. Legge 1 frame da ciascuna delle 4 telecamere
  2. Esegue l'inferenza YOLO singolarmente su ogni frame (no batching)

Misura il throughput escludendo il tempo di caricamento del modello.
"""
import ctypes
import os
import site
import time
from pathlib import Path

# ── Pre-caricamento librerie CUDA per ONNX Runtime GPU ──────────────────────
# LD_LIBRARY_PATH viene letto dal dynamic linker solo all'avvio del processo.
# Per rendere le librerie CUDA disponibili a ONNX Runtime dobbiamo pre-caricarle
# con ctypes.CDLL (RTLD_GLOBAL) prima che onnxruntime venga importato.
_site = Path(site.getsitepackages()[0])
_cuda_lib = _site / "nvidia" / "cu13" / "lib"
_cudnn_lib = _site / "nvidia" / "cudnn" / "lib"

_libs_to_preload = [
    _cuda_lib / "libcublas.so.13",
    _cuda_lib / "libcublasLt.so.13",
    _cuda_lib / "libcufft.so.12",
    _cuda_lib / "libcurand.so.10",
    _cuda_lib / "libcusolver.so.12",
    _cuda_lib / "libcusparse.so.12",
    _cuda_lib / "libcudart.so.13",
    _cudnn_lib / "libcudnn.so.9",
]

_loaded = []
for lib_path in _libs_to_preload:
    if lib_path.exists():
        try:
            ctypes.CDLL(str(lib_path), mode=ctypes.RTLD_GLOBAL)
            _loaded.append(lib_path.name)
        except OSError:
            pass
if _loaded:
    print(f"[INFO] Pre-caricate {len(_loaded)} librerie CUDA/cuDNN via ctypes")

import cv2  # noqa: E402
from ultralytics import YOLO  # noqa: E402

# ── Configurazione ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "yolo26m.onnx"
VIDEO_DIR = PROJECT_ROOT / "data" / "videos"

STREAM_NAMES = ["stream1.mp4", "stream2.mp4", "stream3.mp4", "stream4.mp4"]
MAX_FRAMES_PER_STREAM = 150  # 150 × 4 = 600 frame totali


def load_model() -> YOLO:
    """Carica il modello YOLO26m ONNX forzando il provider GPU CUDA."""
    print(f"[INFO] Caricamento modello: {MODEL_PATH.name}")
    model = YOLO(str(MODEL_PATH), task="detect")
    return model


def open_streams() -> list[cv2.VideoCapture]:
    """Apre i 4 stream video e verifica che siano leggibili."""
    caps: list[cv2.VideoCapture] = []
    for name in STREAM_NAMES:
        path = VIDEO_DIR / name
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"Impossibile aprire {path}")
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"  ✓ {name:15s}  {w}×{h}  ({total} frame disponibili)")
        caps.append(cap)
    return caps


def run_baseline(model: YOLO, caps: list[cv2.VideoCapture]) -> dict:
    """
    Ciclo sequenziale: per ogni step legge 1 frame da ogni telecamera
    ed esegue l'inferenza individualmente. Nessun batching né threading.

    Returns:
        Dizionario con le metriche di prestazione.
    """
    num_streams = len(caps)
    total_frames_target = MAX_FRAMES_PER_STREAM * num_streams
    frames_processed = 0
    detections_total = 0

    print(f"\n[INFO] Avvio elaborazione sequenziale: "
          f"{MAX_FRAMES_PER_STREAM} frame × {num_streams} stream = "
          f"{total_frames_target} frame totali\n")

    # ── Warm-up: 1 inferenza a vuoto per inizializzare CUDA/ONNX Runtime ──
    ret, warmup_frame = caps[0].read()
    if ret:
        model.predict(warmup_frame, device=0, verbose=False)
        # Riporta il cap al frame 0
        caps[0].set(cv2.CAP_PROP_POS_FRAMES, 0)
    print("[INFO] Warm-up completato, avvio misurazione...\n")

    # ── Ciclo principale (misurato) ─────────────────────────────────────────
    t_start = time.perf_counter()

    for step in range(MAX_FRAMES_PER_STREAM):
        for cam_idx, cap in enumerate(caps):
            ret, frame = cap.read()
            if not ret:
                # Se il video è più corto di 150 frame, loop dall'inizio
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
                if not ret:
                    continue

            # Inferenza singola (sequenziale, no batch)
            results = model.predict(frame, device=0, verbose=False)
            detections_total += len(results[0].boxes)
            frames_processed += 1

        # Progresso ogni 30 step
        if (step + 1) % 30 == 0:
            elapsed = time.perf_counter() - t_start
            fps_now = frames_processed / elapsed
            print(f"  [step {step + 1:>3d}/{MAX_FRAMES_PER_STREAM}] "
                  f"{frames_processed} frame — {fps_now:.1f} FPS globali")

    t_end = time.perf_counter()

    # ── Metriche ────────────────────────────────────────────────────────────
    elapsed_s = t_end - t_start
    global_fps = frames_processed / elapsed_s
    per_cam_fps = global_fps / num_streams

    return {
        "frames_processed": frames_processed,
        "detections_total": detections_total,
        "elapsed_s": elapsed_s,
        "global_fps": global_fps,
        "per_cam_fps": per_cam_fps,
        "num_streams": num_streams,
    }


def print_results(metrics: dict) -> None:
    """Stampa una tabella Markdown formattata con i risultati."""
    print("\n")
    print("=" * 62)
    print("  RISULTATI — Baseline Sequenziale (no batch, no threading)")
    print("=" * 62)
    print()
    print("| Metrica                        | Valore              |")
    print("|:-------------------------------|:--------------------|")
    print(f"| Stream elaborati               | {metrics['num_streams']}                   |")
    print(f"| Frame per stream               | {MAX_FRAMES_PER_STREAM}                 |")
    print(f"| Frame totali elaborati         | {metrics['frames_processed']}                 |")
    print(f"| Detections totali              | {metrics['detections_total']:<20}|")
    print(f"| Tempo totale elaborazione      | {metrics['elapsed_s']:.2f} s{'':<14}|")
    print(f"| **Throughput globale (FPS)**    | **{metrics['global_fps']:.2f} FPS**{'':<10}|")
    print(f"| **Throughput per camera (FPS)** | **{metrics['per_cam_fps']:.2f} FPS**{'':<10}|")
    print()
    print("=" * 62)


def main() -> None:
    # 1. Carica modello
    model = load_model()

    # 2. Apri stream video
    print(f"\n[INFO] Apertura {len(STREAM_NAMES)} stream da: {VIDEO_DIR}")
    caps = open_streams()

    # 3. Esegui pipeline sequenziale
    metrics = run_baseline(model, caps)

    # 4. Rilascia risorse video
    for cap in caps:
        cap.release()

    # 5. Stampa risultati
    print_results(metrics)


if __name__ == "__main__":
    main()
