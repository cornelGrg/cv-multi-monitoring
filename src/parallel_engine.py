"""
Parallel Batched Engine — cv-multi-monitoring
================================================
Pipeline ottimizzata con architettura Producer-Consumer multi-threaded
e GPU dynamic batching per massimizzare il throughput.

Architettura:
  - 4 VideoProducer (threads) → code thread-safe (Queue maxsize=10)
  - 1 Consumer centrale → raccoglie 1 frame per coda → batch di 4 → inferenza GPU
"""

import ctypes
import queue
import site
import threading
import time
from pathlib import Path

import numpy as np

# ── Pre-caricamento librerie CUDA per ONNX Runtime GPU ──────────────────────
_site = Path(site.getsitepackages()[0])
_cuda_lib = _site / "nvidia" / "cu13" / "lib"
_cudnn_lib = _site / "nvidia" / "cudnn" / "lib"

for _lib in [
    _cuda_lib / "libcublas.so.13",
    _cuda_lib / "libcublasLt.so.13",
    _cuda_lib / "libcufft.so.12",
    _cuda_lib / "libcurand.so.10",
    _cuda_lib / "libcusolver.so.12",
    _cuda_lib / "libcusparse.so.12",
    _cuda_lib / "libcudart.so.13",
    _cudnn_lib / "libcudnn.so.9",
]:
    if _lib.exists():
        try:
            ctypes.CDLL(str(_lib), mode=ctypes.RTLD_GLOBAL)
        except OSError:
            pass

import cv2  # noqa: E402
import onnxruntime as ort  # noqa: E402

# ── Configurazione ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "yolo26m.onnx"  # batch=4
VIDEO_DIR = PROJECT_ROOT / "data" / "videos"

STREAM_NAMES = ["stream1.mp4", "stream2.mp4", "stream3.mp4", "stream4.mp4"]
MAX_BATCHES = 150          # 150 batch × 4 frame = 600 frame totali
QUEUE_MAXSIZE = 10         # Dimensione massima coda per evitare sovraccarico RAM
NUM_STREAMS = len(STREAM_NAMES)

# Risultati baseline per confronto
BASELINE_GLOBAL_FPS = 51.0
BASELINE_PER_CAM_FPS = 12.7
BASELINE_ELAPSED_S = 11.8


# ═══════════════════════════════════════════════════════════════════════════
#  VideoProducer — Thread che legge frame da un video stream
# ═══════════════════════════════════════════════════════════════════════════

class VideoProducer(threading.Thread):
    """
    Thread produttore: apre uno stream video con cv2.VideoCapture e inserisce
    i frame letti in una coda thread-safe a dimensione limitata.

    La coda con maxsize funge da meccanismo di backpressure: se il consumer
    è più lento, il producer si blocca evitando sovraccarico di RAM.

    Args:
        video_path: Percorso del file video.
        frame_queue: Coda thread-safe in cui inserire i tensori preprocessati.
        max_frames: Numero massimo di frame da leggere.
        name: Nome del thread (per logging).
    """

    def __init__(
        self,
        video_path: str,
        frame_queue: queue.Queue,
        max_frames: int,
        name: str = "",
    ):
        super().__init__(daemon=True, name=name)
        self.video_path = video_path
        self.frame_queue = frame_queue
        self.max_frames = max_frames
        self._stop_event = threading.Event()

    def run(self) -> None:
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Impossibile aprire {self.video_path}")

        frames_read = 0
        while frames_read < self.max_frames and not self._stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                # Se il video è più corto di max_frames, loop dall'inizio
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
                if not ret:
                    break

            # Preprocessing nel thread producer (parallelizzato su 4 core)
            tensor = preprocess_frame(frame)

            # Blocca se la coda è piena (backpressure naturale)
            try:
                self.frame_queue.put(tensor, timeout=5.0)
            except queue.Full:
                break
            frames_read += 1

        # Sentinella per segnalare il completamento
        self.frame_queue.put(None, timeout=5.0)
        cap.release()

    def stop(self) -> None:
        """Segnala al thread di fermarsi."""
        self._stop_event.set()


# ═══════════════════════════════════════════════════════════════════════════
#  Preprocessing YOLO
# ═══════════════════════════════════════════════════════════════════════════

def preprocess_frame(frame: np.ndarray, input_size: int = 640) -> np.ndarray:
    """
    Letterbox resize + normalizzazione per YOLO inference.

    Trasforma un frame BGR (H, W, 3) uint8 in un tensore (3, 640, 640) float32
    normalizzato [0, 1] con padding grigio (114).
    """
    h, w = frame.shape[:2]
    scale = min(input_size / h, input_size / w)
    nh, nw = int(h * scale), int(w * scale)

    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)

    # Canvas con padding grigio
    canvas = np.full((input_size, input_size, 3), 114, dtype=np.uint8)
    top = (input_size - nh) // 2
    left = (input_size - nw) // 2
    canvas[top : top + nh, left : left + nw] = resized

    # HWC → CHW, BGR → RGB, normalize [0, 1]
    canvas = canvas[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    return canvas


# ═══════════════════════════════════════════════════════════════════════════
#  ONNX Runtime Session
# ═══════════════════════════════════════════════════════════════════════════

def create_session(model_path: Path) -> ort.InferenceSession:
    """Crea una sessione ONNX Runtime con CUDAExecutionProvider."""
    providers = [
        ("CUDAExecutionProvider", {"device_id": 0}),
        "CPUExecutionProvider",
    ]
    session = ort.InferenceSession(str(model_path), providers=providers)
    active = session.get_providers()
    print(f"[INFO] ONNX Runtime {ort.__version__} — providers: {active}")
    return session


def count_detections(output: np.ndarray, conf_threshold: float = 0.25) -> int:
    """
    Conta le detections valide dall'output YOLO end2end.

    Output shape: (batch, max_det, 6) dove 6 = [x1, y1, x2, y2, conf, class_id].
    Le entries di padding hanno conf ≈ 0.
    """
    if output.ndim == 3:
        # (batch, 300, 6)
        confs = output[:, :, 4]
        return int(np.sum(confs > conf_threshold))
    elif output.ndim == 2:
        # (300, 6)
        return int(np.sum(output[:, 4] > conf_threshold))
    return 0


# ═══════════════════════════════════════════════════════════════════════════
#  Consumer — Batched inference loop
# ═══════════════════════════════════════════════════════════════════════════

def run_parallel_batched(
    session: ort.InferenceSession,
    queues: list[queue.Queue],
) -> dict:
    """
    Consumer centrale: raccoglie 1 frame da ciascuna coda, costruisce un batch
    di 4 frame e lancia l'inferenza GPU in un singolo passaggio.

    Returns:
        Dizionario con le metriche di prestazione.
    """
    input_name = session.get_inputs()[0].name
    frames_processed = 0
    detections_total = 0

    print(f"\n[INFO] Avvio elaborazione: {MAX_BATCHES} batch × {NUM_STREAMS} frame = "
          f"{MAX_BATCHES * NUM_STREAMS} frame totali\n")

    t_start = time.perf_counter()

    for step in range(MAX_BATCHES):
        # ── Raccolta: 1 tensore preprocessato da ciascuna delle 4 code ──
        tensors: list[np.ndarray] = []
        for q in queues:
            tensor = q.get(timeout=5.0)
            if tensor is None:
                break
            tensors.append(tensor)

        if len(tensors) < NUM_STREAMS:
            print(f"[WARN] Stream terminato prematuramente allo step {step}")
            break

        # ── Batch: stack tensori già preprocessati → (4, 3, 640, 640) ───
        batch = np.stack(tensors, axis=0)

        # ── Inferenza GPU singola su tutto il batch ─────────────────────
        outputs = session.run(None, {input_name: batch})
        detections_total += count_detections(outputs[0])
        frames_processed += NUM_STREAMS

        # Progresso ogni 30 batch
        if (step + 1) % 30 == 0:
            elapsed = time.perf_counter() - t_start
            fps = frames_processed / elapsed
            print(f"  [batch {step + 1:>3d}/{MAX_BATCHES}] "
                  f"{frames_processed} frame — {fps:.1f} FPS globali")

    t_end = time.perf_counter()

    elapsed_s = t_end - t_start
    global_fps = frames_processed / elapsed_s
    per_cam_fps = global_fps / NUM_STREAMS
    speedup_pct = ((global_fps - BASELINE_GLOBAL_FPS) / BASELINE_GLOBAL_FPS) * 100

    return {
        "frames_processed": frames_processed,
        "detections_total": detections_total,
        "elapsed_s": elapsed_s,
        "global_fps": global_fps,
        "per_cam_fps": per_cam_fps,
        "speedup_pct": speedup_pct,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Output formattato
# ═══════════════════════════════════════════════════════════════════════════

def print_comparison(metrics: dict) -> None:
    """Stampa una tabella Markdown che confronta baseline vs parallel engine."""
    m = metrics
    print("\n")
    print("=" * 72)
    print("  CONFRONTO — Baseline Sequenziale vs Parallel Batched Engine")
    print("=" * 72)
    print()
    print("| Metrica                        | Baseline Seq.  | Parallel Batched  |")
    print("|:-------------------------------|:---------------|:------------------|")
    print(f"| Modalità                       | Sequenziale    | MT + GPU Batching |")
    print(f"| Stream                         | 4              | {NUM_STREAMS}                 |")
    print(f"| Frame per stream               | 150            | {MAX_BATCHES}               |")
    print(f"| Frame totali                   | 600            | {m['frames_processed']}               |")
    print(f"| Detections totali              | 232            | {m['detections_total']:<18}|")
    print(f"| Tempo totale (s)               | ~{BASELINE_ELAPSED_S} s         | {m['elapsed_s']:.2f} s{'':<12}|")
    print(f"| **Throughput globale (FPS)**    | **~{BASELINE_GLOBAL_FPS:.1f} FPS**  | **{m['global_fps']:.2f} FPS**{'':<6}|")
    print(f"| **Throughput per camera (FPS)** | **~{BASELINE_PER_CAM_FPS:.1f} FPS**  | **{m['per_cam_fps']:.2f} FPS**{'':<6}|")
    print(f"| **Speedup vs Baseline**        | —              | **{m['speedup_pct']:+.1f}%**{'':<10}|")
    print()
    print("=" * 72)


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 72)
    print("  PARALLEL BATCHED ENGINE — cv-multi-monitoring")
    print("=" * 72)

    # 1. Crea sessione ONNX Runtime su CUDA (modello batch=4)
    print(f"\n[INFO] Caricamento modello: {MODEL_PATH.name} (batch=4)")
    session = create_session(MODEL_PATH)

    input_meta = session.get_inputs()[0]
    output_meta = session.get_outputs()[0]
    print(f"[INFO] Input:  {input_meta.name} → {input_meta.shape}")
    print(f"[INFO] Output: {output_meta.name} → {output_meta.shape}")

    # 2. Crea code e producer
    print(f"\n[INFO] Apertura {NUM_STREAMS} stream da: {VIDEO_DIR}")
    queues: list[queue.Queue] = []
    producers: list[VideoProducer] = []

    for name in STREAM_NAMES:
        path = str(VIDEO_DIR / name)
        q: queue.Queue = queue.Queue(maxsize=QUEUE_MAXSIZE)
        p = VideoProducer(path, q, MAX_BATCHES, name=f"Producer-{name}")
        queues.append(q)
        producers.append(p)

        cap = cv2.VideoCapture(path)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        print(f"  ✓ {name:15s}  {w}×{h}  ({total} frame disponibili)")

    # 3. Warm-up GPU (3 batch dummy da 4 frame)
    dummy = np.random.rand(NUM_STREAMS, 3, 640, 640).astype(np.float32)
    for _ in range(3):
        session.run(None, {input_meta.name: dummy})
    print("\n[INFO] Warm-up GPU completato (3 batch × 4 frame)")

    # 4. Avvia producer threads
    for p in producers:
        p.start()
    time.sleep(0.2)  # Lascia riempire le code

    # 5. Consumer loop (misurato)
    metrics = run_parallel_batched(session, queues)

    # 6. Cleanup threads
    for p in producers:
        p.stop()
        p.join(timeout=2.0)

    # 7. Stampa confronto
    print_comparison(metrics)


if __name__ == "__main__":
    main()
