# Roadmap v2 — Multi-Camera Real-Time Traffic Analytics Pipeline

## 1. Obiettivo del progetto

Trasformare il prototipo YOLO in un progetto portfolio-ready di **traffic video analytics**: una pipeline concorrente che acquisisce più flussi, esegue inferenza GPU batchata, traccia veicoli per camera, conta attraversamenti direzionali e produce benchmark riproducibili.

Titolo finale:

> **Multi-Camera Real-Time Traffic Analytics with Batched GPU Inference and Vehicle Tracking**

Descrizione breve:

> A backpressure-aware, multi-stream traffic analytics pipeline that performs batched GPU inference, per-camera multi-object tracking, directional vehicle counting, and event logging on real traffic footage.

## 2. Principi e limiti di scope

- Il caso d'uso e' **traffic analytics**, non action detection.
- L'inferenza e' ONNX FP16 su NVIDIA RTX 3060; non dichiarare TensorRT finche' non e' integrato e misurato.
- Le sequenze UA-DETRAC sono quattro input **indipendenti**, non camere sincronizzate della stessa rete stradale.
- Il batching puo' essere condiviso; tracker, ROI, stato e contatori devono rimanere isolati per `camera_id`.
- Non aggiungere riconoscimento targhe, re-identification cross-camera, dashboard enterprise o training da zero prima della pubblicazione.
- Ogni claim prestazionale deve dichiarare modalita' di test, durata, numero di stream, risoluzione, modello e hardware.

## 3. Stato corrente

### Completato

- Ubuntu/WSL2 con GPU NVIDIA RTX 3060 12 GB.
- YOLO26m esportato in ONNX FP16.
- Pipeline producer-consumer con code bounded/thread-safe e dynamic batching.
- Baseline iniziale con quattro repliche della stessa clip.
- Metriche globali/per-camera, latenza, queue wait, VRAM e drop rate.
- Gestione shutdown pulito e fine video.
- Script di setup UA-DETRAC che scarica, estrae selettivamente frame/XML, converte quattro sequenze in MP4 H.264 a 25 FPS e genera un manifest YAML.
- Realtime pacing configurabile tramite `source_fps` nel producer.
- Test con quattro sequenze UA-DETRAC differenti completato.

### Risultato da classificare correttamente

Il test UA-DETRAC ha elaborato 2.572 frame in 38,04 s, con 67,61 FPS output globali, 5,82% drop rate e 446,89 ms p95 e2e latency. Le clip avevano durate diverse, quindi le camere hanno terminato in momenti differenti. Questo e' un **mixed-duration realtime simulation**, non un benchmark a carico costante di quattro camere per 38 secondi.

Usare questi numeri come controllo di stabilita', non come headline definitiva del progetto.

## 4. Definizione di completamento

Il progetto e' pronto per il portfolio quando:

- elabora quattro stream con inferenza GPU batchata;
- documenta sia `max_throughput` sia `realtime_simulation`;
- esegue un benchmark a quattro stream a carico costante;
- filtra e visualizza detection veicolari corrette;
- esegue tracking multi-oggetto indipendente per camera;
- conta veicoli per classe e direzione tramite line-crossing configurabile;
- salva tracce, eventi, contatori e metriche in file strutturati;
- include un video demo annotato, grafici, README in inglese e istruzioni riproducibili;
- documenta limiti, assunzioni e validazione del conteggio.

## 5. Architettura target

```text
Independent video source per camera
        |
        v
Paced capture worker per camera
        |
        v
Bounded per-camera queue + drop policy
        |
        v
Shared dynamic batcher
        |
        v
ONNX FP16 GPU inference
        |
        v
Detection audit and vehicle-class filtering
        |
        +--> Per-camera multi-object tracker
        |          |
        |          v
        |    Traffic analytics engine
        |    - ROI filtering
        |    - line crossing
        |    - directional counts
        |    - one simple traffic event
        |
        +--> Annotated video writer
        |
        +--> Tracks / counts / events CSV or SQLite
        |
        +--> Metrics collector and benchmark report
```

## 6. Phase 2.1 — Detection audit and constant-load benchmark

### Goal

Validate the detector outputs and establish a fair realtime baseline before adding a tracker.

### Why this phase is required

The current test has nearly full queues (`Q avg` around 8.7 with maximum 9), and the four clips end at different times. In addition, the observed average of about 39.5 detections per inferred frame must be visually checked before it is passed into ByteTrack.

### A. Detection audit

- Render annotated output for at least 100 sampled frames from every selected sequence.
- Verify no duplicated boxes, obviously invalid boxes or raw/unfiltered model outputs are passed downstream.
- Record detection-count p50/p95 per frame, confidence percentiles and class histogram.
- Filter explicitly to traffic classes used by the application: `car`, `truck`, `bus`, `motorcycle` (adapt only after checking the model label map).
- Make confidence threshold, IoU/NMS or equivalent end-to-end post-processing mode configurable.
- Document the mismatch between COCO classes and UA-DETRAC labels, especially for `van`.

### Acceptance criteria

- Annotated frames look plausible to a human observer.
- The pipeline does not feed duplicate/raw predictions to the future tracker.
- Class filtering and confidence threshold are logged in the run metadata.
- A short markdown note records any remaining known false positives.

### B. Constant-load realtime benchmark

- Add a loop option for local video sources or use clips trimmed to equal duration.
- Start all four producers behind a common barrier.
- Run for at least 60 seconds with every producer paced at 25 FPS.
- Run the same configuration three times after a short warm-up; report the median.
- Keep the existing mixed-duration test as a smoke/stability test, but label it as such.

### Required metrics

- global input/output FPS;
- global and per-camera drop rate;
- per-camera active duration;
- per-camera input/output FPS measured over active duration;
- e2e latency p50, p95 and mean;
- inference time mean/p95;
- queue wait mean/p95 and queue average/maximum;
- VRAM;
- batch-size histogram for sizes 1, 2, 3 and 4;
- metrics restricted to the interval where all four cameras are active.

### Acceptance criteria

- The run keeps four sources active for the entire timed interval.
- The report distinguishes input rate, processed rate and intentional drops.
- `max_throughput` and `realtime_simulation` cannot be confused in output file names or documentation.
- Results are written to timestamped JSON/CSV and can be regenerated by a documented command.

## 7. Phase 3 — Per-camera multi-object tracking

### Goal

Assign persistent local identities to vehicles within each camera stream.

### Tasks

- Integrate ByteTrack (preferred) behind a dedicated tracking module.
- Instantiate one tracker per `camera_id`: `trackers[camera_id]`.
- Route each batch result back to its original camera before tracker update.
- Pass only audited, filtered vehicle detections to the tracker.
- Preserve `camera_id`, original source `frame_id` and capture/inference timestamp. Do not renumber frames after drops.
- Configure tracker frame rate and lost-track buffer consistently with 25 FPS and intentional frame dropping.
- Use `camera_id:track_id` as the externally visible identity.
- Overlay class and persistent ID; write a `tracks.csv` containing timestamp, camera, source frame index, track ID, class, confidence, bounding box and box centre.

### Important constraint

Tracking is local to each camera. The same physical vehicle appearing in two different UA-DETRAC sequences must not be assigned a shared identity.

### Acceptance criteria

- Track state is never shared between cameras.
- IDs remain visually stable for normal vehicle transits in two sequences, including one with occlusion.
- No regression in clean shutdown or metric collection.
- The realtime benchmark is repeated with tracking enabled and its overhead is reported.

## 8. Phase 4 — ROI and directional line crossing

### Goal

Convert tracks into reliable traffic-flow counts.

### Tasks

- Add per-camera polygonal ROI configuration. Prefer normalized coordinates or document resolution assumptions.
- Define each virtual line with two points, an ID, direction label, optional lane label and allowed classes.
- For each `track_id + line_id`, retain previous side-of-line and an already-counted flag.
- Count only a transition across the line in the configured direction.
- Keep the tracking state separate from analytics state; analytics must be restartable without changing tracker code.
- Write `counts.csv` and `events.csv` with camera, line, direction, vehicle class, timestamp and track ID.

### Event schema

```json
{
  "timestamp_ms": 12540,
  "camera_id": "cam_02",
  "track_id": "cam_02:17",
  "event_type": "line_crossing",
  "line_id": "northbound_lane_1",
  "direction": "northbound",
  "vehicle_class": "car"
}
```

### Acceptance criteria

- A single track is counted at most once per configured line.
- Overlay shows ROI, virtual lines and current per-direction counters.
- At least two manually reviewed clips have correct visible line-crossing behavior.

## 9. Phase 5 — One simple traffic event

### Goal

Demonstrate that the pipeline produces analytics, not only detections and counts.

### Select exactly one first

1. **Congestion level** — number of active tracked vehicles in a configured ROI, with `low`, `medium` and `high` thresholds.
2. **Stopped vehicle** — a tracked vehicle remaining below a pixel/frame movement threshold for N seconds; explicitly document false alerts caused by red lights.
3. **Queue-length estimate** — active tracked vehicles inside a designated waiting zone.

### Acceptance criteria

- Thresholds are configured per camera.
- State changes are emitted as structured events and displayed in the demo.
- Known false positives are documented.

## 10. Phase 6 — Output and demo

### Required outputs

- Annotated MP4 per camera or a four-panel composite video.
- `metrics.json`/`metrics.csv`.
- `tracks.csv`.
- `counts.csv`.
- `events.csv`.
- A compact static HTML or Markdown benchmark report. A dashboard is optional and must remain minimal.

### Demo acceptance criteria

- A 60–90 second video shows multiple independent cameras, IDs, ROIs, lines, counters, FPS and the selected event.
- Data outputs are understandable without executing the application.
- Any public media uses dataset attribution and is not committed if its licence/distribution rules prohibit it.

## 11. Phase 7 — Benchmark suite

### Test matrix

| ID | Mode | Configuration | Question |
|---|---|---|---|
| B1 | max_throughput | sequential, batch 1 | What is the baseline? |
| B2 | max_throughput | concurrent, batch 1 | What does concurrency contribute? |
| B3 | max_throughput | concurrent, dynamic batch up to 4 | What does batching contribute? |
| B4 | realtime_simulation | 1, 2 and 4 looped streams at 25 FPS | How does it scale under controlled live input? |
| B5 | realtime_simulation | four different UA-DETRAC streams, 60 s, three repetitions | Does it meet the QoS target on realistic inputs? |
| B6 | realtime_simulation | B5 with tracking | What is tracking overhead? |

Only add the following if TensorRT is actually installed, selected by the runtime and benchmarked:

| ID | Mode | Configuration | Question |
|---|---|---|---|
| B7 | realtime_simulation | ONNX Runtime FP16 vs TensorRT FP16 | What deployment gain is measured? |

### Required graphs

- output FPS and input FPS by configuration;
- p50/p95 end-to-end latency;
- drop rate;
- batch-size distribution;
- stage timing: capture, queue, inference, tracking, rendering/output;
- VRAM;
- counts/events for the demo run.

### Claims discipline

- Never divide a per-camera frame count by global elapsed time and label it simply `FPS per camera`; also report camera active time.
- Never call a short mixed-duration file run a sustained four-camera benchmark.
- Report median of three controlled runs, not a single best result.
- State whether output FPS represents processed source frames, rendered frames or inference frames.

## 12. Phase 8 — Functional validation

### Goal

Report a small, honest validation of the counting feature.

### Tasks

- Select two intervals of 30–60 seconds from two different sequences.
- Manually count crossings for each configured line/direction.
- Compare manual and pipeline count using:

```text
count_error_percent = abs(predicted_count - manual_count) / manual_count * 100
```

- Report at least four line/direction examples.
- Log likely causes of every material discrepancy: missed detection, ID switch, occlusion, poor ROI, ambiguous line crossing or duplicate track.

### Acceptance criteria

- The README includes the examples and methodology.
- No uncomputed accuracy, MOT or speed claims are made.
- The UA-DETRAC XML annotations are preserved for future tracking evaluation but are not misrepresented as a full automatic counting ground truth unless such mapping is implemented and verified.

## 13. Repository structure

```text
traffic-analytics/
├── README.md
├── LICENSE
├── pyproject.toml
├── configs/
│   ├── default.yaml
│   ├── benchmark_max_throughput.yaml
│   ├── benchmark_realtime_constant_load.yaml
│   └── cameras/
├── data/
│   ├── manifests/                 # tracked YAML only
│   ├── raw/                       # ignored
│   └── processed/                 # ignored
├── src/
│   ├── capture/
│   ├── pipeline/
│   ├── inference/
│   ├── tracking/
│   ├── analytics/
│   ├── output/
│   └── metrics/
├── scripts/
│   ├── setup_ua_detrac.sh
│   ├── run_benchmark.*
│   └── generate_report.*
├── tests/
│   ├── test_line_crossing.*
│   ├── test_counting.*
│   ├── test_detection_filter.*
│   └── test_shutdown.*
├── docs/
│   ├── architecture.md
│   ├── benchmark_results.md
│   └── images/
└── outputs/                       # ignored except .gitkeep/examples
```

Do not commit dataset/video files, model weights, ONNX/TensorRT engines, large run outputs, local paths, credentials or RTSP URLs.

## 14. README requirements

Write the README in English and include:

1. One-sentence summary and a demo GIF/screenshot.
2. Architecture diagram and explanation of bounded queues, drop policy and dynamic batching.
3. Hardware/software/runtime versions.
4. Dataset source, attribution/licence instructions and note on independent sequences.
5. Quickstart and reproducible benchmark commands.
6. Camera/ROI/line configuration guide.
7. Benchmark table including test mode and duration.
8. Tracking/counting validation method and results.
9. Limitations and future work.

## 15. Limitations to declare

- The four replicate-stream test is only a synthetic scaling benchmark.
- UA-DETRAC demo streams are independent and not cross-camera synchronized.
- Tracking is per camera; no cross-camera vehicle identity is inferred.
- Counts depend on detector quality, tracker continuity, ROI and virtual-line placement.
- Pixel/frame movement is not calibrated physical speed.
- File-based realtime pacing approximates a live source; it is not a production RTSP deployment.
- Results depend on input resolution/codec, model, runtime, drop policy, hardware and driver versions.
- This is a research/portfolio prototype, not a certified traffic-control or surveillance system.

## 16. Scope explicitly deferred

- Cross-camera vehicle re-identification.
- Licence-plate or face recognition.
- Physical speed in km/h without camera calibration.
- Training/fine-tuning a new detector.
- Enterprise dashboard, authentication, cloud orchestration and Kubernetes.
- Complex action recognition or accident prediction.

## 17. Final checklist

- [x] Concurrent bounded-queue pipeline, GPU inference and basic metrics.
- [x] UA-DETRAC preparation script, manifest and four independent local videos.
- [ ] Detection audit and class/confidence filtering.
- [ ] 60-second constant-load, four-camera benchmark repeated three times.
- [ ] Per-camera ByteTrack integration and `tracks.csv`.
- [ ] ROI and configurable directional line crossing.
- [ ] Counts/events export and one simple traffic event.
- [ ] Annotated demo video.
- [ ] B1–B6 benchmarks with generated graphs.
- [ ] Manual count validation on two clips.
- [ ] English README, demo media and cleaned repository.

## 18. CV bullet

Use only after final measured results are available. Replace bracketed fields with controlled benchmark results:

> Developed a backpressure-aware multi-camera traffic analytics pipeline with ONNX FP16 GPU inference, dynamic batching, per-camera multi-object tracking and directional vehicle counting. Sustained [X] processed FPS across four 25-FPS simulated traffic streams on an NVIDIA RTX 3060, with [Y] ms p95 end-to-end latency and [Z]% frame-drop rate under a 60-second constant-load benchmark.

## 19. Future work

- TensorRT FP16 comparison.
- Camera calibration and physical speed estimation.
- Cross-camera vehicle re-identification using CityFlow or AI City Challenge data.
- Adaptive batching driven by queue pressure and latency target.
- MOT-quality evaluation against UA-DETRAC annotations.
- Extended congestion and anomaly analytics.
