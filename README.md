# Multi-Camera Real-Time Traffic Analytics Pipeline

A backpressure-aware, multi-stream traffic analytics pipeline that performs batched GPU inference on real traffic footage.

*(Attualmente alla **Fase 1**: Pipeline concorrente ottimizzata, ingestion da stream video, dynamic batching, e inferenza su GPU).*

## Architettura

La pipeline è progettata per massimizzare il throughput senza sacrificare la stabilità o la memoria:
- **Multi-Stream Ingestion (Producers):** Thread indipendenti leggono e preprocessano (letterbox resize) i frame per ciascuna telecamera.
- **Bounded Queues & Backpressure:** Ciascun thread comunica tramite code thread-safe. Se l'inferenza è più lenta della lettura, i frame vecchi vengono scartati (`drop_policy: "latest"`) per garantire una latenza end-to-end estremamente bassa.
- **Dynamic Batcher (Consumer):** Un singolo loop centrale raccoglie i frame da tutte le telecamere, costruisce un tensor batch, e richiede l'inferenza.
- **ONNX FP16 GPU Inference:** Modello YOLOv26 esportato in formato ONNX half-precision (FP16) per massimizzare il throughput tramite i Tensor Cores della GPU (batch size: 4).
- **Metriche Dettagliate:** Monitoraggio costante di FPS, latenza (p50, p95), frame scartati ed errori di lettura, con un report JSON generato alla fine di ogni esecuzione.

## Requisiti di Sistema

- **OS:** Linux (testato su Ubuntu in WSL2)
- **Hardware:** CPU Multi-core, GPU NVIDIA compatibile (testato su RTX 3060 12GB)
- **Software:** 
  - Python 3.10+
  - PyTorch con supporto CUDA (es. `2.13.0+cu130`)
  - `onnxruntime-gpu` (versione `1.28.0` testata)
  - `ultralytics`
  - `opencv-python`
  - `pyyaml`
  - Driver NVIDIA e toolkit CUDA/cuDNN correttamente installati e visibili dal sistema operativo.

*(Le dipendenze esatte verranno formalizzate in un file `requirements.txt` nelle fasi successive).*

## Quickstart

L'applicazione è configurabile esternamente tramite file YAML. Assicurati di avere l'ambiente virtuale (`venv`) attivato prima di eseguire.

### Esecuzione Standard
L'esecuzione di default (file `configs/default.yaml`) elabora tutti i frame disponibili nei video e scarta attivamente frame se le code si riempiono, privilegiando una latenza in tempo reale:

```bash
source venv/bin/activate
python -m src.main
```

### Esecuzione Benchmark
Per eseguire un test di benchmark riproducibile (senza frame-dropping e limitato a 150 frame per stream) per misurare accuratamente il throughput:

```bash
source venv/bin/activate
python -m src.main --config configs/benchmark_replicated_streams.yaml
```

I risultati dettagliati verranno mostrati a schermo sotto forma di tabella Markdown e salvati come JSON all'interno della cartella `outputs/`.

## Limitazioni (Fase 1)
- Il sistema elabora 4 stream replicando un video locale di test per verificare la scalabilità della pipeline (`simulated_streams`). L'integrazione di dataset reali con clip indipendenti è programmata per la prossima fase.
- Le performance dipendono fortemente dall'I/O del disco e dalla generazione hardware della GPU utilizzata.
- Non è ancora presente logica di Tracking o Analytics, il sistema attualmemte effettua soltanto l'inferenza ad alta velocità e scarta l'output per misurarne i tempi.
