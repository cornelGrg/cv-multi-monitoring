# Roadmap — Multi-Camera Real-Time Traffic Analytics Pipeline

## 1. Obiettivo del progetto

Trasformare l’attuale prototipo di inferenza YOLO in un progetto portfolio-ready di **video analytics per il traffico urbano**, con pipeline concorrente, inferenza GPU batchata, tracking multi-oggetto, conteggio direzionale e benchmark riproducibili.

Titolo finale:

> **Multi-Camera Real-Time Traffic Analytics with Batched GPU Inference and Vehicle Tracking**

Descrizione breve:

> A backpressure-aware, multi-stream traffic analytics pipeline that performs batched GPU inference, per-camera multi-object tracking, directional vehicle counting, and event logging on real traffic footage.

## 2. Stato iniziale confermato

Sono già completati:

- Ambiente Ubuntu/WSL2 con GPU NVIDIA RTX 3060 12 GB.
- Modello `YOLO26m` esportato in ONNX FP16.
- Quattro stream simulati ottenuti replicando la stessa clip di traffico.
- Pipeline producer–consumer con code thread-safe.
- Dynamic batching su GPU con batch size 4.
- Baseline sequenziale: circa 51 FPS globali.
- Pipeline parallela: circa 22 FPS per camera, 88 FPS globali.

Vincoli da rispettare:

- Non eliminare né riscrivere la pipeline concorrente già funzionante.
- Conservare i benchmark iniziali come riferimento.
- Non chiamare il sistema “action detection”: il caso d’uso è **traffic analytics**.
- Non dichiarare capacità TensorRT finché TensorRT non è realmente integrato e misurato.
- Indicare sempre che lo stress test a quattro stream usa la stessa clip replicata.

## 3. Definizione di completamento

Il progetto è concluso quando soddisfa tutti questi requisiti:

- Elabora almeno quattro stream video con inferenza GPU batchata.
- Usa clip realistiche e differenti per la demo funzionale.
- Esegue tracking multi-oggetto indipendente per ogni camera.
- Conta veicoli per classe e direzione tramite line-crossing configurabile.
- Genera dati strutturati in CSV o SQLite.
- Registra metriche prestazionali e produce benchmark comparativi.
- Produce almeno un video demo annotato.
- Include README in inglese, configurazione riproducibile, diagramma architetturale e istruzioni di esecuzione.
- Documenta chiaramente limiti, assunzioni e risultati ottenuti.

## 4. Architettura target

```text
Video source per camera
        │
        ▼
Frame capture workers
        │
        ▼
Bounded per-camera input queues
        │
        ▼
Dynamic batcher
        │
        ▼
ONNX FP16 GPU inference
        │
        ▼
Detection results routed by camera_id
        │
        ├──────────────► Per-camera multi-object tracker
        │                         │
        │                         ▼
        │                  Traffic analytics engine
        │                  - ROI filtering
        │                  - line crossing
        │                  - directional counts
        │                  - stopped-vehicle events
        │                         │
        ▼                         ▼
Annotated video output       CSV / SQLite / JSON events
        │                         │
        └──────────────► Metrics collector and benchmark report
```

Principio essenziale: il batching può essere condiviso tra stream, ma tracking, stato e contatori devono restare isolati per `camera_id`.

## 5. Fase 1 — Stabilizzazione e misurazione della pipeline esistente

### Obiettivo

Rendere affidabili e confrontabili le prestazioni già ottenute.

### Attività

- Definire una configurazione centralizzata in YAML o JSON:
  - modello ONNX;
  - provider di inferenza;
  - dimensione immagine;
  - soglia confidence;
  - soglia IoU;
  - dimensione massima delle code;
  - batch size;
  - politica di frame dropping;
  - elenco stream e relativo `camera_id`.

- Registrare metriche per ogni camera e globali:
  - frame catturati;
  - frame inferiti;
  - frame scartati;
  - FPS per camera;
  - FPS globale;
  - latenza end-to-end p50 e p95;
  - tempo di inferenza medio/p95;
  - tempo medio in coda;
  - dimensione media/massima di ogni coda;
  - VRAM occupata, se disponibile.

- Gestire correttamente:
  - fine file video;
  - errore di lettura stream;
  - shutdown pulito di thread e code;
  - saturazione della coda;
  - frame obsoleti: privilegiare la bassa latenza rispetto all’elaborazione di ogni frame.

### Criteri di accettazione

- Il sistema termina senza thread bloccati.
- Le metriche vengono salvate in un file strutturato.
- Le definizioni di FPS e latenza sono esplicite nel codice e nel README.
- I risultati 51 FPS sequenziale e 88 FPS concorrente sono riproducibili sullo stesso setup.

## 6. Fase 2 — Dataset e scenari di test

### Obiettivo

Sostituire la sola demo con clip replicate con scenari realistici e riproducibili.

### Attività

- Usare **UA-DETRAC** come dataset primario per demo e validazione del tracking.
- Selezionare quattro clip diverse:
  - traffico basso;
  - traffico denso;
  - occlusioni;
  - meteo o illuminazione più difficile.

- Creare un file di configurazione per ogni telecamera:
  - nome e percorso della clip;
  - `camera_id`;
  - ROI della carreggiata;
  - linee di conteggio;
  - direzione attesa per ogni linea;
  - classi da includere;
  - eventuale soglia per veicolo fermo.

- Non caricare i video nel repository.
- Aggiungere uno script o istruzioni per download e preparazione del dataset.

### Criteri di accettazione

- La demo funzionale usa quattro clip diverse.
- Il benchmark di scalabilità conserva quattro repliche della stessa clip, etichettate come `simulated_streams`.
- Ogni camera ha una configurazione indipendente e caricabile senza modifiche al codice.

## 7. Fase 3 — Tracking multi-oggetto per camera

### Obiettivo

Assegnare un identificatore temporaneamente persistente a ogni veicolo rilevato.

### Attività

- Integrare ByteTrack o BoT-SORT.
- Creare un’istanza tracker separata per ogni `camera_id`.
- Passare al tracker solo detection filtrate:
  - classi veicolo desiderate;
  - confidence threshold;
  - ROI opzionale.

- Rendere visibile nell’overlay:
  - bounding box;
  - classe;
  - confidence opzionale;
  - `camera_id`;
  - `track_id`.

- Salvare per ogni traccia:
  - timestamp;
  - frame id;
  - bounding box;
  - centro bounding box;
  - velocità in pixel/frame, se calcolata;
  - stato attivo/perso/terminato.

### Criteri di accettazione

- Un veicolo mantiene normalmente lo stesso `track_id` durante il transito.
- Gli ID non collidono logicamente fra camere: usare una chiave composta `camera_id:track_id`.
- Il sistema resta in real time oppure documenta chiaramente il costo prestazionale del tracker.
- Il tracking è verificato visivamente su almeno due clip con occlusioni.

## 8. Fase 4 — ROI, conteggio per corsia e direzione

### Obiettivo

Convertire le tracce in metriche di traffico utili.

### Attività

- Implementare filtro ROI poligonale:
  - ignorare detection o tracce fuori dalla sede stradale;
  - configurare coordinate normalizzate oppure coordinate riferite alla risoluzione.

- Implementare line crossing:
  - definire ogni linea con due punti;
  - memorizzare il lato precedente della linea per ogni track;
  - registrare un attraversamento solo quando il centro della traccia cambia lato;
  - evitare doppio conteggio con stato per `track_id + line_id`.

- Associare ogni linea a:
  - nome;
  - direzione;
  - eventuale corsia;
  - classi abilitate.

- Produrre contatori:
  - per camera;
  - per linea;
  - per direzione;
  - per classe veicolo;
  - per intervallo temporale.

### Output atteso

Esempio di evento:

```json
{
  "timestamp_ms": 12540,
  "camera_id": "cam_02",
  "track_id": 17,
  "event_type": "line_crossing",
  "line_id": "northbound_lane_1",
  "direction": "northbound",
  "vehicle_class": "car"
}
```

### Criteri di accettazione

- Lo stesso veicolo viene contato una sola volta per linea.
- Il conteggio è corretto in entrambe le direzioni su una scena testata manualmente.
- I risultati sono esportati in CSV o SQLite.
- L’overlay mostra linee, direzione e contatori aggiornati.

## 9. Fase 5 — Eventi di traffico semplici

### Obiettivo

Aggiungere una componente analitica senza espandere eccessivamente lo scope.

### Eventi da implementare

Implementare almeno uno dei seguenti:

1. **Stopped vehicle**
   - Un track resta nella ROI per almeno `N` secondi.
   - La velocità stimata resta sotto una soglia.
   - Il veicolo non deve essere semplicemente fermo a un semaforo: documentare questo limite.

2. **Congestion level**
   - Stimare densità come numero di veicoli attivi nella ROI.
   - Definire soglie `low`, `medium`, `high`.
   - Salvare cambi di stato con timestamp.

3. **Queue length estimate**
   - Contare veicoli in una specifica zona di attesa.
   - Presentare il dato come stima basata su bounding box, non come misura fisica precisa.

### Criteri di accettazione

- Almeno un evento viene generato e scritto nei log.
- La logica dell’evento è configurabile.
- I falsi positivi osservati vengono documentati nel README.

## 10. Fase 6 — Output e presentazione

### Obiettivo

Rendere il sistema concretamente utilizzabile e facilmente dimostrabile.

### Attività

- Generare video annotati per ogni stream:
  - box;
  - `track_id`;
  - linee/ROI;
  - contatori;
  - FPS;
  - alert o stato congestione.

- Esportare:
  - `metrics.csv`;
  - `events.csv`;
  - `counts.csv`;
  - opzionalmente un database SQLite.

- Creare una vista minimale:
  - opzionale Streamlit;
  - oppure report HTML statico;
  - oppure una schermata composita con quattro stream e metriche aggregate.

Non investire tempo in autenticazione, frontend complesso o dashboard enterprise.

### Criteri di accettazione

- Esiste un video demo di 60–90 secondi.
- I file dati possono essere aperti e analizzati senza eseguire il progetto.
- Un osservatore capisce il funzionamento del sistema senza leggere il codice.

## 11. Fase 7 — Benchmark e analisi

### Obiettivo

Dimostrare i trade-off di systems engineering, non solo mostrare un video.

### Esperimenti obbligatori

Eseguire almeno questi test sullo stesso hardware, risoluzione e dataset:

| ID | Configurazione | Domanda a cui risponde |
|---|---|---|
| B1 | Sequenziale, batch 1 | Qual è la baseline? |
| B2 | Pipeline concorrente, batch 1 | Quanto aiuta il parallelismo tra I/O e inferenza? |
| B3 | Pipeline concorrente, batch 4 | Quanto aiuta il batching dinamico? |
| B4 | 1, 2 e 4 stream simulati | Come scala il throughput? |
| B5 | 4 clip diverse | La pipeline resta stabile su scenari realistici? |

Se TensorRT viene effettivamente integrato, aggiungere:

| ID | Configurazione | Domanda a cui risponde |
|---|---|---|
| B6 | ONNX Runtime FP16 vs TensorRT FP16 | Qual è il guadagno reale del deployment engine? |

### Grafici richiesti

- FPS globale per configurazione.
- FPS per camera.
- Latenza p50 e p95.
- Frame drop rate.
- Tempo medio per stadio: capture, queue, inference, tracking, rendering.
- VRAM, se misurabile.

### Criteri di accettazione

- Tutti gli esperimenti sono lanciabili con un comando o script documentato.
- I grafici sono generati da dati salvati, non inseriti manualmente.
- Il report spiega almeno un collo di bottiglia e un trade-off osservato.
- Il claim nel README è accurato: ad esempio, `88 aggregate FPS on four replicated streams`, non `88 FPS on four independent live cameras`.

## 12. Fase 8 — Validazione funzionale

### Obiettivo

Fornire una misura onesta della qualità del sistema di analytics.

### Attività

- Selezionare due intervalli da 30–60 secondi di due clip diverse.
- Contare manualmente i veicoli che attraversano ogni linea configurata.
- Confrontare conteggio manuale e conteggio della pipeline.
- Calcolare:

```text
count_error_percent = abs(predicted_count - manual_count) / manual_count * 100
```

- Annotare errori osservati:
  - occlusione;
  - ID switch;
  - detection mancata;
  - veicolo fuori ROI;
  - attraversamento ambiguo;
  - duplicazione dovuta al tracker.

### Criteri di accettazione

- Il README riporta almeno quattro esempi di confronto manuale vs pipeline.
- I limiti sono esplicitamente dichiarati.
- Non vengono riportate metriche di accuratezza non realmente calcolate.

## 13. Struttura consigliata del repository

```text
traffic-analytics/
├── README.md
├── LICENSE
├── requirements.txt
├── pyproject.toml
├── configs/
│   ├── default.yaml
│   ├── benchmark_replicated_streams.yaml
│   └── cameras/
│       ├── cam_01.yaml
│       ├── cam_02.yaml
│       ├── cam_03.yaml
│       └── cam_04.yaml
├── src/
│   ├── main.py
│   ├── capture/
│   ├── pipeline/
│   ├── inference/
│   ├── tracking/
│   ├── analytics/
│   ├── output/
│   └── metrics/
├── scripts/
│   ├── prepare_data.*
│   ├── run_benchmark.*
│   └── generate_report.*
├── tests/
│   ├── test_line_crossing.*
│   ├── test_counting.*
│   └── test_shutdown.*
├── docs/
│   ├── architecture.md
│   ├── benchmark_results.md
│   └── images/
└── outputs/
    └── .gitkeep
```

Non committare:

- video;
- pesi del modello;
- engine TensorRT;
- file di output voluminosi;
- percorsi locali;
- credenziali o feed RTSP reali.

## 14. README finale: contenuti obbligatori

Scrivere tutto in inglese e includere:

1. One-sentence project summary.
2. GIF o screenshot della demo.
3. Architettura della pipeline.
4. Funzionalità:
   - multi-stream ingestion;
   - bounded queues e backpressure;
   - dynamic batching;
   - ONNX FP16 GPU inference;
   - per-camera tracking;
   - directional counting;
   - event logging.
5. Requisiti hardware/software.
6. Quickstart.
7. Configurazione di una nuova camera.
8. Dataset e licenze.
9. Tabella benchmark.
10. Risultati validazione conteggi.
11. Limitazioni e future work.
12. Riproducibilità.

## 15. Limitazioni da dichiarare esplicitamente

- I quattro stream replicati servono solo a testare la scalabilità della pipeline.
- Il tracking è per-camera, non cross-camera re-identification.
- Il conteggio si basa sul centro del bounding box e può fallire in presenza di occlusioni o errori di tracking.
- La velocità è in pixel/frame salvo calibrazione prospettica della camera.
- Le prestazioni dipendono da risoluzione, codec, I/O, modello, runtime e GPU.
- Il sistema è una demo di traffic analytics, non un sistema certificato per controllo del traffico o videosorveglianza.

## 16. Scope da non aggiungere prima della pubblicazione

Non implementare ora:

- cross-camera vehicle re-identification;
- riconoscimento targhe;
- riconoscimento facciale;
- dashboard enterprise;
- training o fine-tuning da zero;
- Kubernetes, microservizi o cloud deployment;
- action recognition complessa;
- stima di velocità in km/h senza calibrazione.

Questi punti possono essere indicati come future work.

## 17. Checklist di chiusura

- [ ] La pipeline concorrente è stabile e misurata.
- [ ] Quattro clip differenti funzionano nella demo.
- [ ] Il tracking per camera è integrato.
- [ ] ROI e line crossing sono configurabili.
- [ ] I contatori per direzione/classe funzionano.
- [ ] Almeno un evento di traffico è registrato.
- [ ] CSV/SQLite e video annotati vengono generati.
- [ ] Benchmark B1–B5 completati e graficati.
- [ ] Due clip sono validate con conteggio manuale.
- [ ] README inglese completo.
- [ ] Video demo breve disponibile.
- [ ] Repository pulito, riproducibile e senza file pesanti.

## 18. Bullet finale per il CV

Usare solo dopo aver verificato i numeri finali:

> Developed a backpressure-aware multi-camera traffic analytics pipeline with ONNX FP16 GPU inference, dynamic batching, per-camera multi-object tracking and directional vehicle counting. Achieved 88 aggregate FPS across four replicated streams on an NVIDIA RTX 3060, improving throughput by 73% over the sequential baseline; validated event and counting outputs on real traffic video.

## 19. Future work da indicare nel repository

- TensorRT FP16 deployment and performance comparison.
- Camera calibration for speed estimation in km/h.
- Cross-camera vehicle re-identification using CityFlow / AI City Challenge data.
- Adaptive batching based on queue pressure and latency target.
- Tracking-quality evaluation using MOT metrics.
- Additional traffic anomalies and longer-term congestion analytics.