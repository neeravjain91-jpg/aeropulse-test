# AeroPulse-X AI / ML / DL / LLM Executive Cheat Sheet

> **Document Status**: Complete Engineering Reference Cheat Sheet  
> **Repository**: `neeravjain91-jpg/aeropulse-test`  
> **Target Framework**: AeroPulse-X Edge Propulsion Health & Digital Twin Architecture  
> **Audience**: Systems Engineers, ML Researchers, Flight Software Certifiers, Flight Line Technicians

---

## 1. AeroPulse-X Model Inventory & Deployment Status

| Model Name | Architectural Framework | Input Representation | Model Size | CPU Latency | Primary Metric | Implementation Status | Artifact Path |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **ACES Health State Classifier** | HistGradientBoosting (Scikit-Learn) | 15 Tabular Engineering Features ($t$) | ~964 KB | ~0.10 ms | Weighted F1: **89.19%**, Acc: 89.12% | `CURRENTLY USED` (Primary Production) | `models/aces_health.joblib` |
| **ACES Anomaly Detector** | Isolation Forest (Scikit-Learn) | 15 Tabular Engineering Features ($t$) | ~358 KB | ~0.15 ms | AUROC: **0.8725**, AUPRC: 0.7102 | `CURRENTLY USED` (Primary Unsupervised) | `models/aces_anomaly.joblib` |
| **Physics-Residual TCN** | 3-Block Causal Dilated Conv1D (PyTorch) | 13 Residual Channels $\times$ 30s Window | 88.9 KB (.pt) / 142.7 KB (.ts) | 0.811 ms (17,764 params) | Weighted F1: **88.44%**, Acc: 88.17% | `BENCHMARKED` (Shadow Mode Candidate) | `models/aces_tcn_residual.pt`, `.ts` |
| **Temporal TCN Autoencoder** | Causal Conv1D Bottleneck ($13\to32\to16\to8\to16\to32\to13$) | 13 Residual Channels $\times$ 30s Window | 40.0 KB (.pt) / 73.2 KB (.ts) | 0.506 ms (6,661 params) | AUROC: **0.9683**, AUPRC: 0.8677 | `BENCHMARKED` (Ensemble Shadow Mode) | `models/aces_tcn_autoencoder.pt`, `.ts` |
| **Hybrid Anomaly Ensemble** | Logical OR / Probabilistic Fusion | Feature Vector + 30s Window | ~400 KB | ~0.65 ms | AUROC: **0.9598**, F1: **69.04%**, FAR: **5.79%** | `BENCHMARKED` (Optimal Boundary) | `app/fusion.py` |
| **Reference Physics Twin** | First-Principles Deterministic Solver | RPM, MAP, Ambient ($t$) | Static Code | < 0.05 ms | Residual Baseline $\Delta = 0$ | `CURRENTLY USED` (Authoritative Truth) | `app/digital_twin.py` |
| **C-MAPSS Degradation Model** | RandomForest / GradientBoosting | 14 Turbofan Sensors (Piecewise Linear) | ~126 MB | ~1.20 ms | Test RMSE: **18.42 cycles**, Score: 412.8 | `METHODOLOGY ONLY` (Degradation Transfer) | `models/cmapss_rul_method.joblib` |
| **RUL Mission Extrapolator** | Weibull Hazard + Physics Slope Extrapolation | Dynamic Health Index Time Trend | Static Code | < 0.02 ms | Calibrated Prognostic Horizon | `CURRENTLY USED` (Production RUL) | `app/rul_engine.py` |
| **LLM Mission Report Service** | Provider-Agnostic Client (Gemini / GPT / REST) | Structured JSON Mission Evidence | Remote / Fallback | Remote (1-3s) / Deterministic (4ms) | 100% Numerical Grounding Pass | `IMPLEMENTED WITH FALLBACK` | `app/llm_report_service.py` |
| **NLP Maintenance Parser** | Deterministic Regex & Ontology Matcher | Raw Text Technician Notes | Static Code | < 0.01 ms | Regex Semantic Entity F1: 94.2% | `CURRENTLY USED` (Rule-Based NLP) | `app/nlp_maintenance.py` |

---

## 2. Core Mathematical Formulas

### 2.1 Deep Learning & Convolutional Mathematics

#### Artificial Neuron & Forward Propagation
$$\mathbf{z} = \mathbf{W}\mathbf{x} + \mathbf{b}, \quad \mathbf{a} = \sigma(\mathbf{z})$$

#### Activation Functions
* **ReLU (Rectified Linear Unit)**:
  $$\text{ReLU}(z) = \max(0, z), \quad \frac{d}{dz}\text{ReLU}(z) = \begin{cases} 1 & \text{if } z > 0 \\ 0 & \text{if } z < 0 \end{cases}$$
* **Sigmoid**:
  $$\sigma(z) = \frac{1}{1 + e^{-z}}, \quad \sigma'(z) = \sigma(z)(1 - \sigma(z))$$
* **Hyperbolic Tangent (Tanh)**:
  $$\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}, \quad \tanh'(z) = 1 - \tanh^2(z)$$
* **Softmax (Multi-class Probabilities)**:
  $$\text{Softmax}(\mathbf{z})_i = \frac{e^{z_i}}{\sum_{j=1}^C e^{z_j}}$$

#### Loss Functions
* **Multi-Class Cross-Entropy Loss (with Class Weights $w_c$)**:
  $$\mathcal{L}_{\text{CE}} = - \sum_{c=1}^C w_c \, y_c \log(\hat{y}_c)$$
* **Mean Squared Error (Autoencoder Reconstruction Loss)**:
  $$\mathcal{L}_{\text{MSE}} = \frac{1}{B \cdot C \cdot W} \sum_{b=1}^B \sum_{c=1}^C \sum_{t=1}^W (x_{b,c,t} - \hat{x}_{b,c,t})^2$$

#### Causal Convolution & Receptive Field
* **1D Causal Convolution with Dilation $d$**:
  $$y[t] = (x *_d k)[t] = \sum_{i=0}^{K-1} k[i] \cdot x[t - d \cdot i]$$
* **Exact Temporal Receptive Field ($RF$)**:
  $$RF = 1 + \sum_{l=1}^L (K_l - 1) \cdot d_l$$
  * *AeroPulse TCN Specification*: $L=3$ blocks (each with 2 conv layers), $K=3$, dilations $d \in \{1, 2, 4\}$:
    $$RF = 1 + 2 \cdot (3 - 1) \cdot (1 + 2 + 4) = 1 + 4 \times 7 = 29 \text{ seconds}$$

#### AdamW Optimizer Update Rule
$$\mathbf{m}_t = \beta_1 \mathbf{m}_{t-1} + (1 - \beta_1) \nabla_\theta \mathcal{L}_t$$
$$\mathbf{v}_t = \beta_2 \mathbf{v}_{t-1} + (1 - \beta_2) (\nabla_\theta \mathcal{L}_t)^2$$
$$\hat{\mathbf{m}}_t = \frac{\mathbf{m}_t}{1 - \beta_1^t}, \quad \hat{\mathbf{v}}_t = \frac{\mathbf{v}_t}{1 - \beta_2^t}$$
$$\theta_t = \theta_{t-1} - \eta \left( \frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon} + \lambda \theta_{t-1} \right)$$

---

### 2.2 Transformer & LLM Mathematics

#### Scaled Dot-Product Attention
$$\mathbf{Q} = \mathbf{X}\mathbf{W}_Q, \quad \mathbf{K} = \mathbf{X}\mathbf{W}_K, \quad \mathbf{V} = \mathbf{X}\mathbf{W}_V$$
$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left( \frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}} + \mathbf{M} \right) \mathbf{V}$$
* Where $\mathbf{M}$ is the causal attention mask ($M_{ij} = -\infty$ for $j > i$ to prevent future token lookahead), and $\sqrt{d_k}$ prevents gradient saturation in high-dimensional dot products.

#### Multi-Head Attention (MHA)
$$\text{MultiHead}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) \mathbf{W}_O$$
$$\text{where } \text{head}_i = \text{Attention}(\mathbf{Q}\mathbf{W}_i^Q, \mathbf{K}\mathbf{W}_i^K, \mathbf{V}\mathbf{W}_i^V)$$

#### Vector Cosine Similarity (Embeddings & RAG)
$$\text{sim}(\mathbf{u}, \mathbf{v}) = \cos(\theta) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\| \|\mathbf{v}\|} = \frac{\sum_{i=1}^d u_i v_i}{\sqrt{\sum_{i=1}^d u_i^2} \sqrt{\sum_{i=1}^d v_i^2}}$$

#### Temperature-Scaled Softmax Sampling
$$P(w_i | w_{<t}) = \frac{\exp(z_i / T)}{\sum_{j} \exp(z_j / T)}$$
* $T \to 0$: Greedy argmax (deterministic).
* $T = 1.0$: Standard categorical distribution.
* $T > 1.0$: Higher entropy, flatter distribution.

---

### 2.3 Statistical Calibration & Reliability Metrics

#### Expected Calibration Error (ECE)
$$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$
* AeroPulse Benchmark: HGB $\text{ECE} = 0.0230$, TCN $\text{ECE} = 0.0416$.

#### Multi-Class Brier Score
$$\text{Brier} = \frac{1}{N} \sum_{n=1}^N \sum_{c=1}^C (p_{n,c} - y_{n,c})^2$$
* AeroPulse Benchmark: HGB $\text{Brier} = 0.1562$, TCN $\text{Brier} = 0.1800$.

#### Weibull Cumulative Hazard & Survival Functions (RUL)
$$S(t) = \exp\left( -\left( \frac{t}{\eta} \right)^\beta \right), \quad h(t) = \frac{\beta}{\eta} \left( \frac{t}{\eta} \right)^{\beta - 1}$$
* $\beta < 1$: Infant mortality (burn-in).
* $\beta = 1$: Constant random failure rate (exponential distribution).
* $\beta > 1$: Wear-out degradation regime (AeroPulse piston engine baseline: $\beta = 2.4$).

---

## 3. High-Priority Architectural Glossary

| Term | Definition & AeroPulse Context | Strict Status |
| :--- | :--- | :--- |
| **Physics-Normalized Residual** | Telemetry feature normalized against first-principles thermodynamic Reference Twin ($r_i = x_{i,\text{meas}} - x_{i,\text{twin}}$). Isolates faults from operating condition shifts. | `CURRENTLY USED` |
| **GroupShuffleSplit** | Split methodology grouping all rows by `Flight` ID. Guarantees 0% temporal or trajectory leakage across train/validation/test sets. | `CURRENTLY USED` |
| **Causal Convolution** | 1D convolution where filter at timestep $t$ depends strictly on samples $\le t$, enforced via $(K-1) \cdot d$ left-padding. | `CURRENTLY IMPLEMENTED` |
| **Dilation Rate ($d$)** | Spacing between kernel taps. Exponential dilation ($1, 2, 4, \dots$) expands receptive field exponentially without adding parameters. | `CURRENTLY IMPLEMENTED` |
| **Reconstruction Threshold ($\tau$)** | Anomaly decision boundary calibrated at 98th percentile of nominal flight MSE ($\tau = 0.66747$). Residual MSE $> \tau$ flags anomaly. | `CURRENTLY IMPLEMENTED` |
| **Tokenization (BPE)** | Byte-Pair Encoding subword segmentation mapping arbitrary text into integer indices within a fixed vocabulary. | `TEXTBOOK / FUTURE` |
| **RAG** | Retrieval-Augmented Generation: Grounding LLM prompts with deterministically retrieved chunks from maintenance manuals and flight databases. | `FUTURE PROPOSAL` |
| **Anti-Hallucination Guard** | Deterministic regex/AST post-generation validation verifying every numerical claim against authoritative MissionSummary JSON. | `CURRENTLY IMPLEMENTED` |
| **Safety Interlock** | Deterministic flight software mechanism preventing any statistical ML, DL, or LLM from directly actuating engine controls or overriding flight envelopes. | `CURRENTLY USED` |

---

## 4. AeroPulse-X Dataflow & Safety Boundary

```text
[ Raw Aircraft CAN Telemetry (1 Hz) ]
                 │
                 ▼
[ Deterministic Ingestion & Sanity Checks ] (Range, NaN, Dropout)
                 │
                 ├───────────────────────────────────────────────────────┐
                 ▼                                                       ▼
[ Reference Physics Twin (app/digital_twin.py) ]          [ Sensor Isolation (app/sensor_isolation.py) ]
  (1st-principles thermodynamic baseline)                   (Residual Z-score cross-correlation)
                 │                                                       │
                 ▼                                                       │
[ 13 Physics-Normalized Residuals ]                                     │
                 │                                                       │
                 ├───────────────────────────────┐                       │
                 ▼                               ▼                       ▼
   [ HistGradientBoosting (ACES) ]   [ Causal TCN (Shadow) ]   [ Isolation Forest & TCN-AE ]
   (models/aces_health.joblib)       (models/aces_tcn_residual) (Unsupervised Novelty)
                 │                               │                       │
                 └───────────────────────────────┴───────────────────────┘
                                                 │
                                                 ▼
                              [ Multi-Source Diagnostic Arbiter ]
                              (Conservative Worst-Case Safety State)
                                                 │
                                                 ▼
                              [ RUL Engine (app/rul_engine.py) ]
                              (Weibull Hazard + Degradation Trend)
                                                 │
                                                 ▼
                        [ Structured Mission Intelligence JSON ]
                                                 │
                         ┌───────────────────────┴───────────────────────┐
                         ▼                                               ▼
         [ Deterministic Report Generator ]             [ Grounded LLM Report Service ]
         (100% offline fallback)                        (app/llm_report_service.py)
                         │                                               │
                         │                                               ▼
                         │                                [ Post-Gen Numerical Claim Audit ]
                         │                                (validate_report_claims())
                         │                                               │
                         ├───────────────────────────────────────────────┘
                         ▼
        [ Flight Line Technician / Mission Commander UI ]
```

---

## 5. Source Discipline Compliance Key

* **`CURRENTLY USED`**: Active in primary production execution path (`app/` runtime, `aces_health.joblib`, `aces_anomaly.joblib`).
* **`CURRENTLY IMPLEMENTED`**: Fully written and callable in repository codebase (`app/tcn_model.py`, `app/anomaly_autoencoder.py`, `app/llm_report_service.py`), runs in shadow mode or with deterministic fallback.
* **`BENCHMARKED`**: Systematically evaluated with full test metrics in `models/*_metrics.json` and verification scripts.
* **`METHODOLOGY ONLY`**: Evaluated on external surrogate datasets (e.g. C-MAPSS turbofan, CWRU bearing) for research transferability.
* **`FUTURE PROPOSAL`**: Conceptually designed for future avionics upgrades (e.g. Vector DB RAG, Onboard Multimodal Diagnostics), zero production code present.
* **`NOT PRESENT`**: Architecture or library explicitly audited and absent from the current codebase (e.g. LangChain, LlamaIndex, FAISS, On-device LLM weights).
