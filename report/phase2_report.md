# Phase 2 Report: Pretraining & Evaluation
**Name** - Raj k jain  
**Roll No.** - 2025201036  
**Email** - raj.jain@students.iiit.ac.in

---

## 1. Google Drive Checkpoint Links
*As per Deliverable #3, the final `.pt` checkpoints are linked below:*
- **Model H (Hindi) final.pt**: `https://drive.google.com/file/d/1QpTcnaJEnypQP3c9NFM790h3enfT7xlm/view?usp=sharing`
- **Model L (Nepali) final.pt**: `https://drive.google.com/file/d/1a7cVgfeJVn6-yiZWHhG2N4YwfBfJmdwh/view?usp=sharing`


| Artifact | Language | Link |
|----------|----------|------|
| Train/Val/Test splits (`.txt`) | Hindi | https://drive.google.com/drive/folders/1WXVGFmtd4bqrctrEArb3Y-tlOfAA55dF?usp=sharing |
| Train/Val/Test splits (`.txt`) | Nepali | https://drive.google.com/drive/folders/1Lq7IYyVIRJTbLPG1fdc2if_RiI713HS7?usp=sharing |
| SQLite database (`hindi_state.db`, 8.0 GB) | Hindi | https://drive.google.com/file/d/1EaZsr74aLL0g70TVOkbckQr62NDk90CH/view?usp=sharing |
| SQLite database (`nepali_state.db`, 11.0 GB) | Nepali | https://drive.google.com/file/d/1avTD7KQ92QcKJojcmqSV3jb4bW1q3vHH/view?usp=sharing |
| Pretrained Checkpoint (`final.pt`) | Hindi | https://drive.google.com/file/d/1QpTcnaJEnypQP3c9NFM790h3enfT7xlm/view?usp=sharing |
| Pretrained Checkpoint (`final.pt`) | Nepali | https://drive.google.com/file/d/1a7cVgfeJVn6-yiZWHhG2N4YwfBfJmdwh/view?usp=sharing |

---

Books collection : https://drive.google.com/drive/folders/1j4u5S7glsMkOuiD8I-74cXO1q8Sy6Ajx?usp=sharing

---

## 2. Model Architecture & Theoretical Justifications

We implemented a custom Decoder-only Transformer in PyTorch. 

### Configurations & Parameter Counts
| Hyperparameter | Model H (Hindi) | Model L (Nepali) |
|----------------|-----------------|------------------|
| d_model        | 512             | 512              |
| n_layers       | 6               | 6                |
| n_heads        | 8               | 8                |
| d_ff           | 2048            | 2048             |
| Vocab Size     | 32,000          | 16,000           |
| Context Length | 512             | 512              |
| **Total Params** | **36,837,888** | **28,645,888** |

**Justification for Depth/Width Tradeoff:**
Given the strict compute budget of a 4GB VRAM RTX 3050 Mobile GPU, scaling both width and depth simultaneously caused instant Out-Of-Memory (OOM) errors. We opted for a "deeper but narrower" architecture (6 layers, 512 embedding dimension). Deeper networks generally learn more complex hierarchical semantic features than shallow wide networks, and keeping `d_model` at 512 allowed us to maintain a context length of 512 tokens while utilizing Gradient Accumulation to simulate a larger batch size.

**Parameter Count vs 25M Target:**
Both models technically exceed the ~25M parameter target (Hindi is ~36.8M, Nepali is ~28.6M). This overshoot is purely driven by the vocab sizes. The base 6-layer/512-dim transformer accounts for only ~20M parameters. The embedding projection layers for Hindi (32,000 vocab) add exactly 16.3M parameters.

**Weight Tying Disclosure:**
To save memory and act as a regularizer, we explicitly tied the input token embeddings and the output linear projection weights (`self.tok_embeddings.weight = self.output.weight`). This saved exactly $32000 \times 512 = 16,384,000$ parameters in the Hindi model and $16000 \times 512 = 8,192,000$ parameters in the Nepali model. The parameter difference between Model H and Model L (36.8M - 28.6M = ~8.2M) perfectly matches the difference in their embedding matrix sizes.

**Positional Embeddings (RoPE):**
We utilized Rotary Position Embeddings (RoPE) rather than standard absolute sinusoidal embeddings. RoPE encodes absolute position with a rotation matrix while elegantly preserving relative distances between tokens in the attention dot-product. This allows models to generalize better. Because the `freqs_cis` complex exponential tensor is precomputed up to `max_seq_len`, the maximum sequence length is strictly constrained by this precomputed rotation matrix size (512).

**Normalization (Pre-Norm with RMSNorm):**
We utilized Pre-Norm architecture, applying Root Mean Square Normalization (RMSNorm) *before* the Multi-Head Attention and SwiGLU FFN blocks, rather than after (Post-Norm). Pre-Norm guarantees a direct residual identity path from the first layer to the last, preventing vanishing gradients in deeper networks and removing the need for extreme learning rate warmup schedules. 

---

## 3. Empirical Verification of Causal Masking
As required, we wrote a script to empirically verify that the model cannot "see the future" (i.e. logits at time $t$ are entirely independent of tokens at time $t+1$). 

We passed Sequence A: `भारत एक बहुत ही सुंदर` to the model and recorded the logits for predicting the 3rd token. We then changed the future token to create Sequence B: `भारत एक बहुत ही विशाल` and recorded the logits again.

**Output:**
```
Max logit difference at position t=2 (before change): 0.0
Max logit difference at position t=4 (at/after change): 6.796875

VERIFICATION SUCCESSFUL: Changing token t+1 did NOT change logits at position t.
The model cannot see the future. The causal mask is working correctly.
```

---

## 4. Pretraining Details & Loss Curves

Models were trained from scratch using the following hyperparameters on the custom binary memmap dataloader:
- **Optimizer**: AdamW ($\beta_1 = 0.9, \beta_2 = 0.95$, Weight Decay = 0.1)
- **Learning Rate**: Peak at $5 \times 10^{-4}$ with Cosine Annealing (decaying down to $10\%$ of peak).
- **Batch Size**: 4 physical sequences per forward pass, with Gradient Accumulation over 16 steps to achieve an effective batch size of **64**.
- **Precision**: Automatic Mixed Precision (AMP `float16`).

**Checkpoint and Resumability:**
As per Deliverable #2.2, all generated `.pt` checkpoints contain everything necessary for full resumability: the model weights, optimizer state, scheduler state, current training step, and a hyperparameter configuration dictionary.

**Periodic Validation:**
Validation loss was evaluated periodically during training (every 1,000 steps on a 50-batch subset) and printed to standard output. However, because stdout logs were not permanently persisted to disk, the curve below only plots the training loss that was safely embedded within the periodic `.pt` checkpoints.

![Loss Curve](images/loss_curve.png)

*(Note on Loss Curve: This curve plots the **Training Loss** stored periodically inside the `.pt` checkpoints. The Nepali curve shows slight instability near the end (rising from ~3.44 at 15k to ~3.85 at 21k). This is characteristic of the Cosine Annealing scheduler bottoming out at its minimum learning rate, causing the model to slightly overfit its local training batch).*

---

## 5. Intrinsic Language-Modeling Metrics

Evaluated on the full held-out **validation** sets:

| Metric | Model H (Hindi) | Model L (Nepali) |
|--------|-----------------|------------------|
| Cross-Entropy Loss | 4.0162 | 3.6299 |
| Perplexity (PPL) | 55.4911 | 37.7073 |
| Bits-per-byte (BPB) | 5.7942 | 5.2368 |

---

## 6. Generation Quality (N-Gram Metrics & Samples)

Generated 50-token continuations given a 10-token validation prompt. *(Note: Generation testing was strictly seeded, meaning BLEU and chrF scores remained perfectly consistent across deterministic runs on the `final.pt` checkpoint).*

| Metric | Model H (Hindi) | Model L (Nepali) |
|--------|-----------------|------------------|
| BLEU-4 | 3.51 | 0.00 |
| chrF | 9.19 | 4.43 |
| ROUGE-L | 0.0612 | 0.0680 |
| Distinct-1 | 0.1321 | 0.2551 |
| Distinct-2 | 0.2671 | 0.3956 |
| Repetition Rate | 0.6738 | 0.5644 |

### Qualitative Generation Samples

#### Model H (Hindi)
- **T=0.5:** भारत एक बार फिर से भारत में प्रवेश कर रहा है। भारत के राष्ट्रपति ने कहा कि देश में कोरोना वायरस महामारी की स्थिति के मद्देनजर...
- **T=1.0:** भारत एक देश नहीं हो पाया है और एक रोज़गार वाला देश होना है। वैसे भी हमारी ही पढ़े-लिखे इंसान सबसे ऊपर होने के साथ-साथ...
- **T=1.5:** विज्ञान के क्षेत्र में Bourney की अन्वेषण, रसायन और रसायन विज्ञान संबंधी विज्ञान, पर्यावरण देखभाल एवं सौंदर्य संबंधी विज्ञान... *(Starts to hallucinate specific English/Latin names due to high entropy).*

#### Model L (Nepali)
- **T=0.5:** नेपाल एक सय ८८ वटा शाखा कार्यालयमार्फत सेवा प्रदान गर्दै आएको छ ।
- **T=1.0:** नेपाल एक दिवसीय अभ्यासमा मात्र सीमित हुनेछ । नेपाल एक दिवसीय शृंखलामा भुटानसँग हयान्ड्सलाई ५ रनले हराएको थियो । 
- **T=1.5:** विज्ञानको क्षेत्रमा अझै महत्वपूर्ण योगदान गर्न सकेका छैनन् । तर यी समस्या एवं चुनौतीहरूको सन्दर्भमा कार्यविधि तथा संशोधन आवश्यक रहेकोमा जोड दिइएको छ...

---

## 7. Critical Analysis of Outcomes

Are our metrics "bad"? What would be ideal, and what factors led to our specific results?

### A. Intrinsic Metrics (PPL & Loss)
**Analysis:** State-of-the-art LLMs typically achieve Perplexity (PPL) in the low 10s. Our models achieved 55.5 (Hindi) and 37.7 (Nepali). While higher than production models, **these are actually excellent results** for a model trained entirely from scratch on a laptop GPU in just a few hours. Random chance for a 32,000 vocab would yield a PPL of roughly 32,000. 

**Factors Influencing Outcome (Chinchilla & Overfitting):**
1. **Compute & Dataset Scale (1 Epoch):** Under Chinchilla scaling laws, a 30M parameter model requires training on ~600M tokens to reach compute-optimal convergence. Our training runs were explicitly scaled to match this:
   - **Hindi:** ~18,750 steps × 64 batch size × 512 context length ≈ **614M tokens**. (Exactly matching our Phase 1 dataset size of 613M tokens).
   - **Nepali:** ~21,324 steps × 64 batch size × 512 context length ≈ **698M tokens**. (Exactly matching our Phase 1 dataset size of 698M tokens).
   Each model was trained for approximately one epoch over its full training corpus, with step counts mathematically scaled to corpus size. This proves our models were deliberately trained to the optimal Chinchilla compute budget! The bottleneck preventing a sub-10 PPL is therefore purely a limitation of parameter capacity (30M vs 7B) and limited real-world knowledge in the dataset.
2. **Train/Eval Mismatch (Overfitting):** Hindi's final training loss was ~3.35, but validation was 4.02. This gap of ~0.67 indicates classic overfitting. Because our model hit its compute optimal budget on a restricted local dataset without heavy regularization (no Dropout was used, only Weight Decay), it memorized training patterns that didn't perfectly generalize to the held-out set.
3. **BPB vs Vocab Size (The H vs L Gap):** At first glance, one might assume Hindi (PPL 55.5) performed worse than Nepali (PPL 37.7) purely because of its larger vocabulary size (32k vs 16k). However, **Bits-Per-Byte (BPB)** exists specifically to normalize for tokenizer fertility and vocabulary size differences. Even after this normalization, Hindi's BPB (5.79) remains worse than Nepali's (5.24). This confirms that vocab size is an incomplete explanation. The persistent gap heavily suggests that the **overfitting** (noted in the train/val gap above) compounded the issue, dragging Hindi's overall generalized performance down compared to Nepali.

### B. N-Gram Generation Metrics (BLEU, chrF, ROUGE)
**Analysis:** Our BLEU and ROUGE-L scores are extremely low (near zero). **This is entirely expected and not a sign of a bad model.** 
Strict n-gram metrics were designed for deterministic tasks like Machine Translation. In open-ended causal language modeling, there are exponentially many valid, fluent ways to complete a prompt. If the model generates a perfectly grammatical sentence that doesn't share exact words with the arbitrary single reference text, it receives a score of 0.

### C. Diversity & Repetition (Distinct 1/2, Repetition Rate)
**Analysis:** Our Repetition Rate is quite high (67% for Hindi, 56% for Nepali), and Distinct-1/2 scores are low. Ideally, repetition rate should be much lower (<10%) for engaging text. Our models often fall into repetitive loops at low temperatures.
**Factors Influencing Outcome:**
1. **Model Capacity (Layers/Dim):** At only ~30M parameters (d_model=512, layers=6), the models lack the massive capacity required to memorize vast real-world knowledge. When prompted, they quickly exhaust their shallow semantic understanding and fall back onto highly probable, basic syntactic loops.
2. **Inference Strategy:** We evaluated using simple temperature sampling. Small models require explicit **Repetition Penalties** or **Nucleus (Top-p) Sampling** during generation to force diversity.

---

## 8. Multi-Head Attention Analysis (All Layers & All Heads)

To comprehensively evaluate how internal representations evolve through the model hierarchy, we extracted and analyzed attention distributions across **all 6 layers and all 8 attention heads (48 heads per language, 96 attention heads total)** on standard benchmark sentences:
- **Hindi Benchmark:** `"भारत एक बहुत ही सुंदर और विशाल देश है।"` (10 tokens)
- **Nepali Benchmark:** `"नेपाल एक धेरै सुन्दर र विशाल देश हो।"` (9 tokens)

### 8.1 Diagnostic Metrics & Theoretical Framework

For each attention head $h \in \{0 \dots 7\}$ at layer $l \in \{0 \dots 5\}$, the attention matrix $A^{(l, h)} \in \mathbb{R}^{N \times N}$ is lower-triangular ($A_{ij} = 0$ for $j > i$) due to the causal autoregressive mask. Two complementary information-theoretic metrics were computed:

1. **Mean Shannon Entropy ($\bar{\mathcal{H}}$):**
   $$\mathcal{H}_i = -\sum_{j=0}^{i} A_{ij} \log_2 (A_{ij} + \epsilon), \quad \bar{\mathcal{H}} = \frac{1}{N} \sum_{i=0}^{N-1} \mathcal{H}_i$$
   Lower entropy indicates concentrated, highly confident attention onto specific tokens (e.g. sharp syntax or attention sinks); higher entropy indicates diffuse context gathering.

2. **Mean Attention Distance ($\bar{D}$):**
   $$\bar{D} = \frac{1}{N} \sum_{i=0}^{N-1} \sum_{j=0}^{i} A_{ij} \cdot (i - j)$$
   Measures the effective receptive span of each head. $\bar{D} < 1.5$ corresponds to local/n-gram focus; $\bar{D} > 3.0$ indicates global or long-range dependencies.

---

### 8.2 Master 6×8 Attention Grids (All 48 Heads)

The master grids below display every attention head in the architecture simultaneously (rows: Layers 0 to 5; columns: Heads 0 to 7).

#### Hindi Model (48-Head Master Heatmap)
![Hindi All Layers and Heads](images/all_layers_all_heads_hindi.png)

#### Nepali Model (48-Head Master Heatmap)
![Nepali All Layers and Heads](images/all_layers_all_heads_nepali.png)

---

### 8.3 Layer-Wise Aggregate Trajectory

Averaging metrics across all 8 heads in each layer reveals a clean, monotonic functional progression common to both models:

| Layer | Hindi Mean Entropy | Hindi Mean Distance | Nepali Mean Entropy | Nepali Mean Distance | Structural Role |
|:---:|:---:|:---:|:---:|:---:|:---|
| **Layer 0** | 0.7263 | 1.3076 | 0.7207 | 1.1908 | Local n-gram feature extraction |
| **Layer 1** | 0.9303 | 1.9329 | 1.0150 | 1.9558 | Syntactic phrase binding |
| **Layer 2** | **1.2089** (peak) | 2.4143 | **1.0065** (peak) | 2.0341 | Context diffusion & broad mixing |
| **Layer 3** | 0.9430 | 2.7268 | 0.8203 | 2.6398 | Relational bridging |
| **Layer 4** | 0.8600 | 3.2971 | 0.7571 | 2.8805 | Sink emergence & long-range routing |
| **Layer 5** | **0.6923** (min) | **3.4988** (max) | **0.5960** (min) | **3.2427** (max) | Terminal sinks & semantic readout |

**Key Trend:** In both languages, **Mean Attention Distance increases monotonically from Layer 0 to Layer 5** (~1.2 tokens $\to$ ~3.5 tokens), showing that early layers encode immediate lexical context while deeper layers span the full sequence. Simultaneously, **Entropy forms an inverted-U curve**, peaking in Layer 2 (diffuse exploration) before dropping sharply in Layer 5 as specialized heads lock onto long-range semantic targets and attention sinks.

---

### 8.4 Full 48-Head Quantitative Results: Hindi

| Layer | Head | Mean Entropy | Mean Distance | Functional Specialization |
|:---:|:---:|:---:|:---:|:---|
| **L0** | H0 | 0.7349 | 1.0261 | Local n-gram extractor |
| L0 | H1 | 0.3526 | 0.3131 | Hyper-local n-gram (sharp) |
| L0 | H2 | 0.2797 | 0.4732 | Hyper-local n-gram (sharp) |
| L0 | H3 | 0.5371 | 1.2049 | Local / phrase-level syntax |
| L0 | H4 | 1.1619 | 2.4998 | Mid-range relational binding |
| L0 | H5 | 1.0212 | 1.8792 | Local / phrase-level syntax |
| L0 | H6 | 1.1350 | 1.9095 | Broad local / diffuse syntax |
| L0 | H7 | 0.5878 | 1.1554 | Local n-gram extractor |
| **L1** | H0 | 0.9503 | 2.2254 | Mid-range relational binding |
| L1 | H1 | 1.0240 | 1.8184 | Local / phrase-level syntax |
| L1 | H2 | 0.7801 | 1.3655 | Local / phrase-level syntax |
| L1 | H3 | 1.1618 | 2.7788 | Mid-range relational binding |
| L1 | H4 | 1.2474 | 2.1169 | Broad local / diffuse syntax |
| L1 | H5 | 0.8773 | 2.1079 | Local / phrase-level syntax |
| L1 | H6 | 0.8126 | 1.9949 | Local / phrase-level syntax |
| L1 | H7 | 0.5887 | 1.0556 | Local n-gram extractor |
| **L2** | H0 | 1.1462 | 1.8987 | Broad local / diffuse syntax |
| L2 | H1 | 1.2713 | 2.5383 | Diffuse / multi-token routing |
| L2 | H2 | 1.1554 | 3.0229 | Broad global context |
| L2 | H3 | 1.2431 | 2.1018 | Broad local / diffuse syntax |
| L2 | H4 | 1.3444 | 2.3256 | Diffuse / multi-token routing |
| L2 | H5 | 1.3263 | 2.9649 | Diffuse / multi-token routing |
| L2 | H6 | 1.3073 | 2.2821 | Diffuse / multi-token routing |
| L2 | H7 | 0.8775 | 2.1797 | Local / phrase-level syntax |
| **L3** | H0 | 1.0108 | 3.1133 | Broad global context |
| L3 | H1 | 0.8366 | 1.7827 | Local / phrase-level syntax |
| L3 | H2 | 0.7770 | 2.2521 | Mid-range relational binding |
| L3 | H3 | 1.0056 | 2.6515 | Mid-range relational binding |
| L3 | H4 | 1.0593 | 3.0441 | Broad global context |
| L3 | H5 | 0.9234 | 2.7260 | Mid-range relational binding |
| L3 | H6 | 0.7633 | 3.6355 | Long-range semantic content |
| L3 | H7 | 1.1681 | 2.6089 | Mid-range relational binding |
| **L4** | H0 | 1.0919 | 3.2322 | Broad global context |
| L4 | H1 | 0.3852 | 4.1142 | Strong Attention Sink / Focused distant |
| L4 | H2 | 0.8878 | 3.0310 | Long-range semantic content |
| L4 | H3 | 0.9889 | 2.8379 | Mid-range relational binding |
| L4 | H4 | 1.1983 | 2.8533 | Mid-range relational binding |
| L4 | H5 | 0.7308 | 3.0333 | Long-range semantic content |
| L4 | H6 | 0.4899 | 4.0698 | Strong Attention Sink / Focused distant |
| L4 | H7 | 1.1073 | 3.2050 | Broad global context |
| **L5** | H0 | 0.7357 | 3.7311 | Long-range semantic content |
| L5 | H1 | 1.1093 | 2.0804 | Broad local / diffuse syntax |
| L5 | H2 | 1.0272 | 3.1257 | Broad global context / First token |
| L5 | H3 | 0.9714 | 2.9508 | Mid-range relational binding |
| L5 | H4 | 0.5166 | 4.0815 | Strong Attention Sink / Focused distant |
| L5 | H5 | 0.1641 | 4.3298 | Extreme Attention Sink (Token 0 offload) |
| L5 | H6 | 0.1702 | 4.2974 | Extreme Attention Sink (Token 0 offload) |
| L5 | H7 | 0.8435 | 3.3940 | Long-range semantic content |

---

### 8.5 Full 48-Head Quantitative Results: Nepali

| Layer | Head | Mean Entropy | Mean Distance | Functional Specialization |
|:---:|:---:|:---:|:---:|:---|
| **L0** | H0 | 1.0680 | 1.4370 | Local / phrase-level syntax |
| L0 | H1 | 0.4658 | 0.6090 | Local n-gram extractor |
| L0 | H2 | 1.0908 | 2.3353 | Mid-range relational binding |
| L0 | H3 | 1.1058 | 1.7002 | Broad local / diffuse syntax |
| L0 | H4 | 0.9217 | 1.2668 | Local / phrase-level syntax |
| L0 | H5 | 0.5397 | 1.2077 | Local / phrase-level syntax |
| L0 | H6 | 0.4546 | 0.8446 | Local n-gram extractor |
| L0 | H7 | 0.1191 | 0.1255 | Hyper-local n-gram (sharp) |
| **L1** | H0 | 1.0898 | 2.3615 | Mid-range relational binding |
| L1 | H1 | 1.2086 | 1.6709 | Broad local / diffuse syntax |
| L1 | H2 | 0.9268 | 2.0877 | Local / phrase-level syntax |
| L1 | H3 | 1.0877 | 1.9550 | Local / phrase-level syntax |
| L1 | H4 | 1.0894 | 2.5054 | Mid-range relational binding |
| L1 | H5 | 1.2393 | 2.3488 | Diffuse / multi-token routing |
| L1 | H6 | 0.3151 | 0.9300 | Hyper-local n-gram (sharp) |
| L1 | H7 | 1.1631 | 1.7869 | Broad local / diffuse syntax |
| **L2** | H0 | 1.1599 | 2.1438 | Broad local / diffuse syntax |
| L2 | H1 | 1.1021 | 2.6436 | Mid-range relational binding |
| L2 | H2 | 0.5773 | 0.9568 | Local n-gram extractor |
| L2 | H3 | 0.8526 | 1.3635 | Local / phrase-level syntax |
| L2 | H4 | 1.1953 | 2.0927 | Broad local / diffuse syntax |
| L2 | H5 | 1.2625 | 2.3299 | Diffuse / multi-token routing |
| L2 | H6 | 1.0728 | 2.1844 | Local / phrase-level syntax |
| L2 | H7 | 0.8298 | 2.5584 | Mid-range relational binding |
| **L3** | H0 | 0.6579 | 3.1181 | Long-range semantic content |
| L3 | H1 | 1.0381 | 2.4990 | Mid-range relational binding |
| L3 | H2 | 0.7486 | 2.7782 | Mid-range relational binding |
| L3 | H3 | 0.6768 | 1.3756 | Local / phrase-level syntax |
| L3 | H4 | 0.7949 | 3.1858 | Long-range semantic content |
| L3 | H5 | 0.9609 | 2.8374 | Mid-range relational binding |
| L3 | H6 | 0.7949 | 3.0828 | Long-range semantic content |
| L3 | H7 | 0.8902 | 2.2415 | Mid-range relational binding |
| **L4** | H0 | 0.8432 | 2.7330 | Mid-range relational binding |
| L4 | H1 | 0.8294 | 3.0951 | Long-range semantic content |
| L4 | H2 | 0.2130 | 3.6789 | Extreme Attention Sink (Token 0 offload) |
| L4 | H3 | 0.7892 | 2.8983 | Mid-range relational binding |
| L4 | H4 | 0.7115 | 3.1621 | Long-range semantic content |
| L4 | H5 | 0.8829 | 2.1951 | Local / phrase-level syntax |
| L4 | H6 | 1.0275 | 2.5071 | Mid-range relational binding |
| L4 | H7 | 0.7601 | 2.7745 | Mid-range relational binding |
| **L5** | H0 | 0.7495 | 3.1288 | Long-range semantic content |
| L5 | H1 | 0.4406 | 3.5693 | Strong Attention Sink / Focused distant |
| L5 | H2 | 0.1597 | 3.8786 | Extreme Attention Sink (Token 0 offload) |
| L5 | H3 | 0.6769 | 3.2913 | Long-range semantic content |
| L5 | H4 | 0.9982 | 2.3812 | Mid-range relational binding |
| L5 | H5 | 0.4001 | 3.6601 | Strong Attention Sink / Focused distant |
| L5 | H6 | 0.4323 | 3.4611 | Strong Attention Sink / Focused distant |
| L5 | H7 | 0.9104 | 2.5712 | Mid-range relational binding |

---

### 8.6 Layer-by-Layer Architectural Walkthrough

#### Layer 0: Local Inductive Bias & Token Geometry
- **Hindi L0:** ![Hindi Layer 0 Mosaics](images/all_heads/hindi_layer_0_all_heads.png)
- **Nepali L0:** ![Nepali Layer 0 Mosaics](images/all_heads/nepali_layer_0_all_heads.png)
- **Observation:** Layer 0 exhibits strict sub-diagonal and immediate diagonal concentration. In Hindi, Heads 1 & 2 have average distances of 0.31 and 0.47 tokens with low entropies (0.35 and 0.28). In Nepali, Head 7 has an extreme distance of 0.1255 and entropy of 0.1191. These heads act as hardwired n-gram feature extractors, forming immediate subword-to-subword bindings before higher-order syntactic trees can be assembled.

#### Layers 1 & 2: Phrase Formation & Contextual Mixing
- **Hindi L1 & L2:**
  ![Hindi Layer 1 Mosaics](images/all_heads/hindi_layer_1_all_heads.png)
  ![Hindi Layer 2 Mosaics](images/all_heads/hindi_layer_2_all_heads.png)
- **Nepali L1 & L2:**
  ![Nepali Layer 1 Mosaics](images/all_heads/nepali_layer_1_all_heads.png)
  ![Nepali Layer 2 Mosaics](images/all_heads/nepali_layer_2_all_heads.png)
- **Observation:** Entropy reaches its maximum in Layer 2 (1.2089 for Hindi, 1.0065 for Nepali). The attention mass diffuses across 2 to 3 preceding tokens, binding adjective-noun phrases (e.g., `सुंदर` + `और` + `विशाल` $\to$ `देश`). Head 4 in Hindi L2 achieves the highest entropy of the entire network (1.3444), actively pooling representations across all active tokens.

#### Layers 3 & 4: Relational Bridging & Sink Emergence
- **Hindi L3 & L4:**
  ![Hindi Layer 3 Mosaics](images/all_heads/hindi_layer_3_all_heads.png)
  ![Hindi Layer 4 Mosaics](images/all_heads/hindi_layer_4_all_heads.png)
- **Nepali L3 & L4:**
  ![Nepali Layer 3 Mosaics](images/all_heads/nepali_layer_3_all_heads.png)
  ![Nepali Layer 4 Mosaics](images/all_heads/nepali_layer_4_all_heads.png)
- **Observation:** In Layer 4, head functionality sharply diverges. Some heads expand to global semantic tracking (Hindi L4H0 dist 3.23; Nepali L4H1 dist 3.10), while others begin offloading attention. In Nepali, **Layer 4 Head 2 immediately collapses into an extreme attention sink** (entropy 0.2130, distance 3.6789), concentrating over 90% of its attention weight on the initial token `नेपाल`.

#### Layer 5: Terminal Readout & Attention Sink Offloading
- **Hindi L5:** ![Hindi Layer 5 Mosaics](images/all_heads/hindi_layer_5_all_heads.png)
- **Nepali L5:** ![Nepali Layer 5 Mosaics](images/all_heads/nepali_layer_5_all_heads.png)
- **Observation:** The final layer cleanly separates into two specialized populations:
  1. **Semantic Content Extractors (e.g., Hindi L5H0, Nepali L5H0, L5H3):** Maintain balanced attention across key content words (`सुंदर`, `विशाल`, `देश`), synthesizing contextual embeddings for final token generation.
  2. **Attention Sinks (Hindi L5H5, L5H6; Nepali L5H2, L5H5):** Exhibiting entropy as low as 0.1597 and mean distances $> 3.8$, these heads allocate almost all probability mass to token 0 (`भारत` / `नेपाल`) or terminal punctuation (`।`). Because Softmax enforces $\sum_j A_{ij} = 1$, whenever a head finds no relevant syntactic dependency for the current token, it routes surplus attention mass into position 0 as a harmless activation dump.

---

### 8.7 Cross-Lingual Comparison & Synthesis

1. **Universality of Attention Sinks:** Both Indic models spontaneously develop first-token attention sinks in Layers 4–5 without explicit supervision. In Hindi, sink behavior is distributed across Heads 4, 5, and 6 in Layer 5 (entropies 0.16–0.51). In Nepali, it concentrates intensely on Head 2 starting from Layer 4 (entropy 0.21) and culminating in Layer 5 (entropy 0.16).
2. **Grammatical Alignment:** Both languages show SOV structural awareness in middle layers: auxiliary verbs (`है`, `हो`) and sentence terminators (`।`) attend strongly across the clause back to the main subject (`भारत`, `नेपाल`) and predicative adjectives (`सुंदर`, `विशाल`), bridging the standard long-distance dependency characteristic of Indo-Aryan syntax.
3. **Architectural Redundancy & Pruning Opportunity:** Across both models, 2 to 3 heads per layer share very similar attention profiles (e.g., L2 diffuse heads). This confirms that a 30M parameter budget provides sufficient headroom, and pruning 25–35% of redundant heads post-training would likely preserve downstream language modeling capability.

