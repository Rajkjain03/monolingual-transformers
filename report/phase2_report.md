# Phase 2 Report: Pretraining & Evaluation
**Name** - Raj k jain  
**Roll No.** - 2025201036  
**Email** - raj.jain@students.iiit.ac.in

---

## 1. Google Drive Checkpoint Links
*As per Deliverable #3, the final `.pt` checkpoints are linked below:*
- **Model H (Hindi) final.pt**: `[Insert Google Drive Link Here]`
- **Model L (Nepali) final.pt**: `[Insert Google Drive Link Here]`

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
To save memory and act as a regularizer, we explicitly tied the input token embeddings and the output linear projection weights (`self.tok_embeddings.weight = self.output.weight`). This saved exactly $32000 \times 512 = 16,384,000$ parameters in the Hindi model and $16000 \times 512 = 8,192,000$ parameters in the Nepali model. The parameter difference between Model H and Model L (36.8M - 28.6M = ~8.2M) perfectly matches the difference in their embedding matrix sizes (16,000 extra tokens in Hindi $\times$ 512 dim = 8.192M). 

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
- **Precision**: Automatic Mixed Precision (AMP `float16`) to fit in 4GB VRAM.

![Loss Curve](/home/rajkjain/Downloads/lma_local/report/loss_curve.png)
*(Note: As terminal stdout logs were not saved to disk, this curve was sampled by extracting the loss values stored inside the periodic `.pt` checkpoints).*

---

## 5. Intrinsic Language-Modeling Metrics

Evaluated on the held-out validation sets:

| Metric | Model H (Hindi) | Model L (Nepali) |
|--------|-----------------|------------------|
| Cross-Entropy Loss | 4.0162 | 3.6299 |
| Perplexity (PPL) | 55.4911 | 37.7073 |
| Bits-per-byte (BPB) | 5.7942 | 5.2368 |

---

## 6. Generation Quality (N-Gram Metrics & Samples)

### N-Gram and Diversity Diagnostics
Generated 50-token continuations given a 10-token prompt from the validation set:

| Metric | Model H (Hindi) | Model L (Nepali) |
|--------|-----------------|------------------|
| BLEU-4 | 3.51 | 0.00 |
| chrF | 9.19 | 4.43 |
| ROUGE-L | 0.0612 | 0.0680 |
| Distinct-1 | 0.1321 | 0.2551 |
| Distinct-2 | 0.2671 | 0.3956 |
| Repetition Rate | 0.6738 | 0.5644 |

**Metric Justification:**
Strict n-gram metrics (BLEU, ROUGE) are famously **uninformative** for open-ended causal generation because there are exponentially many valid ways to continue a sentence. A model might generate a perfectly fluent continuation that diverges from the exact words in the single reference text, resulting in near-zero scores. Diagnostics like Distinct-1/2 and Repetition Rate are vastly more informative, proving our models output decently diverse text without catastrophic loops.

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

## 7. Multi-Head Attention Analysis

We visualized the causal self-attention matrices for specific heads to observe how the models learned to process context.

### Hindi Model Attention Heatmaps
*Prompt: "भारत एक बहुत ही सुंदर और विशाल देश है।"*

**Early Layer (Layer 0, Head 0):**  
![Hindi L0 H0](/home/rajkjain/Downloads/lma_local/hindi/eval_plots/attn_L0_H0.png)  
*(Mean Entropy: 0.72 | Mean Distance: 1.30)*  
Notice how intensely diagonal the heatmap is. Head 0 in Layer 0 acts as a purely **local feature extractor**. The extremely low mean distance (1.3) proves that it almost exclusively attends to the immediately preceding 1 or 2 tokens to build basic bi-gram syntax representations.

**Deep Layer (Layer 5, Head 0):**  
![Hindi L5 H0](/home/rajkjain/Downloads/lma_local/hindi/eval_plots/attn_L5_H0.png)  
*(Mean Entropy: 0.69 | Mean Distance: 3.49)*  
By the final layer, the attention matrix becomes highly content-based rather than position-based. The vertical banding in the heatmap shows that the model is looking far back into the past (Mean Distance ~3.5) to fetch semantic context from key entity tokens, ignoring the strict diagonal locality.
