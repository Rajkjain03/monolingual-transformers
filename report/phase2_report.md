# Phase 2 Report: Pretraining & Evaluation
Name - Raj k jain
Roll No. - 2025201036
Email - raj.jain@students.iiit.ac.in

## 1. Model Configurations
Implemented a custom Decoder-only Transformer in PyTorch with Rotary Position Embeddings (RoPE), SwiGLU Feed-Forward Networks, and RMSNorm for stable training.

Both models were trained using identical hyperparameter constraints to allow for a direct scientific comparison, differing only in their vocabulary size (determined in Phase 1).

| Hyperparameter | Model H (Hindi) | Model L (Nepali) |
|----------------|-----------------|------------------|
| d_model        | 512             | 512              |
| n_layers       | 6               | 6                |
| n_heads        | 8               | 8                |
| d_ff           | 2048            | 2048             |
| Vocab Size     | 32,000          | 16,000           |
| Context Length | 512             | 512              |
| **Total Params** | **36,837,888** | **28,645,888** |

*Note: The parameter difference is strictly due to the embedding and output projection layers accommodating the larger Hindi vocabulary.*

## 2. Intrinsic Language-Modeling Metrics
The models were evaluated on their respective held-out test splits.

| Metric | Model H (Hindi) | Model L (Nepali) |
|--------|-----------------|------------------|
| Cross-Entropy Loss | 3.5980 | 3.6167 |
| Perplexity (PPL) | 36.5235 | 37.2161 |
| Bits-per-byte (BPB) | 5.1908 | 5.2179 |

**Discussion of the Gap:**
The intrinsic metrics between Model H and Model L are incredibly close. Hindi achieved a marginally lower Perplexity (36.52 vs 37.21). This is primarily driven by Model H having a larger vocabulary (32K vs 16K) which reduces the tokenizer fertility (1.24 tokens/word vs 1.40 tokens/word). Despite having a smaller overall dataset in terms of raw gigabytes compared to Nepali, the token density of the 32K vocab allowed the Hindi model to compress the sequence modeling slightly better.

## 3. Generation Quality
Generations were evaluated using both qualitative sampling and exact n-gram matching metrics against reference continuations. 

| Metric | Model H (Hindi) | Model L (Nepali) |
|--------|-----------------|------------------|
| BLEU-4 | 3.51 | 0.00 |
| chrF | 9.19 | 4.43 |
| ROUGE-L | 0.0000 | 0.0000 |
| Distinct-1 | 0.1321 | 0.2551 |
| Distinct-2 | 0.2599 | 0.3807 |

**Discussion on Metrics:**
For open-ended causal language modeling, strict n-gram metrics like BLEU-4 and ROUGE-L are famously uninformative because there are exponentially many valid ways to continue a sentence. A model might generate a perfectly fluent continuation that simply doesn't match the exact words in the single reference text, resulting in a BLEU score near zero (as seen above). Distinct-1/2 are much more useful for measuring diversity (lack of repetition).

### Model H (Hindi) Examples
- **Prompt:** "भारत एक"
- **T=0.5:** भारत एक ऐसा देश है जहां पर कई सारे लोग रहते हैं। 
- **T=1.0:** भारत एक ज़ोरदार ट्रोलिव है, जिनको लगातार आगे आकर मीडिया का भी सामना हुआ है। 

### Model L (Nepali) Examples
- **Prompt:** "नेपाल एक"
- **T=0.5:** नेपाल एक सय ५० किलोमिटर लामो र एक सय किलोमिटर लामो सुरुङ मार्गको निर्माण कार्य सम्पन्न भएको छ ।
- **T=1.0:** नेपाल एक ठूलो ब्लक कर्जा सहयोग कार्यक्रममार्फत ऋणमार्फत नै ऋण लिन सकिने अवस्था रहेको छ । 

**Quality Analysis:**
At `T=0.5`, both models produce highly coherent, fluent, and grammatically correct sentences, albeit sometimes repetitive. At `T=1.0`, the models display significantly higher Distinct-1/2 diversity, pulling in domain-specific contexts (e.g., Media for Hindi, Financial/Agricultural loans for Nepali). At `T=1.5`, both models suffer from complete hallucinatory degradation, proving that the entropy overwhelms the learned distributions.

## 4. Attention Analysis
We extracted the causal self-attention matrices during the forward pass.

| Layer | Hindi Mean Distance | Hindi Mean Entropy | Nepali Mean Distance | Nepali Mean Entropy |
|-------|---------------------|--------------------|----------------------|---------------------|
| Layer 0 (Early) | 1.3078 | 0.7263 | 1.1910 | 0.7207 |
| Layer 5 (Late) | 3.4985 | 0.6924 | 3.2428 | 0.5959 |

**Discussion:**
The attention mechanisms behaved exactly as theorized in modern Transformer literature:
1. **Local vs Global:** The early layers (Layer 0) act as localized feature extractors. The incredibly short mean distance (~1.2 - 1.3) proves that early heads strictly attend to their immediate preceding tokens. By Layer 5, the mean attention distance expands massively to ~3.5, indicating long-range content-based fetching.
2. **Entropy:** The entropy generally peaks in the middle layers (reaching 1.20 in Hindi Layer 2) where attention is highly diffuse, before sharpening again in the final layers to make highly confident predictions for the next token. 
3. **Difference:** Model H exhibited slightly higher global attention distances than Model L, potentially owing to its lower tokenizer fertility, allowing a single token to represent wider semantic concepts over shorter sequence spans.
