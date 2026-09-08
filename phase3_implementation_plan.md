# Phase 3 Implementation Plan
### Reasoning Finetuning, Attention Analysis & Final Report
**Deadline: 16 Sep 2026, 11:59 PM (also the final project deadline) — 35 marks**

---

## 0. Prerequisites (confirm before starting)
- [ ] Phase 2 pretrained checkpoints for Hindi and Nepali (final `.pt` files) accessible
- [ ] Phase 1 tokenizer `.model` files for both languages accessible (must stay **frozen** — no retraining)
- [ ] Phase 2 attention-extraction code (from the heatmap/entropy/distance analysis) available to reuse

---

## 1. Synthetic Reasoning Dataset

### 1.1 Task types (per assignment spec)
- **Direct comparison** (2 entities): "A > B, which is bigger/smaller?"
- **Transitive chaining** (3 entities): "A > B, B > C ⇒ relation between A and C?"
- Grounded in concrete attributes: **height, age, price, quantity**

### 1.2 Design requirements
- Text in **target language's own script and natural phrasing** — not English
  templates with swapped-in Hindi/Nepali names
- Multiple template phrasings per reasoning type (2-3 variants) for diversity
- Programmatic generation (not LLM-written) so ground-truth labels are exact
  and verifiable

### 1.3 Example templates

**Hindi — direct comparison:**
```
"{A} की ऊँचाई {val_A} सेमी है और {B} की ऊँचाई {val_B} सेमी है।
इनमें से कौन लंबा है?"
→ answer: name with larger val
```

**Hindi — transitive (3-hop):**
```
"{A} की उम्र {B} से ज़्यादा है, और {B} की उम्र {C} से ज़्यादा है।
इनमें से सबसे छोटा कौन है?"
→ answer: C
```

**Nepali — equivalent structure**, written with Nepali phrasing/postpositions,
not a direct translation of the Hindi template with names swapped.

### 1.4 Generation pipeline (`generate_reasoning_data.py`)
1. Entity name pool: ~40-60 common names per language
2. Attribute value pools with randomized numeric ranges:
   - Height: 140-200 cm
   - Age: 5-90 years
   - Price: ₹10 - ₹10,000 (or NPR equivalent)
   - Quantity: 1-1000 units
3. Combine templates × entity triples × attribute types → generate examples
4. Assign each example a unique ID and store ground-truth answer alongside it

### 1.5 Leakage control (explicitly graded — get this right)
- **Held-out entity names**: reserve ~15-20% of the name pool exclusively for
  test set — these names never appear in ANY training example
- **Held-out relation pattern**: reserve one attribute type (e.g. "price") or
  one template phrasing variant exclusively for test
- Document both exclusions explicitly in the report with exact counts

### 1.6 Split sizes (finetuning-scale, not pretraining-scale)
| Split | Size (per language) |
|---|---|
| Train | 3,000 - 5,000 examples |
| Val | 300 - 500 examples |
| Test | 300 - 500 examples |

### 1.7 Deliverables checklist
- [ ] `generate_reasoning_data.py` (reproducible, seeded)
- [ ] Hindi train/val/test files
- [ ] Nepali train/val/test files
- [ ] Report subsection: template variety, size, leakage-avoidance method

---

## 2. Reasoning Finetuning

### 2.1 Protocol
- Start from each language's **final Phase 2 pretrained checkpoint**
- Keep that language's **tokenizer/vocab frozen** (no retraining)
- Full finetuning (model is small enough — 29-37M params — that LoRA/adapter
  methods aren't necessary)

### 2.2 Suggested hyperparameters (document actual values used)
| Parameter | Suggested value | Notes |
|---|---|---|
| Learning rate | 1e-5 to 5e-5 | Lower than pretraining LR (5e-4) — adaptation, not from-scratch learning |
| Epochs | 3-5 | Small dataset, watch for overfitting on val loss |
| Batch size | Same as pretraining setup (4 physical × 16 accum) | Reuse existing dataloader if compatible |
| Optimizer | AdamW, same betas as pretraining | Consistency with Phase 2 |

### 2.3 Checkpointing
- Same resumable format as Phase 2: weights + optimizer state + scheduler
  state + step + config
- Save finetuned checkpoints separately from pretrained ones (never overwrite
  the Phase 2 checkpoint)

### 2.4 Deliverables checklist
- [ ] `finetune_reasoning.py` (or per-language configs)
- [ ] Finetuned checkpoint — Hindi (Drive link)
- [ ] Finetuned checkpoint — Nepali (Drive link)
- [ ] Finetuning logs (loss curve, same discipline as Phase 2's loss curve)

---

## 3. Evaluation: Pretrained vs. Finetuned

### 3.1 Method
- Run the **same test set** through:
  1. The pretrained (zero-shot) checkpoint
  2. The finetuned checkpoint
- Metric: **exact-match accuracy** on the predicted entity name (simplest
  reliable metric for this task shape)

### 3.2 Reporting table
| | Hindi Pretrained | Hindi Finetuned | Nepali Pretrained | Nepali Finetuned |
|---|---|---|---|---|
| Direct comparison accuracy | | | | |
| Transitive (3-hop) accuracy | | | | |
| Overall accuracy | | | | |

### 3.3 Qualitative examples (required)
- At least one case finetuning **fixed** (wrong pretrained answer → correct
  finetuned answer)
- At least one **genuine failure case**, honestly reported — transitive
  3-hop reasoning is the likely weak point for a model this size; don't
  cherry-pick only successes

### 3.4 Deliverables checklist
- [ ] `evaluate_reasoning.py`
- [ ] Accuracy table (pretrained vs finetuned, both languages)
- [ ] Qualitative success + failure examples
- [ ] Written H vs L comparison on reasoning specifically

---

## 4. Attention Analysis (Post-Finetune)

### 4.1 Method
- Reuse the **exact Phase 2 attention extraction code**
- Point it at the **finetuned checkpoint** instead of pretrained
- Analyze the **same layer/head indices already used in Phase 2** (L0, L5,
  H0, H2) — keeps the comparison directly apples-to-apples
- Use a **reasoning-style prompt** this time (one of the comparison
  templates), not Phase 2's generic descriptive sentence

### 4.2 What to compare
- Pretrained heatmap vs. finetuned heatmap, same layer/head, same prompt
- Did finetuning shift any head to attend more strongly to the
  compared-entity tokens or numeric values specifically?
- Does the Phase 2 attention-sink head (L5 H2) persist after finetuning, or
  does finetuning reduce/redirect it?
- Local (early layer) vs. long-range (late layer) behavior — did finetuning
  change this balance at all?

### 4.3 Deliverables checklist
- [ ] Pretrained vs. finetuned heatmaps — ≥1 early layer, ≥1 late layer, both languages
- [ ] Written comparison discussing local/long-range shifts and head specialization changes
- [ ] Explicit discussion tied to comparative-reasoning prompts specifically

---

## 5. Final Report (Consolidates All Three Phases)

### 5.1 Required synthesis questions to answer explicitly
1. How did data scale and quality differ between Model H and Model L?
   *(pull from Phase 1: manual-collection story, token counts, fertility/UNK)*
2. How do language-modeling and reasoning results compare across resource tiers?
   *(Phase 2 PPL/BPB + Phase 3 accuracy, side by side)*
3. What tokenizer/corpus factors most affected the lower-resource model?
   *(Nepali's higher fertility, manual-scraping failure, etc.)*
4. What evidence explains the observed differences?
   *(attention-sink comparison, overfitting gap, reasoning accuracy gap)*

### 5.2 Structure
- Executive summary (1 paragraph)
- Phase 1 recap (key figures/tables only, not a full re-paste)
- Phase 2 recap (key figures/tables only)
- Phase 3 full detail (dataset, finetuning, evaluation, attention)
- Answers to the 4 synthesis questions above
- Final error analysis + comparison tables
- README with reproduction steps + all Drive links

### 5.3 Deliverables checklist
- [ ] Final consolidated report (md or PDF)
- [ ] README with reproduction steps and all Drive links
- [ ] All Phase 3 code committed to `phase-3` branch

---

## 6. Suggested Timeline

| Day | Task |
|---|---|
| 1 | Build + run `generate_reasoning_data.py` for both languages; verify splits/leakage control |
| 1-2 | Finetuning runs (should be fast — small model, small dataset) |
| 2 | `evaluate_reasoning.py`: pretrained vs finetuned accuracy |
| 2-3 | Attention re-analysis on finetuned checkpoints, reasoning prompts |
| 3 | Final report writing, consolidating all three phases, README |

---

## 7. Known risk points to watch for
- **Overfitting on the small finetune set** — monitor val loss during
  finetuning, don't just train for a fixed epoch count blindly
- **Transitive (3-hop) reasoning is the likely failure mode** for a
  25-37M-parameter model — expect and honestly report this rather than
  being surprised by it
- **Leakage bugs**: double-check held-out names/patterns never leak into
  train via template combination logic before generating the full dataset
