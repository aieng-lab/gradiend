# Token prediction methods

GRADIEND learns from gradients of a **prediction objective**. The same data columns and
[`TextPredictionTrainer`][gradiend.trainer.text.prediction.trainer.TextPredictionTrainer] API
work across model families; only **which token is scored** and **which context the model sees**
change.

Set [`TrainingArguments`][gradiend.trainer.core.arguments.TrainingArguments].prediction_objective
(default `"auto"`). **Always report the objective** when comparing runs — the same feature with
different objectives may not be comparable.

`auto` resolves per architecture:

- **Masked LM encoders** (BERT, RoBERTa, …) → `mlm_mask_token`
- **Decoder-only** (GPT-2, …) → `clm_next_token`, or `clm_mlm_head` when a saved auxiliary
  MLM head already exists under `experiment_dir`
- **Seq2seq** (T5, BART, …) → `seq2seq_encoder_mlm`

Decoder-side seq2seq objectives (`seq2seq_decoder`, `seq2seq_decoder_sequence_cloze`) are
**experimental** — convergence is not reliable yet; prefer encoder-side MLM for T5/BART.

---

## Objectives at a glance

All rows use the same idea: factual vs alternative class tokens in your data. The table shows
one **running input** (target position marked) and **what the loss scores**.

| Objective | Model family | Running example | Scored target | Status | Runnable example |
|-----------|--------------|-----------------|---------------|--------|------------------|
| `mlm_mask_token` | BERT / RoBERTa MLM | `The chef said [MASK] would return.` | Factual or alternative token at `[MASK]` (bidirectional context) | **Default** for MLM encoders | [:material-file-code-outline: `start_workflow.py`](https://github.com/aieng-lab/gradiend/blob/main/gradiend/examples/start_workflow.py) |
| `clm_next_token` | GPT-style decoder-only | `The chef said` **→** `he` / `they` | Next token after the prefix (left context only; set [`TextFilterConfig`][gradiend.data.text.filter_config.TextFilterConfig].min_left_context_words) | **Default** for decoder-only | [:material-file-code-outline: `train_gender_en.py`](https://github.com/aieng-lab/gradiend/blob/main/gradiend/examples/train_gender_en.py) (`--decoder-only`) |
| `clm_sequence_cloze` | GPT-style decoder-only | `The chef said [MASK] …` (+ optional RHS window) | Token at `[MASK]` using left prefix + right window | Optional | — |
| `clm_mlm_head` | Decoder-only + aux head | `The chef said [MASK] would return.` | Token at `[MASK]` after training a small MLM head on the backbone | Stable when head is trained | [:material-file-code-outline: `train_gender_de_decoder_only.py`](https://github.com/aieng-lab/gradiend/blob/main/gradiend/examples/train_gender_de_decoder_only.py) |
| `seq2seq_encoder_mlm` | T5 / BART | `The chef [MASK] the soup.` | Factual or alternative at `[MASK]` on the **encoder** (BERT-like) | **Default** for seq2seq (`auto`) | [:material-file-code-outline: `train_seq2seq_encoder_mlm.py`](https://github.com/aieng-lab/gradiend/blob/main/gradiend/examples/train_seq2seq_encoder_mlm.py) |
| `seq2seq_decoder` | T5 / BART | Decoder input ending before target token | Single token on the **decoder** | Experimental | — |
| `seq2seq_decoder_sequence_cloze` | T5 / BART | Decoder prefix + target continuation | Multi-token continuation on the decoder | Experimental | [:material-file-code-outline: `train_seq2seq_decoder_sequence.py`](https://github.com/aieng-lab/gradiend/blob/main/gradiend/examples/train_seq2seq_decoder_sequence.py) |

[:material-file-code-outline: `train_gender_de.py`](https://github.com/aieng-lab/gradiend/blob/main/gradiend/examples/train_gender_de.py) is another `mlm_mask_token` run (German articles on BERT).

Explicit override:

```python
args = TrainingArguments(prediction_objective="seq2seq_encoder_mlm")
```

---

## Decoder evaluation targets (separate setting)

Training objective ≠ which tokens are scored in **decoder evaluation**. When classes share
surface tokens, use row-wise eval — see [Decoder evaluation targets](decoder-eval-targets.md).
