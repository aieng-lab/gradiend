# Oriented cross-encoding matrix

When you train **many pairwise GRADIENDs** over the same feature domain (e.g. race:
white↔black, white↔asian, black↔asian), a single pairwise cross-encoding heatmap
from [`suite.plot_cross_encoding_heatmap()`][gradiend.trainer.suite.base.TrainerSuite.plot_cross_encoding_heatmap] is no longer enough. You want to know:

- Does a GRADIEND trained on one pair also **encode** snippets from other pairs?
- After **orienting** signs so “left class = +1, right class = −1”, do anchors line up
  on the diagonal?

**Dense cross-encoding** answers this by encoding every trained GRADIEND on a **shared
test pool** (all directed transitions merged across trainers) and aggregating into
square anchor-aligned matrices.

Use this workflow when:

- you have **≥3 feature classes** trained as separate pairwise GRADIENDs;
- classes share surface tokens or transitions and you need comparable signs;
- you want publication-style **feature × feature** leakage matrices.

For two-run comparisons, [`suite.plot_cross_encoding_heatmap()`][gradiend.trainer.suite.base.TrainerSuite.plot_cross_encoding_heatmap] is simpler and faster.

**Computation details:** [Oriented cross-encoding: computation](cross-encoding-matrix-computation.md)  
**Formal notation:** [Oriented cross-encoding matrix (paper)](../paper/cross_encoding_matrix.tex)

---

## Two matrix views

| Output | API | Rows | Columns | Read as |
|--------|-----|------|---------|---------|
| Pre-anchor GRADIEND × transition | [`plot_gradiend_transition_cross_encoding_heatmap`][gradiend.visualizer.heatmaps.encoding.plot_gradiend_transition_cross_encoding_heatmap] | trained GRADIEND id | directed transition `factual→alternative` | Raw mean encoding per model and input transition |
| Oriented square heatmaps | [`plot_cross_encoding_heatmap`][gradiend.visualizer.heatmaps.encoding.plot_cross_encoding_heatmap] | feature-class **anchor** | factual class, counterfactual class, or transition (`alignment`) | Sign-aligned, aggregated across all GRADIENDs whose pair contains the anchor |

**Pre-anchor** values are direct means from the cross-task encoder pass. **Oriented**
matrices flip signs so the anchor class is always “positive”, then average within each
GRADIEND and across GRADIENDs that share the anchor.

---

## Real example plots (trained models)

The figures below come from a **real** small multilingual demo: three race GRADIENDs
plus German der/dem article pairs, trained and evaluated on held-out data.

[:material-file-code-outline: `multilingual_gradiend_demo_small.py`](https://github.com/aieng-lab/gradiend/blob/main/experiments/multilingual_gradiend_demo_small.py)

```bash
python experiments/multilingual_gradiend_demo_small.py --plot-only
# Copy PDFs from runs/.../ to docs/img/ (see docs/img/README.md)
```

![GRADIEND × transition (pre-anchor, trained models)](../img/cross_encoding_gradiend_by_transition.png)

<!-- DOC_PLOT: docs/img/cross_encoding_gradiend_by_transition.png
Regenerate: experiments/multilingual_gradiend_demo_small.py (--plot-only if checkpoints exist)
Copy: runs/multilingual_gradiend_demo_small/cross_encoding_gradiend_by_transition_heatmap.pdf -> docs/img/
-->

![Oriented counterfactual matrix (trained models)](../img/cross_encoding_oriented_counterfactual.png)

<!-- DOC_PLOT: docs/img/cross_encoding_oriented_counterfactual.png
Regenerate: experiments/multilingual_gradiend_demo_small.py
Copy: runs/multilingual_gradiend_demo_small/cross_encoding_oriented_counterfactual_heatmap.pdf -> docs/img/
-->

!!! note "Real vs synthetic"
    These overview plots reflect **actual encoder evaluations** (checkpoint-dependent
    values). The step-by-step aggregation walkthrough below uses **hand-set synthetic
    data** so cell arithmetic is stable in the docs without retraining.

---

## Runnable pipeline

For **one symmetric suite** with a few peer classes (e.g. three race GRADIENDs),
[`suite.plot_cross_encoding_heatmap()`][gradiend.trainer.suite.base.TrainerSuite.plot_cross_encoding_heatmap] is enough — see
[Trainer suites](trainer-suites.md) and
[train_race_symmetric_suite.py](https://github.com/aieng-lab/gradiend/blob/main/gradiend/examples/train_race_symmetric_suite.py).

The workflows below are for **dense** matrices when many pairwise GRADIENDs share a
large feature domain.

The **small demo**
[multilingual_gradiend_demo_small.py](https://github.com/aieng-lab/gradiend/blob/main/experiments/multilingual_gradiend_demo_small.py)
trains three race GRADIENDs plus German der/dem article pairs — enough for a 6×6
oriented matrix without the full multilingual runtime. See that file for CLI flags
(e.g. `--plot-only` to replot from cache).

The **full demo**
[multilingual_gradiend_demo.py](https://github.com/aieng-lab/gradiend/blob/main/experiments/multilingual_gradiend_demo.py)
adds pronouns, religion, sentiment, and the complete German case grid.

Core plotting pattern (shared by both demos):

```python
encoder_summary = build_cross_task_encoder_summary(
    trainers_by_id,
    feature_order,
    split="test",
    max_size=config.args.encoder_eval_max_size,
)
transition_order = collect_unified_test_transitions(trainers_by_id, split="test")

# 1) Rectangular: every GRADIEND × every input transition
plot_gradiend_transition_cross_encoding_heatmap(
    trainers_by_id,
    trainer_order=trainer_order,
    transition_order=transition_order,
    encoder_summary=encoder_summary,
    output_path="cross_encoding_gradiend_by_transition_heatmap.pdf",
)

# 2) Square: one heatmap per alignment mode
for alignment in ("factual", "counterfactual", "transition"):
    plot_cross_encoding_heatmap(
        trainers_by_id,
        feature_order,
        alignment=alignment,
        encoder_summary=encoder_summary,
        cross_task_eval=False,
        output_path=f"cross_encoding_oriented_{alignment}_heatmap.pdf",
    )
```

**What you can change:**

| Parameter | Effect |
|-----------|--------|
| `feature_order` | Row/column order in oriented matrices; use sorted class ids for reproducible papers |
| `alignment` | `"factual"` columns = `factual_class`; `"counterfactual"` = `alternative_class`; `"transition"` = directed pair |
| `encoder_summary` | Precompute once and pass to all plots to avoid re-encoding |
| `cross_task_eval=False` | Use the shared cross-task encoder cache (recommended for suites) |
| `split`, `max_size` | Which held-out rows enter the matrix; cap for speed during development |

Cross-task encoder rows use the **same** per-trainer cache as
``evaluate_encoder`` — ``encoded_values_max_size_{N}_split_test.csv`` (or
``encoded_values_split_test.csv`` when ``max_size`` is unset). A cross-task pool
wider than a pair-local (train-only) cache replaces it automatically. Set
``TrainingArguments.use_cache=True`` to reuse it across plotting runs.

---

## How aggregation works (synthetic walkthrough) { #synthetic-walkthrough }

The rest of this section uses **synthetic encoder means** — not the trained-model
figures above. Four GRADIENDs:

| GRADIEND | Pair | Family |
|----------|------|--------|
| `race_white_asian` | white / asian | race |
| `race_black_asian` | black / asian | race |
| `race_white_black` | white / black | race |
| `gender_he_she` | he / she | English gender (unrelated) |

Feature order in the oriented matrix: **White**, **Black**, **Asian** (Race bracket),
then **he**, **she** (Gender). Alignment: **counterfactual** (column keys =
counterfactual class).

![Synthetic pre-anchor matrix (full transition pool)](../img/cross_encoding_synthetic_preanchor_overview.png)

<!-- DOC_PLOT: docs/img/cross_encoding_synthetic_preanchor_overview.png
Regenerate: python scripts/generate_cross_encoding_matrix_doc_figures.py
-->

![Synthetic oriented overview (counterfactual probes)](../img/cross_encoding_synthetic_oriented_overview.png)

<!-- DOC_PLOT: docs/img/cross_encoding_synthetic_oriented_overview.png
Regenerate: python scripts/generate_cross_encoding_matrix_doc_figures.py
-->

!!! tip "Mixed-sign diagonals are normal"
    Unlike a correlation matrix, oriented entries depend on **which side of each
    binary pair** a feature was trained on. In this synthetic matrix,
    `white`/`white` and `black`/`black` are **negative** while `asian`/`asian`
    is **positive** — the same pattern you often see in real dense matrices.
    For English gender, `she` is the **second** class in the pair, so
    `she`/`she` is negative by the anchor sign convention even when the encoder
    is strong; `he`/`he` is positive. The two gender classes are **inverted by
    definition** (off-diagonal `he`↔`she` have opposite signs).

Regenerate all synthetic figures:

```bash
python scripts/generate_cross_encoding_matrix_doc_figures.py
```

Fixture source: [`scripts/synthetic_cross_encoding_fixture.py`](https://github.com/aieng-lab/gradiend/blob/main/scripts/synthetic_cross_encoding_fixture.py)

### Example 1 — diagonal cell `(asian, asian)`

**Question:** Why is $M_{\texttt{asian},\texttt{asian}} \approx +0.55$ on the synthetic
matrix?

#### Step 1 — Column key (counterfactual alignment)

Counterfactual column `asian` collects transitions whose **counterfactual** class is
`asian`:

- `white→asian`
- `black→asian`

(There is no `asian→asian` transition.)

#### Step 2 — Row key (anchor aggregation)

Anchor row `asian` aggregates GRADIENDs whose pair **contains** `asian`:

- `race_white_asian`
- `race_black_asian`

`race_white_black` and `gender_he_she` do **not** contribute.

#### Step 3 — Pre-anchor contributors (highlighted)

![Synthetic pre-anchor highlights for (asian, asian)](../img/cross_encoding_synthetic_preanchor_diagonal_highlight.png)

Orange outlines mark the four GRADIEND × transition cells that feed this oriented entry.

#### Step 4 — Anchor sign and aggregate

`asian` is the **second** class in both race pairs → anchor sign **−1**. Raw
pre-anchor means on incoming-asian snippets are negative; flipping yields positive
signed contributions. Mean across the four signed values, then across the two
GRADIENDs → **≈ +0.55**.

![Synthetic aggregation table (asian, asian)](../img/cross_encoding_synthetic_aggregation_diagonal.png)

#### Step 5 — Oriented cell

![Synthetic oriented highlight (asian, asian)](../img/cross_encoding_synthetic_oriented_diagonal_highlight.png)

---

### Example 2 — off-diagonal cell `(white, asian)`

**Question:** How does cross-encoding differ when the orienting feature is **not** the
probe class?

Same column (`asian` counterfactual probes), but row `white` now aggregates
`race_white_asian` and `race_white_black`. Signed contributions partially cancel;
the synthetic entry is **≈ −0.24** (weak cross-encoding / mixed signs).

![Synthetic pre-anchor highlights for (white, asian)](../img/cross_encoding_synthetic_preanchor_offdiag_highlight.png)

![Synthetic aggregation table (white, asian)](../img/cross_encoding_synthetic_aggregation_offdiag.png)

![Synthetic oriented highlight (white, asian)](../img/cross_encoding_synthetic_oriented_offdiag_highlight.png)

Compare the two oriented highlights: mixed-sign **race** diagonals (`asian` positive,
`white`/`black` negative), near-zero **race × gender** cells, and the inverted
**he**/`she` block (positive `he`/`he`, negative `she`/`she`).

---

### Example 3 — negative race diagonal `(white, white)`

Not every diagonal is positive. `white` is the **first** class in two race GRADIENDs
but aggregating counterfactual probes with anchor sign **+1** can still yield a
**negative** oriented entry (≈ **−0.50** here) when pre-anchor means on
`→white` transitions are predominantly negative under `source=alternative`.

This is the same aggregation recipe as Example 1 — only the signed contributions
and final mean differ. Do **not** read a negative diagonal as “the encoder failed”
without checking anchor position and alignment.

---

### What to look for in real matrices

- **Diagonal magnitude** (often ≈ ±1 for selective encoders): sign depends on anchor
  position in each binary pair — **not** every diagonal is positive.
- **Binary pairs** (e.g. English `he`/`she`, sentiment `Pos`/`Neg`): the second class
  picks up anchor sign **−1**, so its diagonal can be negative while the first class
  is positive; off-diagonal entries are inverted relative to each other.
- **Near-zero off-diagonal** within a family: little cross-feature leakage.
- **Cross-family cells** (e.g. race × gender): often near zero when no GRADIEND pair
  linked those features — see [computation limitation](cross-encoding-matrix-computation.md#limitation-incomplete-cross-family-identification).
- **Factual vs counterfactual** heatmaps differ because column keys use different
  class columns — compare both when explaining a result.

---

## Related docs

- [Oriented cross-encoding: computation](cross-encoding-matrix-computation.md) — definitions and API map
- [Cross-model comparison](cross-model-comparison.md) — when to use dense matrices vs top-k overlap
- [Trainer suites](trainer-suites.md) — orchestrating many pairwise runs
- [Evaluation & visualization](evaluation-visualization.md) — heatmap customization
