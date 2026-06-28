# Oriented cross-encoding: computation

This page defines the **oriented cross-encoding matrix** $M$ and how library code
aggregates pre-anchor encoder means into one square cell. It complements the
workflow guide [Oriented cross-encoding matrix](cross-encoding-matrix.md).

Paper appendix: [cross_encoding_matrix.tex](../paper/cross_encoding_matrix.tex).

---

## Setup

Let $\mathcal{F}=\{f_1,\ldots,f_K\}$ be feature classes (e.g. `white`, `der`,
`fem_nom`) and $\mathcal{G}=\{G_1,\ldots,G_M\}$ a set of pairwise
[GRADIEND](../api/training/Trainer.md) models. Each $G\in\mathcal{G}$ is trained
on an ordered pair $(a_G,b_G)\in\mathcal{F}^2$.

The **shared test pool** $\mathcal{D}$ contains directed contrasts: every probe
$x\in\mathcal{D}$ has a factual class $\mathrm{fac}(x)\in\mathcal{F}$ and a
counterfactual $\mathrm{cnf}(x)\in\mathcal{F}$. For $f,c\in\mathcal{F}$ with
$\mathcal{D}_{f,c}\neq\emptyset$,

$$
\mathcal{D}_{f,c}=\{x\in\mathcal{D}:\mathrm{fac}(x)=f,\ \mathrm{cnf}(x)=c\}.
$$

For model $G$, the **encoding mean** on that contrast slice is

$$
\bar{e}_G(f,c)=\frac{1}{|\mathcal{D}_{f,c}|}\sum_{x\in\mathcal{D}_{f,c}}\mathrm{enc}_G(x),
$$

where $\mathrm{enc}_G(x)$ is the scalar encoded feature value of probe $x$ under
$G$. In code, these are rows of `encoder_df` grouped by directed transition
`factual→counterfactual`; see
[`compute_gradiend_transition_cross_encoding_matrix`][gradiend.comparison.feature_cross_encoding.compute_gradiend_transition_cross_encoding_matrix]
for the rectangular **pre-anchor** matrix (GRADIEND × transition).

---

## Rows vs columns

Rows and columns are both labeled by features in $\mathcal{F}$, but play different roles:

| Role | Index | Meaning |
|------|-------|---------|
| **Column** $f_j$ | Probe feature | Average $\mathrm{enc}_G(x)$ over probes whose **aligned class** is $f_j$ (factual, counterfactual, or transition depending on `alignment`) |
| **Row** $f_i$ | Orienting feature | Average over GRADIENDs whose pair contains $f_i$, after anchor sign alignment |

Define $\mathcal{G}_{f_i}=\{G\in\mathcal{G}: f_i\in\{a_G,b_G\}\}$ and the
**anchor sign**

$$
\mathrm{sign}_G(f_i)=
\begin{cases}
+1 & \text{if } f_i=a_G \\
-1 & \text{if } f_i=b_G
\end{cases}
$$

Let $\mu_G(f_j)$ be the mean encoded response of $G$ on probes aligned to column
$f_j$ (within-model average over contributing transitions). The **oriented**
matrix entry is

$$
M_{f_i,f_j}
= \frac{1}{|\mathcal{G}_{f_i}|}
\sum_{G\in\mathcal{G}_{f_i}} \mathrm{sign}_G(f_i)\,\mu_G(f_j).
$$

Implementation:
[`compute_anchor_aligned_encoding_matrix`][gradiend.comparison.anchor_aligned.compute_anchor_aligned_encoding_matrix]
builds per-contribution rows in `aligned_rows`, then
[`aggregate_anchor_aligned_encoding_rows`][gradiend.comparison.anchor_aligned.aggregate_anchor_aligned_encoding_rows]
pivots to $M$.

---

## What $M$ is (and is not)

- $M$ is **not** a similarity matrix: $M_{f_i,f_j}\neq M_{f_j,f_i}$ in general
  because rows and columns aggregate different quantities.
- **Diagonal** $M_{f_i,f_i}$: encoding strength when probe and orienting feature
  match. Selective encoders often show $|M_{f_i,f_i}|\approx 1$, but the **sign**
  depends on whether $f_i$ was the first ($+1$) or second ($-1$) class in each
  contributing binary pair — so diagonals can be **negative** even for a working
  encoder (common for the second class in binary features such as English `he`/`she`).
- **Off-diagonal** $M_{f_i,f_j}$: cross-encoding from orienting feature $f_i$ to
  probe feature $f_j$ (leakage / shared structure). For inverted binary pairs,
  $M_{f_a,f_b}$ and $M_{f_b,f_a}$ typically have opposite signs.

When `model.source` is `alternative` and `alignment="counterfactual"`, column
keys use counterfactual classes; factual alignment uses factual classes. See
[`encoding_view_sign_for_source`][gradiend.model._source_target.encoding_view_sign_for_source]
for the view sign applied before anchor aggregation.

---

## Limitation: incomplete cross-family identification { #limitation-incomplete-cross-family-identification }

Row $f_i$ uses **only** GRADIENDs in $\mathcal{G}_{f_i}$. A feature that appears
only in one feature family (e.g. `christian` from religion pairs) never borrows
signal from unrelated families. Cross-family off-diagonal cells can be **near zero**
simply because no trained pair linked those families — not because the encoder
is perfectly selective.

The synthetic walkthrough in
[How aggregation works (synthetic walkthrough)](cross-encoding-matrix.md#synthetic-walkthrough)
uses three race GRADIENDs plus one unrelated English gender (`he`/`she`) pair to show
within-family structure, mixed-sign diagonals, and binary-class inversion.

---

## API map

| Step | Function |
|------|----------|
| Shared test pool encoding | [`build_cross_task_encoder_summary`][gradiend.comparison.feature_cross_encoding.build_cross_task_encoder_summary] |
| Pre-anchor GRADIEND × transition | [`compute_gradiend_transition_cross_encoding_matrix`][gradiend.comparison.feature_cross_encoding.compute_gradiend_transition_cross_encoding_matrix] |
| Oriented rows + pivot | [`compute_anchor_aligned_encoding_matrix`][gradiend.comparison.anchor_aligned.compute_anchor_aligned_encoding_matrix] |
| Plot | [`plot_cross_encoding_heatmap`][gradiend.visualizer.heatmaps.encoding.plot_cross_encoding_heatmap] |
| Inspect contributions | `payload["aligned_rows"]` on oriented payload |

---

## Related

- [Oriented cross-encoding matrix](cross-encoding-matrix.md) — runnable pipeline and figures
- [Evaluation & visualization](evaluation-visualization.md) — heatmap customization
- [Cross-model comparison](cross-model-comparison.md) — when to use dense matrices
