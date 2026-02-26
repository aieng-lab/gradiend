# Tutorial: Evaluation (inter-model)

This tutorial covers **inter-model** evaluation: comparing **multiple** trained runs to see whether they change the same or different parameters. You need several trained models (e.g. different class pairs or seeds, each with its own `run_id` under the same `experiment_dir`). For evaluating a **single** model (encoder/decoder) and then applying rewrites, see [Tutorial: Evaluation (intra-model)](evaluation-intra-model.md) and [Tutorial: Model Rewrite](model-rewrite.md).

!!! tip "Optional dependency: plotting"
    The Venn diagrams and heatmap in this tutorial require the **plot** extra. If you did not install it with GRADIEND, install it with:

    ```bash
    pip install gradiend[plot]
    ```

---

## Why compare across runs?

When you train GRADIEND for different feature pairs (e.g. different [Gender-Case transitions](https://github.com/aieng-lab/gradiend/blob/main/gradiend/examples/gender_de_detailed.py)), a natural question is: *Do these runs modify the same parameters or different ones?* 

We measure this by looking at the **top-k** most important parameters (e.g. decoder weights with largest absolute value) in each GRADIEND model and seeing how much they overlap.
Their overlap can be visualized using two main tools:

- **Venn diagrams**: Show the size of the top-k sets and their intersection for 2-6 GRADIEND models (including intersection counts of each subset)
- **Overlap heatmap**: Show a heatmap of pairwise GRADIEND top-k intersection counts. Can be used with any number of GRADIENDs, enabling large-scale comparisons.

If the top-*k* most important parameters overlap strongly across runs, the model may be reusing similar weights for different features. 
If they overlap little, the learned directions are more distinct. This kind of analysis is used, for example, to study whether language models encode grammatical features in a rule-like way or via memorized associations—see [Understanding or Memorizing? A Case Study of German Definite Articles in Language Models](https://arxiv.org/abs/2601.09313).

---

## Top-k overlap: what it measures

For each trained GRADIEND model, we can rank parameters (e.g. decoder weights) by **importance** (e.g. absolute weight magnitude) and take the **top-k**. **Top-k overlap** between two models is the overlap of these two sets: e.g. how many of the top-k parameters in run A are also in the top-k of run B. A **heatmap** of pairwise overlaps (across several runs) shows which runs affect similar parameters and which do not.

- **part** selects which parameters to rank: `encoder-weight`, `decoder-weight`, `decoder-bias`, or `decoder-sum`. By default, we use **decoder-weight**.
- **topk**: Number or fraction of dimensions to take as “top” (e.g. 1000 or 0.01).
- **value**: Metric for the heatmap cells, e.g. `"intersection_frac"` (fraction of the smaller set that lies in the intersection). See [Evaluation & visualization](../guides/evaluation-visualization.md#4-top-k-overlap-heatmap) for full customization options.

---

## Pre-requisites

To compare across runs, you need to have trained multiple GRADIEND models. For the remaining of the code, we assume you have a list of `Trainer` objects (one per run) that you want to compare. Each trainer should have a unique `run_id` that is used in the plot labels.

> Note that storing a lot of GRADIEND models may become memory-intensive. It is highly recommended to use *pruning* to pre-select important weights and only keep these in memory!


```python
# e.g. one trainer per class pair
models = {t.run_id: t.get_model() for t in trainers}
```

## Venn Diagrams

For 2-6 runs, you can use Venn diagrams to visualize the top-k overlap. The advantage over heatmaps is that Venn diagrams show the size of each set and the intersection counts for all subsets (e.g. how many parameters are in the top-k of run A and B but not run C or D). 
Moreover, for 2-3 sets, the Venn diagram is a very intuitive way to see the overlap because the area of the circles and their intersections visually represent the set sizes and overlaps. For more than 3 sets, Venn diagrams become harder to read, and heatmaps may be more effective.

```python
from gradiend.visualizer.topk import plot_topk_overlap_venn
plot_topk_overlap_venn(
    models,
    topk=1000,
    part="decoder-weight",
    output_path="topk_overlap_venn.pdf",
)
```

- **models**: Dict keys are used as **set labels** on the diagram; use the same keys as for the heatmap so labels are consistent.
- *Run the code above to generate the Venn diagram; use `output_path="topk_overlap_venn.pdf"` (or `.png`) to save.*

## Overlap Heatmap

The heatmap generalizes to any number of runs and shows pairwise overlaps in a compact way. It does not show the size of the sets or the intersection counts for all subsets, but it is more scalable and can be used to quickly identify which runs are more similar to each other.

```python
from gradiend.visualizer.topk.pairwise_heatmap import plot_topk_overlap_heatmap
plot_topk_overlap_heatmap(
    models,
    topk=1000,
    part="decoder-weight",
    value="intersection_frac",
    output_path="topk_overlap_heatmap.pdf",
)
```

- **models**: Dict[str, ModelWithGradiend] — **dict keys are the axis labels**. Use display labels (e.g. run_id or ``"3SG ↔ 3PL"``) as keys; use the same keying for the Venn diagram so labels match.
- **topk**: Keep this many (or this fraction of) top weights per model.
- **part**: Which importance to use (e.g. `"decoder-weight"`).
- **output_path**: Where to save the figure (e.g. under `experiment_dir`).

High values (e.g. bright cells) mean the two runs share many of their top-k parameters; low values mean they focus on different parts of the model.

*Run the code above to generate the heatmap; use `output_path="topk_overlap_heatmap.pdf"` (or `.png`) to save.*


---

## Next steps

- [Tutorial: Evaluation (intra-model)](evaluation-intra-model.md) — Encoder/decoder evaluation and decoder config selection.
- [Tutorial: Model Rewrite](model-rewrite.md) — Apply decoder-selected rewrites and save changed checkpoints.
- [Evaluation & visualization](../guides/evaluation-visualization.md) — Heatmap and Venn customization options.
- [API reference](../api/index.md) — `plot_topk_overlap_heatmap`, `plot_topk_overlap_venn`.
