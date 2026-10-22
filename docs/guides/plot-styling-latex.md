# Plot styling and LaTeX

GRADIEND matplotlib plots share a single style layer: [`configure_plot_style()`][gradiend.visualizer.plot_style.configure_plot_style] with [`PlotStyleConfig`][gradiend.visualizer.plot_style_config.PlotStyleConfig]. It controls LaTeX rendering, optional custom fonts, extra LaTeX preamble lines, and how transition labels such as `M -> F` are rendered.

For per-plot kwargs (colorbars, annotations, figure size), see [Evaluation & visualization](evaluation-visualization.md).

## Quick start

```python
from gradiend import configure_plot_style, PlotStyleConfig

configure_plot_style(PlotStyleConfig(use_latex=True))
```

Or rely on environment variables (applied automatically when the first plot imports matplotlib):

```bash
export GRADIEND_PLOT_USE_LATEX=auto   # default: enable when LaTeX works
export GRADIEND_PLOT_TRANSITION_ARROWS=latex
```

Verify your setup:

```python
from gradiend import check_plot_environment

check_plot_environment()
```

## Environment variables

| Variable | Values | Default | Purpose |
|----------|--------|---------|---------|
| `GRADIEND_PLOT_USE_LATEX` | `auto`, `1`/`true`, `0`/`false` | `auto` | Enable matplotlib `text.usetex` when LaTeX is usable |
| `GRADIEND_PLOT_FONT_PATH` | path to `.ttf`/`.otf` | unset | Register a custom sans-serif font |
| `GRADIEND_PLOT_LATEX_PREAMBLE_EXTRA` | LaTeX preamble lines | unset | Your extra `\usepackage{...}` lines |
| `GRADIEND_PLOT_TRANSITION_ARROWS` | `auto`, `latex`, `unicode`, `ascii` | `auto` | Arrow style for `A -> B` / `A <-> B` labels |

When `text.usetex` is enabled, GRADIEND **always** appends `\usepackage{amsmath}` and `\usepackage{amssymb}` (required for `\rightleftarrows` on transition labels). Put only your own extra packages in `GRADIEND_PLOT_LATEX_PREAMBLE_EXTRA`.

## Transition label helpers

Heatmaps and comparison plots call [`format_transition_label()`][gradiend.visualizer.labels.format_transition_label] so raw ids like `masc_nom -> fem_nom` render consistently:

```python
from gradiend import format_transition_label, transition_bidi_arrow

format_transition_label("he -> she")          # follows active style
transition_bidi_arrow(use_latex=True)         # r"$\rightleftarrows$"
```

With `GRADIEND_PLOT_TRANSITION_ARROWS=ascii`, labels keep plain `->` / `<->` delimiters. With `unicode` (or `auto` when LaTeX is off), you get `→` / `↔`. With `latex` (or `auto` when LaTeX is on), arrows are wrapped in inline math.

## Programmatic configuration

```python
from gradiend import PlotStyleConfig, configure_plot_style

status = configure_plot_style(
    PlotStyleConfig(
        use_latex="auto",
        latex_preamble_extra=r"\usepackage{textcomp}",
        font_path="/path/to/MyFont.ttf",
        transition_arrows="auto",
    ),
    force=True,
)
print(status.text_usetex, status.transition_arrows)
```

`configure_plot_style()` is idempotent: the first matplotlib plot in a process applies env defaults unless you pass `force=True`.

## Troubleshooting

**`Undefined control sequence \rightleftarrows`**

Install a LaTeX distribution that includes `amsmath` and `amssymb` (e.g. `texlive-latex-recommended` on Debian/Ubuntu). GRADIEND adds both packages automatically; if the error persists, run [`check_plot_environment()`][gradiend.visualizer.plot_style.check_plot_environment] and confirm `resolved_text_usetex=True` only when the LaTeX smoke test passes.

**Tick labels show broken fragments like `Encoding (`**

Axis text that is not intentional math is rendered with `usetex=False` per label while math segments (`$...$`) keep LaTeX. If you force raw `%` or underscores in labels, escape them or disable LaTeX for that plot.

**Deprecation**

`configure_matplotlib_style()` remains as a thin alias but emits `DeprecationWarning`; prefer `configure_plot_style()`.

## Related

- [`check_plot_environment`][gradiend.visualizer.plot_style.check_plot_environment] — diagnostic summary
- [Evaluation & visualization](evaluation-visualization.md) — plot catalog and kwargs
- [API: PlotStyleConfig](../api/visualization/PlotStyleConfig.md)
