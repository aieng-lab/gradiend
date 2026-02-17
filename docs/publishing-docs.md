# Publishing the documentation

The documentation (including the **auto-generated API reference** from docstrings) is built with [MkDocs](https://www.mkdocs.org/) and the Material theme. You can build and publish it so it is **publicly accessible** when the package is released.

## Build locally

From the project root, install dev dependencies and build:

```bash
pip install -e ".[dev]"
pip install mkdocs mkdocs-material
mkdocs build
```

The static site is written to `site/`. Preview it with:

```bash
mkdocs serve
```

Then open [http://127.0.0.1:8000](http://127.0.0.1:8000). The **Reference → API reference** section is generated from the code (docstrings and type hints) via [mkdocstrings](https://mkdocstrings.github.io/).

### Test the same build as CI

CI runs `mkdocs build` (no `--strict`). To match it locally:

```bash
pip install -e ".[dev]"
pip install mkdocs mkdocs-material
mkdocs build
```

To treat warnings as errors, run `mkdocs build --strict`. Passing strict mode requires fixing all reported issues: broken links, missing images, and griffe warnings (missing type/annotation in docstrings). Add type hints and fix docstring parameter docs to clear griffe warnings over time.

## Hosting (public access)

### GitHub Pages

1. In the repo: **Settings → Pages → Source**: choose **GitHub Actions** (or “Deploy from a branch” and select `gh-pages`).
2. Add a workflow under `.github/workflows/` that runs `pip install -e ".[dev]"`, then `mkdocs gh-deploy` (or `mkdocs build` and upload `site/` to the `gh-pages` branch).
3. Set **Project URLs** in `pyproject.toml` so **Documentation** points to the Pages URL, e.g. `https://<org>.github.io/gradiend/`.

Once configured, every push to the docs (or to main) can update the published site. Users and package index will see the documentation link in the project metadata.

### Read the Docs

1. Sign up at [readthedocs.org](https://readthedocs.org/) and import the repository.
2. Set the **Documentation type** to **MkDocs**.
3. Use a **config file** path of `mkdocs.yml` (default).
4. Add `mkdocs-material` and `mkdocstrings[python]` to the **Requirements file** (e.g. create `docs/requirements.txt` with those packages, or add them in the RTD admin).
5. Build. Read the Docs will run `mkdocs build` and host the site.

Then set **Documentation** in `pyproject.toml` to the Read the Docs URL (e.g. `https://gradiend.readthedocs.io/`) so the PyPI project page links to the published docs.

## Linking from the package

In `pyproject.toml`, set:

```toml
[project.urls]
Documentation = "https://your-docs-url/"
```

That URL is shown on [PyPI](https://pypi.org/) and by `pip show gradiend`, so users can reach the same documentation (including the API reference) from the released package.
