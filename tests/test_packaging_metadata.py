from pathlib import Path


def _toml_array(name: str, *, after: str = "") -> list[str]:
    text = Path("pyproject.toml").read_text(encoding="utf-8")
    if after:
        text = text[text.index(after):]
    start = text.index(f"{name} = [")
    end = text.index("]", start)
    block = text[start:end]
    return [
        line.strip().strip('",')
        for line in block.splitlines()
        if line.strip().startswith('"')
    ]


def test_accelerate_is_recommended_for_device_map_loading():
    core_deps = _toml_array("dependencies")
    recommended_deps = _toml_array("recommended", after="[project.optional-dependencies]")

    assert not any(dep.startswith("accelerate") for dep in core_deps)
    assert any(dep.startswith("accelerate>=") for dep in recommended_deps)
