"""Native documentation; no framework import is needed to build the site."""

project = "MACE"
author = "MACE developers"
release = "V1 reference (unpublished)"
extensions = ["myst_parser"]
source_suffix = {".md": "markdown"}
root_doc = "index"
html_theme = "sphinx_book_theme"
html_title = "MACE V1 reference"
myst_enable_extensions = ["dollarmath", "colon_fence"]
exclude_patterns = ["_build"]


def link_supporting_sources(app, docname, source):
    """Bundle linked source Markdown that is outside the published guide tree."""
    import re
    from pathlib import Path

    document = Path(app.env.doc2path(docname))
    root = Path(app.srcdir).resolve()

    def replace(match):
        label, target = match.groups()
        if "://" in target or not target.split("#")[0].endswith(".md"):
            return match.group(0)
        resolved = (document.parent / target.split("#")[0]).resolve()
        if not resolved.is_file():
            return match.group(0)  # Let Sphinx diagnose a real broken link.
        try:
            name = resolved.relative_to(root).with_suffix("").as_posix()
        except ValueError:
            name = None
        if name not in app.env.found_docs:
            return f"{{download}}`{label} <{target.split(chr(35))[0]}>`"
        return match.group(0)

    source[0] = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", replace, source[0])


def setup(app):
    app.connect("source-read", link_supporting_sources)
