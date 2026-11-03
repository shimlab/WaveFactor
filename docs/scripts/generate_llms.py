#!/usr/bin/env python3
"""
Generates llms.md by concatenating documentation pages in mkdocs.yml nav order
and converting the mkdocstrings-rendered API Reference HTML into clean, AI-optimized Markdown.
"""

from html.parser import HTMLParser
from pathlib import Path
import re
import subprocess
import sys
from typing import List, Tuple

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


class MkdocsHTMLToMarkdown(HTMLParser):
    """Converts the main article HTML from MkDocs/mkdocstrings into clean, AI-friendly Markdown."""

    def __init__(self):
        super().__init__()
        self.output = []
        self.in_pre = False
        self.in_ignored = False
        self.ignored_depth = 0
        self.list_depth = 0

        # Heading state
        self.in_heading = False
        self.heading_level = 0
        self.heading_text = []
        self.heading_badges = []
        self.in_badge = False
        self.badge_depth = 0
        self.badge_text = []

        # Table state
        self.in_table = False
        self.in_thead = False
        self.table_row = []
        self.in_cell = False
        self.cell_parts = []

    def handle_starttag(self, tag, attrs):
        attr_dict = dict(attrs)
        classes = attr_dict.get("class", "").split()

        # Ignore permalinks, headerlinks, source code links
        if any(c in classes for c in ["headerlink", "doc-github-link", "doc-permalink"]):
            self.in_ignored = True
            self.ignored_depth += 1
            return
        if self.in_ignored:
            self.ignored_depth += 1
            return

        # Badges (e.g. property, instance-attribute, classmethod)
        if any(c.startswith("doc-label") for c in classes) or "doc-labels" in classes:
            self.in_badge = True
            self.badge_depth += 1
            return
        if self.in_badge:
            self.badge_depth += 1
            return

        # Headings
        if tag in ["h1", "h2", "h3", "h4", "h5", "h6"]:
            self.in_heading = True
            self.heading_level = int(tag[1])
            self.heading_text = []
            self.heading_badges = []
            return

        if self.in_heading:
            if tag == "code":
                self.heading_text.append("`")
            return

        # Tables
        if tag == "table":
            self.in_table = True
            self.output.append("\n\n")
            return
        elif tag == "thead":
            self.in_thead = True
            return
        elif tag == "tr":
            self.table_row = []
            return
        elif tag in ["th", "td"]:
            self.in_cell = True
            self.cell_parts = []
            return

        if self.in_cell:
            if tag == "code":
                self.cell_parts.append("`")
            elif tag in ["strong", "b"]:
                self.cell_parts.append("**")
            elif tag in ["em", "i"]:
                self.cell_parts.append("*")
            return

        # Block & inline elements outside headings/cells
        if tag == "pre":
            self.in_pre = True
            self.output.append("\n\n```python\n")
        elif tag == "code" and not self.in_pre:
            self.output.append("`")
        elif tag in ["strong", "b"]:
            self.output.append("**")
        elif tag in ["em", "i"]:
            self.output.append("*")
        elif tag == "p":
            self.output.append("\n\n")
        elif tag == "hr":
            self.output.append("\n\n---\n\n")
        elif tag in ["ul", "ol"]:
            self.list_depth += 1
            self.output.append("\n")
        elif tag == "li":
            indent = "  " * max(0, self.list_depth - 1)
            self.output.append(f"\n{indent}- ")

    def handle_endtag(self, tag):
        if self.in_ignored:
            self.ignored_depth -= 1
            if self.ignored_depth <= 0:
                self.in_ignored = False
                self.ignored_depth = 0
            return

        if self.in_badge:
            self.badge_depth -= 1
            if self.badge_depth <= 0:
                txt = "".join(self.badge_text).strip()
                if txt:
                    if self.in_heading:
                        self.heading_badges.append(txt)
                    else:
                        self.output.append(f" *({txt})*")
                self.in_badge = False
                self.badge_depth = 0
                self.badge_text = []
            return

        if self.in_heading:
            if tag in ["h1", "h2", "h3", "h4", "h5", "h6"]:
                raw = "".join(self.heading_text).strip()
                # Clean multiple spaces & normalize punctuation
                raw = re.sub(r"[ \t]+", " ", raw)
                raw = re.sub(r"`\s+", "`", raw)
                raw = re.sub(r"\s+`", "`", raw)
                raw = re.sub(r"`\s*`", " ", raw)
                # Clean function signature spacing like fit ( X , coords ) -> fit(X, coords)
                raw = re.sub(r"\(\s+", "(", raw)
                raw = re.sub(r"\s+\)", ")", raw)
                raw = re.sub(r"\s*,\s*", ", ", raw)
                raw = re.sub(r"\s*=\s*", "=", raw)
                if raw.startswith("`") and not raw.endswith("`"):
                    raw += "`"
                elif not raw.startswith("`") and raw.endswith("`"):
                    raw = "`" + raw

                # Clean instance attribute assignments like elbo=float(elbo) -> elbo
                if any("attribute" in b for b in self.heading_badges):
                    m = re.match(r"^`?([a-zA-Z_][a-zA-Z0-9_]*)=.*", raw)
                    if m:
                        raw = f"`{m.group(1)}`"

                badges_str = ""
                if self.heading_badges:
                    badges_str = " " + " ".join([f"*({b})*" for b in self.heading_badges])

                hashes = "#" * self.heading_level
                self.output.append(f"\n\n{hashes} {raw}{badges_str}\n\n")
                self.in_heading = False
                self.heading_text = []
                self.heading_badges = []
            elif tag == "code":
                self.heading_text.append("`")
            return

        if self.in_cell:
            if tag in ["th", "td"]:
                raw_cell = "".join(self.cell_parts).strip()
                raw_cell = re.sub(r"\s+", " ", raw_cell)
                raw_cell = raw_cell.replace("|", "\\|")
                self.table_row.append(raw_cell)
                self.in_cell = False
                self.cell_parts = []
            elif tag == "code":
                self.cell_parts.append("`")
            elif tag in ["strong", "b"]:
                self.cell_parts.append("**")
            elif tag in ["em", "i"]:
                self.cell_parts.append("*")
            return

        if tag == "thead":
            self.in_thead = False
        elif tag == "tr":
            if self.table_row:
                row_str = "| " + " | ".join(self.table_row) + " |"
                self.output.append(row_str + "\n")
                if self.in_thead:
                    sep = "| " + " | ".join(["---"] * len(self.table_row)) + " |"
                    self.output.append(sep + "\n")
            self.table_row = []
        elif tag == "table":
            self.in_table = False
            self.output.append("\n\n")
        elif tag == "pre":
            self.in_pre = False
            self.output.append("\n```\n\n")
        elif tag == "code" and not self.in_pre:
            self.output.append("`")
        elif tag in ["strong", "b"]:
            self.output.append("**")
        elif tag in ["em", "i"]:
            self.output.append("*")
        elif tag == "p":
            self.output.append("\n\n")
        elif tag in ["ul", "ol"]:
            self.list_depth = max(0, self.list_depth - 1)
            self.output.append("\n")

    def handle_data(self, data):
        if self.in_ignored:
            return
        if self.in_badge:
            self.badge_text.append(data)
        elif self.in_heading:
            self.heading_text.append(data)
        elif self.in_cell:
            self.cell_parts.append(data)
        else:
            self.output.append(data)


def clean_markdown_formatting(md: str) -> str:
    """Cleans up table rows, collapses excess whitespace, and tightens markdown for LLMs."""
    lines = md.splitlines()
    cleaned = []
    in_table = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            in_table = True
            cleaned.append(stripped)
        elif in_table and stripped == "":
            continue  # remove blank lines inside markdown tables
        elif stripped == "":
            in_table = False
            if cleaned and cleaned[-1] != "":
                cleaned.append("")
        else:
            if in_table:
                # Blank line after table before following text
                cleaned.append("")
            in_table = False
            cleaned.append(stripped)

    result = "\n".join(cleaned)
    # Ensure blank line before section labels like Returns:, Parameters:, Attributes:
    result = re.sub(r"([^\n])\n(Parameters:|Returns:|Attributes:)", r"\1\n\n\2", result)
    # Collapse 3+ newlines into 2
    result = re.sub(r"\n{3,}", "\n\n", result)
    # Clean up empty code spans like ``
    result = re.sub(r"``", "", result)
    return result.strip()


def extract_api_markdown_from_site(site_dir: Path) -> str:
    """Extracts and converts the rendered API HTML from mkdocs build output."""
    api_html_path = site_dir / "api" / "index.html"
    if not api_html_path.is_file():
        raise FileNotFoundError(
            f"Rendered API HTML not found at {api_html_path}. Run 'mkdocs build' first."
        )

    html = api_html_path.read_text(encoding="utf-8")
    match = re.search(
        r"<article[^>]*class=\"[^\"]*md-content__inner[^\"]*\"[^>]*>(.*?)</article>",
        html,
        re.DOTALL,
    )
    if not match:
        raise ValueError(f"Could not locate main <article> in {api_html_path}.")

    parser = MkdocsHTMLToMarkdown()
    parser.feed(match.group(1))
    raw_md = "".join(parser.output)
    return clean_markdown_formatting(raw_md)


def get_nav_files(repo_root: Path) -> List[Tuple[str, str]]:
    """Reads mkdocs.yml and extracts navigation titles and file paths in order."""
    mkdocs_path = repo_root / "mkdocs.yml"
    with open(mkdocs_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    nav = config.get("nav", [])
    pages = []
    for item in nav:
        if isinstance(item, dict):
            for title, path in item.items():
                if isinstance(path, str) and path.endswith(".md"):
                    pages.append((title, path))
    return pages


def build_llms_md(repo_root: Path):
    """Builds site/llms.md following mkdocs.yml nav order using rendered mkdocs output."""
    docs_dir = repo_root / "docs"
    site_dir = repo_root / "site"
    api_html = site_dir / "api" / "index.html"

    # If site/api/index.html doesn't exist, run mkdocs build
    if not api_html.is_file():
        print("[INFO] Building mkdocs site to generate rendered API documentation...")
        subprocess.run(["mkdocs", "build"], cwd=repo_root, check=True)

    nav_files = get_nav_files(repo_root)

    header = (
        "# WaveFactor LLM Documentation Reference\n\n"
        "> Single-file documentation reference optimized for LLM context windows and developer agents."
    )
    content_blocks = [header]

    for title, rel_path in nav_files:
        if rel_path == "api.md":
            api_md = extract_api_markdown_from_site(site_dir)
            content_blocks.append(api_md)
        else:
            src_file = docs_dir / rel_path
            if not src_file.is_file():
                print(f"[WARNING] Nav file not found: {src_file}", file=sys.stderr)
                continue
            raw_content = src_file.read_text(encoding="utf-8").strip()
            # Strip redundant single-file LLM context pointer at bottom of index.md if present
            if rel_path == "index.md":
                lines = [
                    l for l in raw_content.splitlines()
                    if not l.startswith("> **LLM Context**:")
                ]
                raw_content = "\n".join(lines).rstrip()

            content_blocks.append(raw_content)

    full_text = "\n\n---\n\n".join(content_blocks).rstrip() + "\n"

    # Write to site/llms.md for deployment
    site_dir.mkdir(parents=True, exist_ok=True)
    site_llms = site_dir / "llms.md"
    site_llms.write_text(full_text, encoding="utf-8")
    print(f"[SUCCESS] Written {site_llms} ({len(full_text):,} bytes).")


if __name__ == "__main__":
    build_llms_md(REPO_ROOT)
