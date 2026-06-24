"""Pure-Python Markdown rendering for the Help plugin."""

from __future__ import annotations

import html as _html
import re
from typing import Optional


def render_markdown(text: str) -> Optional[str]:
    """Render Markdown *text* to an HTML fragment (wrapped in a full document).

    Uses the ``markdown`` package if available; otherwise falls back to a
    built-in basic renderer.

    Parameters
    ----------
    text : str
        Raw Markdown source.

    Returns
    -------
    str or None
        Full HTML document string, or *None* if rendering fails.

    """
    prepared = _prepare_markdown_with_heading_ids(text)
    body = _render_with_markdown_lib(prepared)
    if body is None:
        body = _basic_markdown_to_html(prepared)

    css = (
        "<style>"
        "body { font-family: 'Segoe UI', Arial, sans-serif; font-size: 10pt; }"
        "h1, h2, h3 { margin-top: 0.8em; margin-bottom: 0.4em; }"
        "pre, code { font-family: 'Consolas', 'Courier New', monospace; }"
        "pre { padding: 6px; border-radius: 3px; }"
        "</style>"
    )
    return f"<html><head>{css}</head><body>{body}</body></html>"


def extract_title(text: str) -> Optional[str]:
    """Return the first Markdown heading from *text*, or *None*."""
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        m = re.match(r"^(#{1,6})\s+(.*)", stripped)
        if not m:
            continue
        heading = m.group(2)
        heading = re.sub(r"\{\s*#[-\w]+\s*\}\s*$", "", heading).strip()
        if heading:
            return heading
    return None


def slugify_heading(text: str) -> str:
    """Convert a heading string to an HTML-anchor-compatible slug."""
    slug = text.strip().lower()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"\s+", "-", slug)
    slug = re.sub(r"-+", "-", slug)
    return slug or "section"


# ── internal helpers ────────────────────────────────────────────────


def _prepare_markdown_with_heading_ids(text: str) -> str:
    """Ensure every heading has an ``{#id}`` attribute marker."""
    lines = text.splitlines()
    out_lines = []
    for line in lines:
        m = re.match(r"^(#{1,6})\s+(.*)", line)
        if not m:
            out_lines.append(line)
            continue
        prefix, title = m.group(1), m.group(2)
        if re.search(r"\{\s*#[-\w]+\s*\}\s*$", title):
            out_lines.append(line)
            continue
        slug = slugify_heading(title)
        out_lines.append(f"{prefix} {title} " + "{" + f"#{slug}" + "}")
    return "\n".join(out_lines)


def _render_with_markdown_lib(text: str) -> Optional[str]:
    """Render with the ``markdown`` package if available."""
    try:
        import markdown as _md
    except Exception:
        return None
    try:
        return _md.markdown(
            text,
            output_format="html5",
            extensions=["attr_list"],
        )
    except Exception:
        try:
            return _md.markdown(text, output_format="html5")
        except Exception:
            return None


def _basic_markdown_to_html(text: str) -> str:
    """Convert simple Markdown to HTML."""
    lines = text.splitlines()
    html_lines = []

    in_ul = False
    in_ol = False

    def _process_inline_with_images(raw: str) -> str:
        placeholders = {}

        def _img_repl(m):
            alt = m.group(1)
            src = m.group(2).strip()
            key = f"__CS_IMG_{len(placeholders)}__"
            alt_esc = _html.escape(alt)
            src_esc = _html.escape(src, quote=True)
            placeholders[key] = f'<img src="{src_esc}" alt="{alt_esc}">'
            return key

        tmp = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", _img_repl, raw)
        escaped = _html.escape(tmp)
        escaped = _apply_inline_markdown(escaped)
        for key, tag in placeholders.items():
            escaped = escaped.replace(key, tag)
        return escaped

    for line in lines:
        stripped = line.lstrip()
        if not stripped:
            if in_ul:
                html_lines.append("</ul>")
                in_ul = False
            if in_ol:
                html_lines.append("</ol>")
                in_ol = False
            html_lines.append("")
            continue

        m_ul = re.match(r"^[-*+]\s+(.*)", stripped)
        m_ol = re.match(r"^(\d+)[.)]\s+(.*)", stripped)

        if m_ul:
            if in_ol:
                html_lines.append("</ol>")
                in_ol = False
            if not in_ul:
                html_lines.append("<ul>")
                in_ul = True
            content = _process_inline_with_images(m_ul.group(1))
            html_lines.append(f"<li>{content}</li>")
            continue

        if m_ol:
            if in_ul:
                html_lines.append("</ul>")
                in_ul = False
            if not in_ol:
                html_lines.append("<ol>")
                in_ol = True
            content = _process_inline_with_images(m_ol.group(2))
            html_lines.append(f"<li>{content}</li>")
            continue

        if in_ul:
            html_lines.append("</ul>")
            in_ul = False
        if in_ol:
            html_lines.append("</ol>")
            in_ol = False

        m = re.match(r"^(#{1,6})\s+(.*)", stripped)
        if m:
            level = len(m.group(1))
            raw_content = m.group(2)
            anchor = None
            m_id = re.search(r"\{\s*#([-\w]+)\s*\}\s*$", raw_content)
            if m_id:
                anchor = m_id.group(1)
                raw_content = raw_content[:m_id.start()].rstrip()
            else:
                anchor = slugify_heading(raw_content)
            content = _process_inline_with_images(raw_content)
            if anchor:
                html_lines.append(f'<h{level} id="{anchor}">{content}</h{level}>')
            else:
                html_lines.append(f"<h{level}>{content}</h{level}>")
        else:
            content = _process_inline_with_images(line)
            html_lines.append(f"<p>{content}</p>")

    if in_ul:
        html_lines.append("</ul>")
    if in_ol:
        html_lines.append("</ol>")

    return "\n".join(html_lines)


def _apply_inline_markdown(text: str) -> str:
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"\*([^*]+)\*", r"<em>\1</em>", text)
    return text
