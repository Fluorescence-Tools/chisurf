#!/usr/bin/env python
"""Convert manual.docx to RST files in docs/manual/ then update the toctree.

Usage:
    python build_tools/docs/convert_manual.py

Requires python-docx.
"""

import pathlib
import re
import shutil
import subprocess
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
MANUAL_PATH = REPO_ROOT / "docs" / "_old_manual" / "manual.docx"
OUTPUT_DIR = REPO_ROOT / "docs" / "manual"

sys.path.insert(0, str(REPO_ROOT))

try:
    from docx import Document
    from docx.opc.constants import RELATIONSHIP_TYPE as RT
except ImportError:
    print("python-docx required: pip install python-docx")
    sys.exit(1)

RST_HEADING = {0: "#", 1: "=", 2: "-", 3: "~", 4: '"', 5: "^", 6: "'"}


def extract_images(doc, output_dir):
    images_dir = output_dir / "_images"
    images_dir.mkdir(parents=True, exist_ok=True)
    image_rels = {}
    for rel in doc.part.rels.values():
        if rel.reltype == RT.IMAGE:
            data = rel.target_part.blob
            ct = rel.target_part.content_type
            ext = {
                "image/png": ".png",
                "image/jpeg": ".jpg",
                "image/gif": ".gif",
                "image/bmp": ".bmp",
                "image/tiff": ".tif",
                "image/x-emf": ".emf",
                "image/x-wmf": ".wmf",
            }.get(ct, ".png")
            name = f"image_{rel.rId}{ext}"
            path = images_dir / name
            path.write_bytes(data)
            # For EMF/WMF, attempt conversion to PNG for browser display.
            # The original vector file is kept as source.
            if ext in (".emf", ".wmf"):
                png_path = convert_vector_to_png(path, images_dir)
                if png_path:
                    image_rels[rel.rId] = f"_images/{png_path.name}"
                else:
                    image_rels[rel.rId] = f"_images/{name}"
            else:
                image_rels[rel.rId] = f"_images/{name}"
    return image_rels


def convert_vector_to_png(vector_path, images_dir):
    """Convert EMF/WMF vector image to PNG for browser display.

    Tries LibreOffice (soffice) first, then ImageMagick (magick/convert).
    Keeps the original vector file as source. After conversion, trims
    surrounding whitespace since LO renders on an A4-sized canvas.
    Returns the PNG Path on success, or None if all converters fail.
    """
    import shutil
    import subprocess

    png_path = images_dir / f"{vector_path.stem}.png"

    converters = []

    soffice = shutil.which("soffice")
    if soffice:
        converters.append(("libreoffice", soffice))

    magick = shutil.which("magick") or shutil.which("convert")
    if magick:
        converters.append(("imagemagick", magick))

    for name, cmd in converters:
        try:
            if name == "libreoffice":
                subprocess.run(
                    [cmd, "--headless", "--convert-to", "png",
                     "--outdir", str(images_dir), str(vector_path)],
                    check=True, capture_output=True, timeout=120,
                )
                if png_path.exists():
                    _trim_png_whitespace(png_path)
                    return png_path
            elif name == "imagemagick":
                subprocess.run(
                    [cmd, str(vector_path), str(png_path)],
                    check=True, capture_output=True, timeout=120,
                )
                if png_path.exists():
                    _trim_png_whitespace(png_path)
                    return png_path
        except Exception:
            continue

    print(f"  Warning: no converter found for {vector_path.name}")
    print(f"           Install LibreOffice (brew install --cask libreoffice)")
    return None


def _trim_png_whitespace(png_path):
    """Trim surrounding white/off-white margins from a PNG in-place."""
    try:
        from PIL import Image
        import numpy as np
    except ImportError:
        return
    im = Image.open(png_path).convert("RGB")
    arr = np.array(im)
    # Mask: pixels within tolerance of white (255,255,255)
    mask = np.all(arr > 225, axis=2)
    rows = np.any(~mask, axis=1)
    cols = np.any(~mask, axis=0)
    if not rows.any():
        return
    y_min, y_max = rows.argmax(), len(rows) - rows[::-1].argmax()
    x_min, x_max = cols.argmax(), len(cols) - cols[::-1].argmax()
    if x_min < x_max and y_min < y_max:
        im.crop((x_min, y_min, x_max, y_max)).save(png_path)


CODE_STYLES = {"Code block", "HTML Preformatted"}


def indent_block(text, indent=3):
    pad = " " * indent
    return "\n".join(pad + line if line.strip() else "" for line in text.split("\n"))


def merge_code_blocks(content):
    """Merge adjacent code blocks of the same language into one block."""
    result = []
    i = 0
    while i < len(content):
        item = content[i]
        m = re.match(r"^\.\. code-block:: (\w+)\n\n", item)
        if m:
            lang = m.group(1)
            codes = [item[m.end():].strip()]
            j = i + 1
            while j < len(content):
                m2 = re.match(r"^\.\. code-block:: (\w+)\n\n", content[j])
                if m2 and m2.group(1) == lang:
                    codes.append(content[j][m2.end():].strip())
                    j += 1
                else:
                    break
            merged = "\n".join(codes)
            result.append(f".. code-block:: {lang}\n\n{indent_block(merged)}\n")
            i = j
        else:
            result.append(item)
            i += 1
    return result


def clean_text(text):
    """Normalize common DOCX artifacts in paragraph text."""
    # Normalize smart/curly quotes to ASCII
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    # Collapse multiple spaces (but keep intentional double spaces after period)
    text = re.sub(r" {3,}", "  ", text)
    return text.strip()


def para_to_rst(p, image_rels):
    W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    A_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"
    R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    V_NS = "urn:schemas-microsoft-com:vml"

    style = p.style.name if p.style else ""

    # Code / preformatted blocks — skip empty to avoid blank code blocks
    if style in CODE_STYLES:
        text = p.text.strip()
        if not text:
            return ""
        lang = "python" if style == "Code block" else "none"
        # Normalize smart quotes in code blocks to avoid lexer errors
        text = text.replace("\u2018", "'").replace("\u2019", "'")
        text = text.replace("\u201c", '"').replace("\u201d", '"')
        return f".. code-block:: {lang}\n\n{indent_block(text)}\n"

    # Collect annotated segments from all runs, preserving images inline.
    segments = []
    seen_in_para = set()
    for run in p.runs:
        imgs = []
        # Standard drawing images (a:blip inside w:drawing)
        for drawing in run._element.iter(f"{{{W_NS}}}drawing"):
            blip = drawing.find(f".//{{{A_NS}}}blip")
            if blip is not None:
                rid = blip.get(f"{{{R_NS}}}embed")
                if rid in image_rels and rid not in seen_in_para:
                    seen_in_para.add(rid)
                    imgs.append(rid)
        # VML images (v:imagedata, used for EMF/WMF files)
        for imagedata in run._element.iter(f"{{{V_NS}}}imagedata"):
            rid = imagedata.get(f"{{{R_NS}}}id")
            if rid in image_rels and rid not in seen_in_para:
                seen_in_para.add(rid)
                imgs.append(rid)
        t = run.text
        if imgs:
            if t.strip():
                segments.append(("text", t, run.bold, run.italic))
            for rid in imgs:
                segments.append(("image", rid))
        else:
            segments.append(("text", t, run.bold, run.italic))

    # Also check for VML images at paragraph level (outside any run)
    for imagedata in p._element.iter(f"{{{V_NS}}}imagedata"):
        rid = imagedata.get(f"{{{R_NS}}}id")
        if rid in image_rels and rid not in seen_in_para:
            seen_in_para.add(rid)
            segments.append(("image", rid))

    # Merge consecutive text segments with identical (bold, italic).
    merged = []
    for seg in segments:
        if seg[0] == "image":
            merged.append(seg)
        else:
            if merged and merged[-1][0] == "text" and merged[-1][2] == seg[2] and merged[-1][3] == seg[3]:
                merged[-1] = ("text", merged[-1][1] + seg[1], seg[2], seg[3])
            else:
                merged.append(seg)

    # Build RST string.
    # Use interpreted text roles (:strong:, :emphasis:) for markup —
    # they are robust to surrounding punctuation and avoid the fragile
    # `*`-based inline rules.  Roles use backticks for delimiters;
    # DOCX text almost never contains backticks.
    parts = []
    for i, seg in enumerate(merged):
        if seg[0] == "image":
            parts.append(f"\n\n.. image:: {image_rels[seg[1]]}\n   :align: center\n\n")
        else:
            t = seg[1]
            if i > 0 and merged[i - 1][0] == "image":
                t = t.lstrip()
            if t:
                leading = trailing = ""
                m = re.match(r"^(\s*)(.*?)(\s*)$", t, re.DOTALL)
                if m:
                    leading, t, trailing = m.groups()
                if not t:
                    parts.append(leading + trailing)
                    continue
                has_backtick = "`" in t
                if seg[2] and seg[3]:
                    t = f":strong:`{t}`" if not has_backtick else f"***{t}***"
                elif seg[2]:
                    t = f":strong:`{t}`" if not has_backtick else f"**{t}**"
                elif seg[3]:
                    t = f":emphasis:`{t}`" if not has_backtick else f"*{t}*"
                # If the next text segment starts immediately with a char
                # that RST disallows after inline-markup end-strings
                # (e.g. `=`, `(`), escape it with `\` so the end-string
                # is recognized (backslash is an allowed follow-char).
                if (seg[2] or seg[3]) and i + 1 < len(merged):
                    nxt = merged[i + 1]
                    if nxt[0] == "text" and nxt[1] and not nxt[1][0].isspace():
                        fc = nxt[1][0]
                        if fc in "=<>([{":
                            merged[i + 1] = ("text", "\\" + nxt[1], nxt[2], nxt[3])
                parts.append(leading + t + trailing)
    text = "".join(parts)

    # List-item prefix.
    if style.startswith("List"):
        level = 0
        m = re.search(r"\d+$", style)
        if m:
            level = int(m.group()) - 1
        indent = "  " * level
        prefix = "*" if "Bullet" in style else "#."
        text = f"{indent}{prefix} {text}"
    return text


def _table_images(table_elem, image_rels):
    """Extract image references from all cells in a table."""
    W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    A_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"
    R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    V_NS = "urn:schemas-microsoft-com:vml"

    seen = set()
    results = []
    # Find all a:blip elements inside the table
    for blip in table_elem.iter(f"{{{A_NS}}}blip"):
        rid = blip.get(f"{{{R_NS}}}embed")
        if rid in image_rels and rid not in seen:
            seen.add(rid)
            results.append(f"\n.. image:: {image_rels[rid]}\n   :align: center\n")
    # Also check for VML images
    for imagedata in table_elem.iter(f"{{{V_NS}}}imagedata"):
        rid = imagedata.get(f"{{{R_NS}}}id")
        if rid in image_rels and rid not in seen:
            seen.add(rid)
            results.append(f"\n.. image:: {image_rels[rid]}\n   :align: center\n")
    return results


def extract_structure(doc, image_rels):
    W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"

    structure = []
    current = {"level": 0, "title": "Introduction", "content": []}

    # Iterate body children in document order to correctly interleave
    # body paragraphs (direct w:p children) with tables (w:tbl children).
    para_iter = iter(doc.paragraphs)
    for child in doc.element.body.iterchildren():
        tag = child.tag

        if tag == f"{{{W_NS}}}p":
            # Match this body-level w:p with the next doc.paragraph
            try:
                p = next(para_iter)
            except StopIteration:
                continue
            s = p.style.name if p.style else ""
            if s.startswith("Heading"):
                if current["content"]:
                    structure.append(current)
                level = int(s.replace("Heading ", ""))
                current = {"level": level, "title": p.text, "content": []}
            else:
                rst = para_to_rst(p, image_rels)
                if rst.strip():
                    current["content"].append(rst)

        elif tag == f"{{{W_NS}}}tbl":
            imgs = _table_images(child, image_rels)
            current["content"].extend(imgs)

    if current["content"]:
        structure.append(current)
    return structure


def slug(title):
    s = title.lower().replace(" ", "_").replace("/", "_")
    s = re.sub(r"[^\w_]", "", s)
    return s or "section"


def write_rst(heading, output_dir, image_rels, seen_slugs):
    content = merge_code_blocks(heading["content"])
    content = [clean_text(p) for p in content if p.strip()]
    name = slug(heading["title"])
    if name in seen_slugs:
        seen_slugs[name] += 1
        name = f"{name}_{seen_slugs[name]}"
    else:
        seen_slugs[name] = 1
    path = output_dir / f"{name}.rst"
    lines = [heading["title"], RST_HEADING[heading["level"]] * len(heading["title"]), ""]
    for p in content:
        lines.append(p)
        lines.append("")
    # Collapse runs of blank lines to at most one.
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    path.write_text(text, encoding="utf-8")
    return name


def main():
    print(f"Reading {MANUAL_PATH} ...")
    doc = Document(MANUAL_PATH)
    print("Extracting images ...")
    image_rels = extract_images(doc, OUTPUT_DIR)
    print(f"  {len(image_rels)} images")
    print("Extracting structure ...")
    structure = extract_structure(doc, image_rels)
    print(f"  {len(structure)} sections")
    print("Writing RST files ...")
    seen_slugs = {}
    names = []
    for h in structure:
        name = write_rst(h, OUTPUT_DIR, image_rels, seen_slugs)
        names.append((name, h["title"]))
    # Write manual index — deduplicate by slug
    seen = set()
    unique = []
    for name, title in names:
        if name not in seen:
            seen.add(name)
            unique.append((name, title))
    idx = [
        "ChiSurf Manual",
        "==============",
        "",
        ".. toctree::",
        "   :maxdepth: 2",
        "",
    ]
    for name, title in unique:
        idx.append(f"   {title} <{name}>")
    idx.append("")
    (OUTPUT_DIR / "index.rst").write_text("\n".join(idx), encoding="utf-8")
    print(f"Written {len(names)} RST files + index.rst to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
