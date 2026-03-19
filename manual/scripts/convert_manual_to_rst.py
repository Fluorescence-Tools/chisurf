#!/usr/bin/env python
"""
Script to convert manual.docx to reStructuredText format for Sphinx documentation.

This script reads the manual.docx file and converts its content to reStructuredText format,
which is the format used by Sphinx for documentation. The converted content is organized
into separate .rst files according to the document's structure.

Requirements:
- python-docx: for reading Word documents
- docutils: for reStructuredText processing

Usage:
    python convert_manual_to_rst.py

The script will:
1. Read the manual.docx file
2. Extract its content and structure
3. Convert the content to reStructuredText format
4. Create separate .rst files for each section
5. Organize the files in the docs directory
"""

import os
import sys
import re
import pathlib
from typing import List, Dict, Tuple, Optional

# Add the parent directory to the Python path
script_dir = pathlib.Path(__file__).parent.absolute()
repo_root = script_dir.parent.parent
sys.path.insert(0, str(repo_root))

try:
    from docx import Document
    from docx.opc.constants import RELATIONSHIP_TYPE as RT
    from docx.oxml.shared import qn
except ImportError:
    print("python-docx package not found. Please install it with:")
    print("pip install python-docx")
    sys.exit(1)

# Define paths
MANUAL_PATH = repo_root / "manual" / "manual.docx"
DOCS_PATH = repo_root / "docs"

# Define mapping of heading levels to RST heading styles
RST_HEADING_STYLES = {
    0: "#",  # ########
    1: "=",  # ========
    2: "-",  # --------
    3: "~",  # ~~~~~~~~
    4: "\"", # """"""""
    5: "^",  # ^^^^^^^^
    6: "'",  # ''''''''
}

def extract_images(doc_path: pathlib.Path, output_dir: pathlib.Path) -> Dict[str, str]:
    """
    Extract images from the Word document and save them to the output directory.

    Args:
        doc_path: Path to the Word document
        output_dir: Directory where images will be saved

    Returns:
        Dictionary mapping image relationship IDs to file paths
    """
    # Create images directory if it doesn't exist
    images_dir = output_dir / "_static" / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Open the document
    doc = Document(doc_path)

    # Get the document part
    document_part = doc.part

    # Dictionary to store image relationship IDs and their file paths
    image_rels = {}

    # Extract images from the document
    for rel in document_part.rels.values():
        if rel.reltype == RT.IMAGE:
            # Get the image data
            image_data = rel.target_part.blob

            # Generate a filename based on the relationship ID
            rel_id = rel.rId
            image_filename = f"image_{rel_id}.png"
            image_path = images_dir / image_filename

            # Save the image
            with open(image_path, "wb") as f:
                f.write(image_data)

            # Store the relationship ID and file path
            image_rels[rel_id] = str(image_path.relative_to(output_dir))

    return image_rels

def convert_paragraph_to_rst(paragraph, image_rels: Dict[str, str]) -> str:
    """
    Convert a paragraph to reStructuredText format.

    Args:
        paragraph: A paragraph from the Word document
        image_rels: Dictionary mapping image relationship IDs to file paths

    Returns:
        The paragraph in reStructuredText format
    """
    # Get the text and formatting
    text = ""
    for run in paragraph.runs:
        # Check if this run contains an image
        try:
            # Handle namespace properly
            w_namespace = "{http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing}"
            a_namespace = "{http://schemas.openxmlformats.org/drawingml/2006/main}"

            # Look for drawing elements
            for element in run._element.findall(f".//{w_namespace}drawing"):
                # Find the relationship ID for the image
                blip = element.find(f".//{a_namespace}blip")
                if blip is not None:
                    rel_id = blip.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed")
                    if rel_id in image_rels:
                        # Add image reference
                        image_path = image_rels[rel_id]
                        text += f"\n\n.. image:: {image_path}\n   :align: center\n\n"
        except Exception as e:
            # If there's an error processing images, just continue with the text
            print(f"Warning: Error processing image in paragraph: {str(e)}")

        # Add text with formatting
        run_text = run.text
        if run.bold:
            run_text = f"**{run_text}**"
        if run.italic:
            run_text = f"*{run_text}*"
        text += run_text

    # Handle list items
    if paragraph.style.name.startswith("List"):
        # Determine list level and type
        level = 0
        list_type = "bullet"
        if "Bullet" in paragraph.style.name:
            list_type = "bullet"
        elif "Number" in paragraph.style.name:
            list_type = "number"

        # Extract level from style name (e.g., "List Bullet 2" -> level 2)
        match = re.search(r"\d+$", paragraph.style.name)
        if match:
            level = int(match.group(0)) - 1

        # Format as list item
        indent = "  " * level
        if list_type == "bullet":
            text = f"{indent}* {text}"
        else:
            text = f"{indent}#. {text}"

    return text

def extract_headings_and_content(doc_path: pathlib.Path, image_rels: Dict[str, str]) -> List[Dict]:
    """
    Extract headings and content from the Word document.

    Args:
        doc_path: Path to the Word document
        image_rels: Dictionary mapping image relationship IDs to file paths

    Returns:
        List of dictionaries containing heading information and content
    """
    # Open the document
    doc = Document(doc_path)

    # List to store document structure
    structure = []

    # Current heading and its content
    current_heading = {"level": 0, "title": "Introduction", "content": []}

    # Process paragraphs
    for paragraph in doc.paragraphs:
        # Check if this is a heading
        if paragraph.style.name.startswith("Heading"):
            # Extract heading level
            level = int(paragraph.style.name.replace("Heading ", ""))

            # If we have content for the previous heading, add it to the structure
            if current_heading["content"]:
                structure.append(current_heading)

            # Create a new heading
            current_heading = {
                "level": level,
                "title": paragraph.text,
                "content": []
            }
        else:
            # Add paragraph to current heading's content
            rst_paragraph = convert_paragraph_to_rst(paragraph, image_rels)
            if rst_paragraph.strip():
                current_heading["content"].append(rst_paragraph)

    # Add the last heading
    if current_heading["content"]:
        structure.append(current_heading)

    return structure

def create_rst_files(structure: List[Dict], output_dir: pathlib.Path) -> Dict[str, str]:
    """
    Create reStructuredText files from the document structure.

    Args:
        structure: List of dictionaries containing heading information and content
        output_dir: Directory where RST files will be created

    Returns:
        Dictionary mapping section titles to file paths
    """
    # Dictionary to store section titles and their file paths
    section_files = {}

    # Process each heading
    for heading in structure:
        # Create a filename from the heading title
        filename = heading["title"].lower().replace(" ", "_").replace("/", "_")
        filename = re.sub(r"[^\w_]", "", filename)
        if not filename:
            filename = "section"
        filename = f"{filename}.rst"

        # Create the file path
        file_path = output_dir / filename

        # Create the RST content
        rst_content = []

        # Add the heading
        rst_content.append(heading["title"])
        rst_content.append(RST_HEADING_STYLES[heading["level"]] * len(heading["title"]))
        rst_content.append("")

        # Add the content
        for paragraph in heading["content"]:
            rst_content.append(paragraph)
            rst_content.append("")

        # Write the RST file
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(rst_content))

        # Store the section title and file path
        section_files[heading["title"]] = filename

    return section_files

def update_index_rst(section_files: Dict[str, str], output_dir: pathlib.Path):
    """
    Update the index.rst file with the sections from the manual.

    Args:
        section_files: Dictionary mapping section titles to file paths
        output_dir: Directory where the index.rst file is located
    """
    # Read the existing index.rst file
    index_path = output_dir / "index.rst"
    with open(index_path, "r", encoding="utf-8") as f:
        index_content = f.read()

    # Find the toctree section
    toctree_match = re.search(r".. toctree::(.*?)(?=\n\n)", index_content, re.DOTALL)
    if toctree_match:
        # Extract the toctree options
        toctree_options = toctree_match.group(1)

        # Create a new toctree with the sections
        new_toctree = ".. toctree::" + toctree_options + "\n"

        # Add the sections
        for title, filename in section_files.items():
            # Remove the .rst extension
            filename = filename[:-4]
            new_toctree += f"   {filename}\n"

        # Replace the old toctree with the new one
        new_index_content = index_content.replace(toctree_match.group(0), new_toctree)

        # Write the updated index.rst file
        with open(index_path, "w", encoding="utf-8") as f:
            f.write(new_index_content)

def main():
    """
    Main function to convert the manual.docx to reStructuredText format.
    """
    print(f"Converting {MANUAL_PATH} to reStructuredText format...")

    # Check if the manual.docx file exists
    if not MANUAL_PATH.exists():
        print(f"Error: {MANUAL_PATH} does not exist.")
        sys.exit(1)

    # Check if the docs directory exists
    if not DOCS_PATH.exists():
        print(f"Error: {DOCS_PATH} does not exist.")
        sys.exit(1)

    # Extract images from the document
    print("Extracting images...")
    image_rels = extract_images(MANUAL_PATH, DOCS_PATH)
    print(f"Extracted {len(image_rels)} images.")

    # Extract headings and content
    print("Extracting headings and content...")
    structure = extract_headings_and_content(MANUAL_PATH, image_rels)
    print(f"Extracted {len(structure)} sections.")

    # Create RST files
    print("Creating RST files...")
    section_files = create_rst_files(structure, DOCS_PATH)
    print(f"Created {len(section_files)} RST files.")

    # Update index.rst
    print("Updating index.rst...")
    update_index_rst(section_files, DOCS_PATH)

    print("Conversion complete!")

if __name__ == "__main__":
    main()
