"""
Markdown export utilities with asset extraction for Open Notebook.

Handles exporting DoclingDocument to markdown with images and tables
extracted to subdirectories with proper hyperlinks.
"""

import base64
import os
from pathlib import Path
from typing import List, Optional, Tuple

from docling_core.types.doc import DoclingDocument, PictureItem, TableItem
from loguru import logger


def extract_images_from_document(
    doc: DoclingDocument,
    images_dir: str,
) -> List[Tuple[str, str]]:
    """
    Extract images from DoclingDocument to images subdirectory.

    Args:
        doc: DoclingDocument containing images
        images_dir: Directory to save images

    Returns:
        List of tuples (image_id, relative_path) for hyperlink generation

    """
    extracted_images = []

    # Ensure images directory exists
    Path(images_dir).mkdir(parents=True, exist_ok=True)

    # Iterate through document elements to find images
    image_counter = 1

    for item in doc.iterate_items():
        if isinstance(item, PictureItem):
            try:
                # Generate filename
                image_filename = f"image_{image_counter:03d}.png"
                image_path = os.path.join(images_dir, image_filename)

                # Extract image data
                if hasattr(item, 'image') and item.image:
                    # If image data is available directly
                    with open(image_path, 'wb') as f:
                        if isinstance(item.image, bytes):
                            f.write(item.image)
                        elif isinstance(item.image, str):
                            # Handle base64 encoded images
                            image_data = base64.b64decode(item.image)
                            f.write(image_data)

                    # Store relative path for markdown
                    relative_path = f"images/{image_filename}"
                    extracted_images.append((str(item.self_ref), relative_path))

                    logger.debug(f"Extracted image {image_counter}: {image_filename}")
                    image_counter += 1

            except Exception as e:
                logger.warning(f"Failed to extract image {image_counter}: {e}")
                continue

    logger.info(f"✅ Extracted {len(extracted_images)} images")
    return extracted_images


def extract_tables_from_document(
    doc: DoclingDocument,
    tables_dir: str,
) -> List[Tuple[str, str]]:
    """
    Extract tables from DoclingDocument to tables subdirectory.

    Args:
        doc: DoclingDocument containing tables
        tables_dir: Directory to save tables

    Returns:
        List of tuples (table_id, relative_path) for hyperlink generation
    """
    extracted_tables = []

    # Ensure tables directory exists
    Path(tables_dir).mkdir(parents=True, exist_ok=True)

    # Iterate through document elements to find tables
    table_counter = 1

    for item in doc.iterate_items():
        if isinstance(item, TableItem):
            try:
                # Generate filename
                table_filename = f"table_{table_counter:03d}.csv"
                table_path = os.path.join(tables_dir, table_filename)

                # Export table to CSV
                if hasattr(item, 'export_to_dataframe'):
                    df = item.export_to_dataframe()
                    df.to_csv(table_path, index=False)

                    # Store relative path for markdown
                    relative_path = f"tables/{table_filename}"
                    extracted_tables.append((str(item.self_ref), relative_path))

                    logger.debug(f"Extracted table {table_counter}: {table_filename}")
                    table_counter += 1

            except Exception as e:
                logger.warning(f"Failed to extract table {table_counter}: {e}")
                continue

    logger.info(f"✅ Extracted {len(extracted_tables)} tables")
    return extracted_tables


def create_markdown_with_links(
    doc: DoclingDocument,
    images: List[Tuple[str, str]],
    tables: List[Tuple[str, str]],
) -> str:
    """
    Create markdown content with hyperlinks to extracted assets.

    Args:
        doc: DoclingDocument to export
        images: List of (image_id, relative_path) tuples
        tables: List of (table_id, relative_path) tuples

    Returns:
        Markdown string with proper hyperlinks to images and tables
    """
    # Export base markdown from DoclingDocument
    markdown_content = doc.export_to_markdown()

    # Create lookup dictionaries for quick access
    images_dict = dict(images)
    tables_dict = dict(tables)

    # Replace image references with markdown links
    for img_id, img_path in images:
        # Look for image references in various formats
        # This is a simple replacement - may need to be more sophisticated
        placeholder = f"[Image: {img_id}]"
        markdown_link = f"![Image]({img_path})"

        if placeholder in markdown_content:
            markdown_content = markdown_content.replace(placeholder, markdown_link)

    # Replace table references with markdown links
    for table_id, table_path in tables:
        placeholder = f"[Table: {table_id}]"
        markdown_link = f"[Table (CSV)]({table_path})"

        if placeholder in markdown_content:
            markdown_content = markdown_content.replace(placeholder, markdown_link)

    logger.info("✅ Created markdown with asset hyperlinks")
    return markdown_content


def export_document_to_markdown_with_assets(
    doc: DoclingDocument,
    output_dir: str,
    document_name: str,
) -> Tuple[str, int, int]:
    """
    Export DoclingDocument to markdown with images and tables in subdirectories.

    Creates:
        output_dir/document.md (main markdown with hyperlinks)
        output_dir/images/ (extracted images)
        output_dir/tables/ (extracted tables as CSV)

    Args:
        doc: DoclingDocument to export
        output_dir: Directory to save markdown and assets
        document_name: Base name for the markdown file

    Returns:
        Tuple of (markdown_path, num_images, num_tables)

    Raises:
        OSError: If file operations fail
    """
    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Define subdirectory paths
    images_dir = os.path.join(output_dir, "images")
    tables_dir = os.path.join(output_dir, "tables")

    # Extract images
    images = extract_images_from_document(doc, images_dir)

    # Extract tables
    tables = extract_tables_from_document(doc, tables_dir)

    # Create markdown with hyperlinks
    markdown_content = create_markdown_with_links(doc, images, tables)

    # Save markdown file
    markdown_filename = f"{document_name}.md"
    markdown_path = os.path.join(output_dir, markdown_filename)

    with open(markdown_path, 'w', encoding='utf-8') as f:
        f.write(markdown_content)

    logger.info(f"✅ Exported markdown with assets: {markdown_path}")
    logger.info(f"   Images: {len(images)}, Tables: {len(tables)}")

    return markdown_path, len(images), len(tables)
