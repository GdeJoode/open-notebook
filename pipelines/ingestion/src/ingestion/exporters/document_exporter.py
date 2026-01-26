"""
Document content exporter.

Exports extracted document content to a fixed directory structure:
    ./[source_name]/
        original_file/          # Original document copy
        output/
            extracted_info/
                document.md     # Full markdown
                document.json   # Structured JSON
                metadata.json   # Extraction metadata
                images/         # Extracted images
                tables/         # Tables in MD and CSV
"""

import base64
import json
import shutil
from pathlib import Path
from typing import Optional

from loguru import logger

from ingestion.config import OutputConfig, TableFormat
from ingestion.models import ExtractedDocument, ExtractedImage, ExtractedTable


class DocumentExporter:
    """
    Export extracted documents to the standard directory structure.

    Features:
    - Markdown and JSON export
    - Image file extraction
    - Table export in both MD and CSV formats
    - Original file preservation
    - Metadata export
    """

    def __init__(self, config: Optional[OutputConfig] = None):
        """
        Initialize the document exporter.

        Args:
            config: OutputConfig instance. If None, uses defaults.
        """
        self.config = config or OutputConfig()

    def export(
        self,
        document: ExtractedDocument,
        source_name: Optional[str] = None,
        copy_original: bool = True,
        table_format: TableFormat = TableFormat.BOTH,
    ) -> dict[str, Path]:
        """
        Export a document to the standard directory structure.

        Args:
            document: Extracted document to export
            source_name: Name for the output directory (defaults to source filename)
            copy_original: Whether to copy the original file
            table_format: Format for table export (MD, CSV, or BOTH)

        Returns:
            Dictionary of exported file paths
        """
        # Determine source name
        if source_name is None:
            source_name = document.source_path.stem

        logger.info(f"Exporting document: {source_name}")

        # Create directories
        dirs = self.config.create_directories(source_name, include_original=copy_original)

        exported_paths: dict[str, Path] = {
            "root": dirs["root"],
            "output": dirs["output"],
        }

        # Copy original file
        if copy_original and document.source_path.exists():
            original_dest = dirs["original"] / document.source_path.name
            shutil.copy2(document.source_path, original_dest)
            exported_paths["original"] = original_dest
            logger.debug(f"Copied original to: {original_dest}")

        # Export markdown
        markdown_path = dirs["output"] / "document.md"
        markdown_content = self._generate_markdown(document)
        markdown_path.write_text(markdown_content, encoding="utf-8")
        exported_paths["markdown"] = markdown_path
        logger.debug(f"Exported markdown to: {markdown_path}")

        # Export JSON
        json_path = dirs["output"] / "document.json"
        json_path.write_text(document.to_json(), encoding="utf-8")
        exported_paths["json"] = json_path
        logger.debug(f"Exported JSON to: {json_path}")

        # Export metadata
        metadata_path = dirs["output"] / "metadata.json"
        metadata = self._generate_metadata(document)
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        exported_paths["metadata"] = metadata_path

        # Export images
        image_paths = self._export_images(document, dirs["images"])
        exported_paths["images"] = list(image_paths.values())

        # Export tables
        table_paths = self._export_tables(document, dirs["tables"], table_format)
        exported_paths["tables"] = list(table_paths.values())

        logger.info(
            f"Document exported: {len(image_paths)} images, {len(table_paths)} tables"
        )

        return exported_paths

    def _generate_markdown(self, document: ExtractedDocument) -> str:
        """Generate markdown with page organization."""
        parts = []

        # Title
        if document.title:
            parts.append(f"# {document.title}")
        else:
            parts.append(f"# {document.source_path.name}")

        # Metadata header
        parts.append("")
        parts.append(f"> **Source**: {document.source_path.name}")
        parts.append(f"> **Pages**: {document.page_count}")
        parts.append(f"> **Tables**: {document.table_count}")
        parts.append(f"> **Images**: {document.image_count}")
        parts.append(f"> **Extracted**: {document.extracted_at.isoformat()}")
        parts.append("")

        # Page content
        for page in document.pages:
            parts.append(f"\n---\n\n## Page {page.page_number}\n")

            # Elements in reading order
            page_content = page.to_markdown(include_images=True, include_tables=True)
            parts.append(page_content)

        return "\n".join(parts)

    def _generate_metadata(self, document: ExtractedDocument) -> dict:
        """Generate extraction metadata."""
        return {
            "source_path": str(document.source_path),
            "source_name": document.source_path.name,
            "title": document.title,
            "page_count": document.page_count,
            "table_count": document.table_count,
            "image_count": document.image_count,
            "extracted_at": document.extracted_at.isoformat(),
            "extraction_time_seconds": document.extraction_time_seconds,
            "pages": [
                {
                    "page_number": page.page_number,
                    "element_count": page.element_count,
                    "table_count": len(page.tables),
                    "image_count": len(page.images),
                }
                for page in document.pages
            ],
            "docling_metadata": document.metadata,
        }

    def _export_images(
        self,
        document: ExtractedDocument,
        images_dir: Path,
    ) -> dict[str, Path]:
        """Export all extracted images."""
        exported = {}

        for idx, image in enumerate(document.all_images, 1):
            # Generate filename
            filename = f"image_{idx:03d}_page{image.page}.{image.format}"
            image_path = images_dir / filename

            # Export image
            if image.image_data:
                # Decode base64
                try:
                    # Handle data URI or raw base64
                    data = image.image_data
                    if "base64," in data:
                        data = data.split("base64,")[1]
                    image_bytes = base64.b64decode(data)
                    image_path.write_bytes(image_bytes)
                    exported[f"image_{idx}"] = image_path

                    # Update image path reference
                    image.image_path = image_path
                except Exception as e:
                    logger.warning(f"Failed to export image {idx}: {e}")

            elif image.image_path and image.image_path.exists():
                # Copy existing file
                shutil.copy2(image.image_path, image_path)
                exported[f"image_{idx}"] = image_path
                image.image_path = image_path

            # Write description file
            if image.description:
                desc_path = images_dir / f"image_{idx:03d}_description.txt"
                desc_content = f"Classification: {image.classification}\n\n{image.description}"
                desc_path.write_text(desc_content, encoding="utf-8")

        return exported

    def _export_tables(
        self,
        document: ExtractedDocument,
        tables_dir: Path,
        table_format: TableFormat,
    ) -> dict[str, Path]:
        """Export all extracted tables."""
        exported = {}

        for idx, table in enumerate(document.all_tables, 1):
            base_name = f"table_{idx:03d}_page{table.page}"

            # Export Markdown
            if table_format in (TableFormat.MARKDOWN, TableFormat.BOTH):
                md_path = tables_dir / f"{base_name}.md"
                md_content = table.to_markdown_string() or table.markdown
                if md_content:
                    md_path.write_text(md_content, encoding="utf-8")
                    exported[f"table_{idx}_md"] = md_path

            # Export CSV
            if table_format in (TableFormat.CSV, TableFormat.BOTH):
                csv_path = tables_dir / f"{base_name}.csv"
                csv_content = table.to_csv_string() or table.csv
                if csv_content:
                    csv_path.write_text(csv_content, encoding="utf-8")
                    exported[f"table_{idx}_csv"] = csv_path

        return exported


def export_document(
    document: ExtractedDocument,
    output_dir: Optional[Path] = None,
    source_name: Optional[str] = None,
    copy_original: bool = True,
    table_format: TableFormat = TableFormat.BOTH,
) -> dict[str, Path]:
    """
    Convenience function to export a document.

    Args:
        document: Extracted document to export
        output_dir: Base output directory (optional)
        source_name: Name for output directory (optional)
        copy_original: Whether to copy original file
        table_format: Format for table export

    Returns:
        Dictionary of exported file paths
    """
    config = OutputConfig()
    if output_dir:
        config.base_dir = output_dir

    exporter = DocumentExporter(config)
    return exporter.export(
        document=document,
        source_name=source_name,
        copy_original=copy_original,
        table_format=table_format,
    )
