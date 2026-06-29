"""Extract structured content from scientific PDF papers.

Primary parser:
    Docling, one page at a time to reduce memory usage.

Fallback parser:
    PyMuPDF, only for pages that Docling cannot process.

Generated files for each paper:
    data/extracted/<paper_id>/document.md
    data/extracted/<paper_id>/document.json
    data/extracted/<paper_id>/metadata.json
"""

from __future__ import annotations

import gc
import json
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pymupdf
from docling.datamodel.accelerator_options import (
    AcceleratorDevice,
    AcceleratorOptions,
)
from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.datamodel.pipeline_options import TableFormerMode

try:
    from docling.datamodel.pipeline_options import (
        ThreadedPdfPipelineOptions as PdfPipelineOptionsClass,
    )
except ImportError:
    from docling.datamodel.pipeline_options import (
        PdfPipelineOptions as PdfPipelineOptionsClass,
    )
from docling.document_converter import DocumentConverter, PdfFormatOption


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PAPERS_METADATA_PATH = PROJECT_ROOT / "data" / "papers.json"
EXTRACTED_OUTPUT_DIR = PROJECT_ROOT / "data" / "extracted"

MIN_PAGE_TEXT_LENGTH = 20


def load_papers_metadata() -> list[dict[str, Any]]:
    """Load and validate data/papers.json."""
    if not PAPERS_METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Metadata file was not found: {PAPERS_METADATA_PATH}"
        )

    with PAPERS_METADATA_PATH.open("r", encoding="utf-8") as file:
        papers = json.load(file)

    if not isinstance(papers, list) or not papers:
        raise ValueError(
            "papers.json must contain a non-empty JSON list."
        )

    required_fields = {
        "paper_id",
        "short_name",
        "title",
        "file_path",
    }
    observed_ids: set[str] = set()

    for index, paper in enumerate(papers, start=1):
        if not isinstance(paper, dict):
            raise ValueError(
                f"Paper number {index} must be a JSON object."
            )

        missing_fields = required_fields - set(paper)
        if missing_fields:
            raise ValueError(
                f"Paper number {index} is missing fields: "
                f"{sorted(missing_fields)}"
            )

        paper_id = str(paper["paper_id"]).strip()
        if not paper_id:
            raise ValueError(
                f"Paper number {index} has an empty paper_id."
            )

        if paper_id in observed_ids:
            raise ValueError(f"Duplicate paper ID: {paper_id}")
        observed_ids.add(paper_id)

        pdf_path = PROJECT_ROOT / Path(paper["file_path"])
        if not pdf_path.exists():
            raise FileNotFoundError(
                f"PDF for {paper_id} was not found: {pdf_path}"
            )

        if pdf_path.stat().st_size == 0:
            raise ValueError(f"PDF file is empty: {pdf_path}")

    return papers


def set_option_if_supported(
    options: Any,
    option_name: str,
    value: Any,
) -> None:
    """Set a Docling option only when the installed version supports it."""
    if hasattr(options, option_name):
        setattr(options, option_name, value)


def create_docling_converter() -> DocumentConverter:
    """Create a low-memory Docling converter for text-based research PDFs."""
    pipeline_options = PdfPipelineOptionsClass()

    pipeline_options.accelerator_options = AcceleratorOptions(
        num_threads=2,
        device=AcceleratorDevice.CPU,
    )

    set_option_if_supported(pipeline_options, "ocr_batch_size", 1)
    set_option_if_supported(pipeline_options, "layout_batch_size", 1)
    set_option_if_supported(pipeline_options, "table_batch_size", 1)
    set_option_if_supported(pipeline_options, "queue_max_size", 2)

    pipeline_options.do_ocr = False
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options.mode = (
        TableFormerMode.ACCURATE
    )

    set_option_if_supported(pipeline_options, "do_code_enrichment", False)
    set_option_if_supported(pipeline_options, "do_formula_enrichment", False)
    set_option_if_supported(pipeline_options, "do_picture_classification", False)
    set_option_if_supported(pipeline_options, "do_picture_description", False)
    set_option_if_supported(pipeline_options, "generate_page_images", False)
    set_option_if_supported(pipeline_options, "generate_picture_images", False)
    set_option_if_supported(pipeline_options, "generate_table_images", False)
    set_option_if_supported(pipeline_options, "generate_parsed_pages", False)

    return DocumentConverter(
        allowed_formats=[InputFormat.PDF],
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options
            )
        },
    )


def get_pdf_page_count(pdf_path: Path) -> int:
    """Return the number of pages in a PDF."""
    with pymupdf.open(pdf_path) as document:
        if document.page_count <= 0:
            raise ValueError(f"PDF contains no pages: {pdf_path}")
        return document.page_count


def model_to_json(value: Any) -> Any:
    """Convert a Pydantic-like object into JSON-compatible data."""
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")

    if hasattr(value, "dict"):
        return value.dict()

    return str(value)


def serialize_docling_errors(errors: list[Any]) -> list[Any]:
    """Serialize Docling error objects for metadata output."""
    return [model_to_json(error) for error in errors]


def extract_page_with_pymupdf(
    pdf_path: Path,
    page_number: int,
) -> tuple[str, dict[str, Any]]:
    """Extract one page with PyMuPDF as a reliable fallback."""
    with pymupdf.open(pdf_path) as document:
        if page_number < 1 or page_number > document.page_count:
            raise ValueError(
                f"Page {page_number} is outside the PDF page range."
            )

        page = document.load_page(page_number - 1)
        text = page.get_text("text", sort=True).strip()

    return (
        text,
        {
            "schema_name": "pymupdf_page_fallback",
            "page_number": page_number,
            "text": text,
            "character_count": len(text),
            "word_count": len(text.split()),
        },
    )


def extract_one_page(
    converter: DocumentConverter,
    pdf_path: Path,
    page_number: int,
) -> dict[str, Any]:
    """Extract one PDF page with Docling, falling back to PyMuPDF."""
    docling_status: str | None = None
    docling_errors: list[Any] = []

    try:
        result = converter.convert(
            pdf_path,
            page_range=(page_number, page_number),
            raises_on_error=False,
        )

        docling_status = str(result.status.value)
        docling_errors = serialize_docling_errors(result.errors)

        if (
            result.status == ConversionStatus.SUCCESS
            and not result.errors
        ):
            markdown = result.document.export_to_markdown().strip()

            if len(markdown) >= MIN_PAGE_TEXT_LENGTH:
                return {
                    "page_number": page_number,
                    "engine": "docling",
                    "docling_status": docling_status,
                    "errors": [],
                    "markdown": markdown,
                    "structured_document": (
                        result.document.export_to_dict()
                    ),
                    "table_count": len(
                        getattr(result.document, "tables", [])
                    ),
                    "text_item_count": len(
                        getattr(result.document, "texts", [])
                    ),
                    "character_count": len(markdown),
                    "word_count": len(markdown.split()),
                }

            docling_errors.append(
                {
                    "message": (
                        "Docling returned suspiciously little text "
                        f"for page {page_number}."
                    )
                }
            )

    except Exception as error:
        docling_status = "exception"
        docling_errors.append(
            {
                "type": type(error).__name__,
                "message": str(error),
            }
        )

    fallback_text, fallback_document = extract_page_with_pymupdf(
        pdf_path=pdf_path,
        page_number=page_number,
    )

    return {
        "page_number": page_number,
        "engine": "pymupdf_fallback",
        "docling_status": docling_status,
        "errors": docling_errors,
        "markdown": fallback_text,
        "structured_document": fallback_document,
        "table_count": 0,
        "text_item_count": None,
        "character_count": len(fallback_text),
        "word_count": len(fallback_text.split()),
    }


def extract_paper_pagewise(
    converter: DocumentConverter,
    paper: dict[str, Any],
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    """Extract all pages sequentially to keep memory use low."""
    pdf_path = PROJECT_ROOT / Path(paper["file_path"])
    page_count = get_pdf_page_count(pdf_path)

    page_results: list[dict[str, Any]] = []
    markdown_parts: list[str] = []

    docling_pages = 0
    fallback_pages: list[int] = []
    total_tables = 0
    total_text_items = 0
    total_characters = 0
    total_words = 0

    for page_number in range(1, page_count + 1):
        print(
            f"    Page {page_number}/{page_count}",
            end="",
            flush=True,
        )

        page_result = extract_one_page(
            converter=converter,
            pdf_path=pdf_path,
            page_number=page_number,
        )

        engine = page_result["engine"]
        print(f" — {engine}")

        if engine == "docling":
            docling_pages += 1
        else:
            fallback_pages.append(page_number)

        total_tables += int(page_result["table_count"] or 0)
        total_text_items += int(
            page_result["text_item_count"] or 0
        )
        total_characters += page_result["character_count"]
        total_words += page_result["word_count"]

        markdown_parts.append(
            f"<!-- PAGE {page_number} | ENGINE: {engine} -->\n\n"
            f"{page_result['markdown']}"
        )

        page_results.append(
            {
                key: value
                for key, value in page_result.items()
                if key != "markdown"
            }
        )

        del page_result
        gc.collect()

    combined_markdown = "\n\n".join(markdown_parts).strip()

    if len(combined_markdown) < 1000:
        raise ValueError(
            f"Suspiciously little text was extracted from "
            f"{paper['paper_id']}."
        )

    structured_document = {
        "schema_name": "pagewise_docling_extraction",
        "page_count": page_count,
        "pages": page_results,
    }

    statistics = {
        "page_count": page_count,
        "docling_pages": docling_pages,
        "fallback_page_count": len(fallback_pages),
        "fallback_pages": fallback_pages,
        "table_count": total_tables,
        "text_item_count": total_text_items,
        "markdown_character_count": total_characters,
        "markdown_word_count": total_words,
    }

    return combined_markdown, structured_document, statistics


def save_outputs(
    paper: dict[str, Any],
    markdown: str,
    structured_document: dict[str, Any],
    statistics: dict[str, Any],
) -> Path:
    """Save Markdown, structured JSON and extraction metadata."""
    paper_id = paper["paper_id"]
    paper_output_dir = EXTRACTED_OUTPUT_DIR / paper_id
    paper_output_dir.mkdir(parents=True, exist_ok=True)

    markdown_path = paper_output_dir / "document.md"
    document_json_path = paper_output_dir / "document.json"
    metadata_path = paper_output_dir / "metadata.json"

    markdown_path.write_text(markdown, encoding="utf-8")

    document_payload = {
        "paper": paper,
        "parser": {
            "strategy": "pagewise_docling_with_pymupdf_fallback",
            "extracted_at_utc": datetime.now(
                timezone.utc
            ).isoformat(),
        },
        "document": structured_document,
    }

    with document_json_path.open("w", encoding="utf-8") as file:
        json.dump(
            document_payload,
            file,
            ensure_ascii=False,
            indent=2,
        )

    metadata_payload = {
        "paper_id": paper_id,
        "short_name": paper["short_name"],
        "title": paper["title"],
        "source_pdf": paper["file_path"],
        "parser_engine": (
            "docling"
            if statistics["fallback_page_count"] == 0
            else "docling_with_pymupdf_fallback"
        ),
        "statistics": statistics,
        "output_files": {
            "markdown": str(
                markdown_path.relative_to(PROJECT_ROOT)
            ).replace("\\", "/"),
            "structured_json": str(
                document_json_path.relative_to(PROJECT_ROOT)
            ).replace("\\", "/"),
        },
    }

    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(
            metadata_payload,
            file,
            ensure_ascii=False,
            indent=2,
        )

    return paper_output_dir


def process_one_paper(
    converter: DocumentConverter,
    paper: dict[str, Any],
) -> dict[str, Any]:
    """Process and save one scientific paper."""
    markdown, structured_document, statistics = (
        extract_paper_pagewise(
            converter=converter,
            paper=paper,
        )
    )

    output_dir = save_outputs(
        paper=paper,
        markdown=markdown,
        structured_document=structured_document,
        statistics=statistics,
    )

    return {
        "paper_id": paper["paper_id"],
        "statistics": statistics,
        "output_dir": output_dir,
    }


def process_all_papers() -> None:
    """Process every paper listed in data/papers.json."""
    papers = load_papers_metadata()
    EXTRACTED_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(papers)} papers.")
    print("Initializing low-memory Docling pipeline...")

    converter = create_docling_converter()

    successful_papers = 0
    failed_papers = 0

    for paper in papers:
        print()
        print(
            f"Processing {paper['paper_id']}: "
            f"{paper['short_name']}"
        )

        try:
            result = process_one_paper(
                converter=converter,
                paper=paper,
            )
            statistics = result["statistics"]

            print(
                f"  Pages: {statistics['page_count']}"
            )
            print(
                f"  Docling pages: "
                f"{statistics['docling_pages']}"
            )
            print(
                f"  Fallback pages: "
                f"{statistics['fallback_pages']}"
            )
            print(
                f"  Tables: {statistics['table_count']}"
            )
            print(
                f"  Words: "
                f"{statistics['markdown_word_count']}"
            )
            print(
                f"  Output: "
                f"{result['output_dir'].relative_to(PROJECT_ROOT)}"
            )

            successful_papers += 1

        except Exception as error:
            failed_papers += 1
            print(
                f"  ERROR: {type(error).__name__}: {error}"
            )
            traceback.print_exc()

    print()
    print("-" * 60)
    print(f"Successful papers: {successful_papers}")
    print(f"Failed papers: {failed_papers}")

    if failed_papers:
        raise SystemExit(
            "Extraction finished with one or more errors."
        )

    print("All papers were extracted successfully.")


if __name__ == "__main__":
    process_all_papers()
