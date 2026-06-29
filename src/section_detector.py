"""Detect the main sections of the selected scientific papers."""

import json
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PAPERS_PATH = PROJECT_ROOT / "data" / "papers.json"
EXTRACTED_DIR = PROJECT_ROOT / "data" / "extracted"
SECTIONS_DIR = PROJECT_ROOT / "data" / "sections"


PAGE_PATTERN = re.compile(
    r"<!--\s*PAGE\s+(\d+)\s*\|\s*ENGINE:.*?-->",
    re.IGNORECASE,
)

INLINE_ABSTRACT_PATTERN = re.compile(
    r"^Abstract\s*[.:]\s*(.*)$",
    re.IGNORECASE,
)


SECTION_NAMES = {
    "front_matter": "Front Matter",
    "abstract": "Abstract",
    "introduction": "Introduction",
    "related_work": "Related Work",
    "methodology": "Methodology",
    "experiments": "Experiments",
    "results": "Results",
    "conclusion": "Conclusion",
    "references": "References",
    "appendix": "Appendix",
}


def load_json(path: Path):
    """Load a JSON file."""

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path: Path, data):
    """Save data as JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
        )


def clean_heading(text: str) -> str:
    """Remove Markdown signs and section numbers."""

    text = text.strip()

    # Remove Markdown heading symbols.
    text = re.sub(
        r"^#{1,6}\s*",
        "",
        text,
    )

    # Remove section numbers such as 1, 2.1 or 3.2.1.
    text = re.sub(
        r"^\d+(?:\.\d+)*[.)]?\s*",
        "",
        text,
    )

    text = text.strip(" .:-–—")

    return text.lower()


def get_section_name(heading: str):
    """Convert a paper heading to a standard section name."""

    title = clean_heading(heading)

    if title == "abstract":
        return "abstract"

    if title in {
        "introduction",
        "intro",
    }:
        return "introduction"

    if title in {
        "related work",
        "related works",
        "previous work",
        "prior work",
        "background",
        "literature review",
    }:
        return "related_work"

    if title in {
        "method",
        "methods",
        "methodology",
        "approach",
        "proposed method",
        "proposed approach",
        "model architecture",
        "architecture",
        "the detr model",
        "deformable detr",
        "conditional detr",
        "dab-detr",
        "dynamic anchor boxes",
    }:
        return "methodology"

    if title in {
        "results",
        "experimental results",
        "quantitative results",
        "qualitative results",
        "evaluation results",
        "performance comparison",
    }:
        return "results"

    if title in {
        "experiments",
        "experiment",
        "experimental setup",
        "experimental settings",
        "implementation details",
        "training details",
        "evaluation",
        "ablation study",
        "ablation studies",
    }:
        return "experiments"

    if title in {
        "conclusion",
        "conclusions",
        "discussion",
        "concluding remarks",
        "future work",
        "conclusion and future work",
        "limitations",
    }:
        return "conclusion"

    if title in {
        "references",
        "bibliography",
    }:
        return "references"

    if title.startswith("appendix"):
        return "appendix"

    return None


def detect_heading(line: str):
    """Return a standard section name if the line is a heading."""

    stripped = line.strip()

    if not stripped:
        return None

    # Markdown headings created by Docling.
    if stripped.startswith("#"):
        return get_section_name(stripped)

    # Numbered headings such as "1 Introduction".
    numbered_match = re.match(
        r"^\d+(?:\.\d+)*[.)]?\s+(.+)$",
        stripped,
    )

    if numbered_match:
        title = numbered_match.group(1)

        # Avoid treating long numbered sentences as headings.
        if len(title.split()) <= 12:
            return get_section_name(title)

    # Unnumbered short headings.
    if len(stripped.split()) <= 6:
        return get_section_name(stripped)

    return None


def create_section(
    paper_id: str,
    section_name: str,
    original_heading: str,
    start_page: int,
    end_page: int,
    lines: list[str],
    number: int,
):
    """Create one section dictionary."""

    text = "\n".join(lines).strip()

    if not text:
        return None

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text,
    )

    return {
        "section_id": (
            f"{paper_id}_{section_name}_{number:02d}"
        ),
        "canonical_name": section_name,
        "display_name": SECTION_NAMES[section_name],
        "original_heading": original_heading,
        "page_start": start_page,
        "page_end": end_page,
        "word_count": len(text.split()),
        "character_count": len(text),
        "text": text,
    }


def detect_paper_sections(paper: dict):
    """Detect sections in one paper."""

    paper_id = paper["paper_id"]

    markdown_path = (
        EXTRACTED_DIR
        / paper_id
        / "document.md"
    )

    if not markdown_path.exists():
        raise FileNotFoundError(
            f"File not found: {markdown_path}"
        )

    markdown = markdown_path.read_text(
        encoding="utf-8"
    )

    sections = []
    counters = {}

    current_page = 1
    current_section = "front_matter"
    current_heading = "Front Matter"
    start_page = 1
    end_page = 1
    current_lines = []

    def save_current_section():
        """Save the currently collected section."""

        nonlocal current_lines

        if not current_lines:
            return

        counters[current_section] = (
            counters.get(current_section, 0) + 1
        )

        section = create_section(
            paper_id=paper_id,
            section_name=current_section,
            original_heading=current_heading,
            start_page=start_page,
            end_page=end_page,
            lines=current_lines,
            number=counters[current_section],
        )

        if section is not None:
            sections.append(section)

        current_lines = []

    for line in markdown.splitlines():
        page_match = PAGE_PATTERN.match(line.strip())

        if page_match:
            current_page = int(page_match.group(1))
            continue

        stripped = line.strip()

        # Detect inline abstract:
        # Abstract. We present a new method...
        abstract_match = INLINE_ABSTRACT_PATTERN.match(
            stripped
        )

        if abstract_match:
            save_current_section()

            current_section = "abstract"
            current_heading = "Abstract"
            start_page = current_page
            end_page = current_page

            abstract_text = abstract_match.group(1)

            if abstract_text:
                current_lines.append(abstract_text)

            continue

        detected_section = detect_heading(line)

        if detected_section is not None:
            save_current_section()

            current_section = detected_section
            current_heading = clean_heading(line).title()
            start_page = current_page
            end_page = current_page

            continue

        current_lines.append(line)

        if stripped:
            end_page = current_page

    save_current_section()

    found_sections = sorted(
        {
            section["canonical_name"]
            for section in sections
        }
    )

    return {
        "paper": paper,
        "source_file": str(
            markdown_path.relative_to(PROJECT_ROOT)
        ).replace("\\", "/"),
        "section_count": len(sections),
        "found_sections": found_sections,
        "sections": sections,
    }


def create_normalized_markdown(data: dict) -> str:
    """Create readable Markdown from detected sections."""

    output = [
        f"# {data['paper']['title']}",
        "",
    ]

    for section in data["sections"]:
        output.append(
            f"<!-- SECTION: "
            f"{section['canonical_name']} | "
            f"PAGES: "
            f"{section['page_start']}-"
            f"{section['page_end']} -->"
        )

        output.append(
            f"## {section['display_name']}"
        )

        output.append("")
        output.append(section["text"])
        output.append("")

    return "\n".join(output).strip() + "\n"


def process_all_papers():
    """Detect sections for all papers."""

    papers = load_json(PAPERS_PATH)

    SECTIONS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary = []

    print(f"Found {len(papers)} papers.")

    for paper in papers:
        paper_id = paper["paper_id"]

        print()
        print(
            f"Processing {paper_id}: "
            f"{paper['short_name']}"
        )

        data = detect_paper_sections(paper)

        output_dir = SECTIONS_DIR / paper_id
        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        save_json(
            output_dir / "sections.json",
            data,
        )

        normalized_markdown = (
            create_normalized_markdown(data)
        )

        (
            output_dir
            / "normalized.md"
        ).write_text(
            normalized_markdown,
            encoding="utf-8",
        )

        print(
            f"  Sections: {data['section_count']}"
        )

        print(
            "  Found: "
            + ", ".join(data["found_sections"])
        )

        summary.append(
            {
                "paper_id": paper_id,
                "short_name": paper["short_name"],
                "section_count": data["section_count"],
                "found_sections": data["found_sections"],
            }
        )

    save_json(
        SECTIONS_DIR / "summary.json",
        summary,
    )

    print()
    print(
        "Section detection completed successfully."
    )


if __name__ == "__main__":
    process_all_papers()