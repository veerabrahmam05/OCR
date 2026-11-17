"""Prompt text for LLM extraction."""

from __future__ import annotations

from ocr_pipeline.models.document import PageContent

SYSTEM_PROMPT = """\
You extract data from business documents. The document text was produced by PDF \
text extraction or OCR and is given as Markdown, one <page> element per page. \
Tables appear as Markdown or HTML tables.

Rules:
- Use only information written in the document. Do not guess or fill in values \
that are not there; use null (or an empty list) when a value is absent.
- Copy values as written. Only reformat when the schema requires a type: numbers \
without currency symbols or thousands separators, dates as YYYY-MM-DD when the \
field is a date.
- OCR text can contain recognition errors. Correct an obvious character error only \
when the intended value is unambiguous from context.
- For every non-null value, add an entry to "sources": "field" is the value's JSON \
path inside "data" (for example invoice_number or line_items[2].amount), "page" is \
the page number it came from, and "quote" is an excerpt copied exactly from that \
page's text that contains the value with enough context to find it unambiguously:
  - A labelled value: include the label, e.g. "Invoice date: 12/11/2026", not "12/11/2026".
  - A table cell: quote the whole table row, e.g. "Steel bolts 200 0.50 100.00", not "200".
  - When all values of a list item come from one table row, give ONE source for the \
item (field "line_items[0]"), not one per value.
  - Never use a bare number or a single word as the quote.
"""


def render_pages(pages: list[PageContent]) -> str:
    parts = [
        f'<page number="{p.page_number}">\n{p.markdown.strip()}\n</page>'
        for p in pages
        if p.markdown.strip()
    ]
    return "<document>\n" + "\n".join(parts) + "\n</document>"


DATE_HINTS = {
    "DMY": (
        "Numeric dates in this document are written day first (DD/MM/YYYY): "
        "12/11/2026 means 12 November 2026, returned as 2026-11-12."
    ),
    "MDY": (
        "Numeric dates in this document are written month first (MM/DD/YYYY): "
        "12/11/2026 means 11 December 2026, returned as 2026-12-11."
    ),
    "YMD": (
        "Numeric dates in this document are written year first (YYYY/MM/DD): "
        "2026/11/12 means 12 November 2026, returned as 2026-11-12."
    ),
}


def user_prompt(
    pages: list[PageContent],
    instructions: str | None,
    chunk_index: int,
    chunk_count: int,
    free_form: bool,
    date_order: str = "auto",
) -> str:
    sections: list[str] = []
    if chunk_count > 1:
        first, last = pages[0].page_number, pages[-1].page_number
        sections.append(
            f"This is part {chunk_index + 1} of {chunk_count} of a longer document "
            f"(pages {first}-{last}). Extract only what appears in these pages; the other "
            "parts are processed separately and merged afterwards."
        )
    if instructions:
        sections.append(f"<instructions>\n{instructions.strip()}\n</instructions>")
    if date_order in DATE_HINTS:
        sections.append(DATE_HINTS[date_order])
    if free_form:
        sections.append(
            "Return each value the instructions ask for as an entry in data.fields, with a "
            'short snake_case name. In "sources", use that name as the "field".'
        )
    else:
        sections.append("Fill the fields of the response schema from the document.")
    sections.append(render_pages(pages))
    return "\n\n".join(sections)


def retry_prompt(previous_output: str, error: str) -> str:
    return (
        "\n\nYour previous answer did not pass validation.\n"
        f"<previous_answer>\n{previous_output}\n</previous_answer>\n"
        f"<validation_error>\n{error}\n</validation_error>\n"
        "Answer again, fixing these errors."
    )
