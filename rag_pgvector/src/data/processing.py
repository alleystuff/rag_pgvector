import pdfplumber


def _table_to_pipe_delimited(table: list[list], header: str | None = None) -> str:
    """Convert a pdfplumber table (list of rows) to a pipe-delimited string.

    If a header string is provided (extracted from above the table bbox), it is
    prepended as the header row followed by a separator line.
    If a row's cells already contain pipe characters, the cells are joined
    with a space instead of adding extra pipes.
    """
    if not table:
        return ""

    def format_row(cells):
        already_delimited = any("|" in cell for cell in cells)
        return " ".join(cells) if already_delimited else " | ".join(cells)

    rows = []

    if header:
        rows.append(header)
        rows.append("---")

    for row in table:
        cells = [cell if cell is not None else "" for cell in row]
        rows.append(format_row(cells))

    return "\n".join(rows)


def _extract_table_header(page, table_bbox, lookup_height: int = 40) -> str | None:
    """Extract the last text line immediately above a table's bounding box.

    This captures column headers (e.g. '2025  2024  Change') that pdfplumber
    places outside the detected table region.
    """
    x0, top, x1, _ = table_bbox
    crop_top = max(0, top - lookup_height)
    header_area = page.crop((x0, crop_top, x1, top))
    text = header_area.extract_text()
    if not text:
        return None
    last_line = text.strip().splitlines()[-1].strip()
    return last_line if last_line else None


def load_pdf(path: str) -> list[dict]:
    """Load a PDF and extract text and table content from each page separately.

    For each page, tables are detected and their bounding boxes are used to
    filter out those characters before extracting plain text. Table contents
    are extracted separately in pipe-delimited format with column headers.

    Returns a list of dicts with keys:
        - page (int): 1-based page number
        - has_table (bool): whether a table was detected on the page
        - table_content (str): pipe-delimited table text (empty string if no table)
        - text (str): non-table text extracted from the page
    """
    pages = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages):
            tables = page.find_tables()
            has_table = len(tables) > 0

            if has_table:
                table_parts = []
                for t in tables:
                    rows = t.extract()
                    if rows:
                        header = _extract_table_header(page, t.bbox)
                        table_parts.append(_table_to_pipe_delimited(rows, header=header))
                table_content = "\n\n".join(table_parts)

                table_bboxes = [t.bbox for t in tables]

                def not_in_table(obj, bboxes=table_bboxes):
                    for bbox in bboxes:
                        x0, top, x1, bottom = bbox
                        if (obj["x0"] >= x0 and obj["x1"] <= x1
                                and obj["top"] >= top and obj["bottom"] <= bottom):
                            return False
                    return True

                text = page.filter(not_in_table).extract_text() or ""
            else:
                table_content = ""
                text = page.extract_text() or ""

            pages.append({
                "page": i + 1,
                "has_table": has_table,
                "table_content": table_content,
                "text": text,
            })
    return pages