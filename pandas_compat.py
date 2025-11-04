"""Minimal pandas compatibility layer for environments without pandas."""
from __future__ import annotations

import math
import re
from html import unescape
from typing import Iterable, List


def _strip_html(text: str) -> str:
    cleaned = re.sub(r"<.*?>", "", text, flags=re.S)
    return unescape(cleaned).strip()


class MiniRow:
    def __init__(self, data: dict[str, str], columns: List[str]):
        self._data = data
        self._columns = columns

    def get(self, key: str, default=None):
        return self._data.get(key, default)

    def __getitem__(self, key: str) -> str:
        return self._data[key]

    @property
    def values(self) -> List[str]:
        return [self._data.get(col, "") for col in self._columns]


class MiniDataFrame:
    def __init__(self, rows: List[dict[str, str]], columns: List[str]):
        self._rows = rows
        self._columns = list(columns)

    def copy(self) -> "MiniDataFrame":
        rows_copy = [dict(row) for row in self._rows]
        return MiniDataFrame(rows_copy, list(self._columns))

    def iterrows(self):
        for idx, row in enumerate(self._rows):
            yield idx, MiniRow(row, self._columns)

    @property
    def columns(self) -> List[str]:
        return self._columns

    @columns.setter
    def columns(self, new_columns: Iterable[str]) -> None:
        new_columns = list(new_columns)
        old_columns = list(self._columns)
        self._columns = new_columns
        for row in self._rows:
            values = [row.get(col, "") for col in old_columns]
            row.clear()
            for idx, col in enumerate(new_columns):
                row[col] = values[idx] if idx < len(values) else ""

    def __len__(self) -> int:
        return len(self._rows)


def _parse_table(table_html: str) -> MiniDataFrame | None:
    row_htmls = re.findall(r"<tr[^>]*>(.*?)</tr>", table_html, flags=re.S | re.I)
    if not row_htmls:
        return None
    header: List[str] | None = None
    rows: List[List[str]] = []
    for row_html in row_htmls:
        cells = re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", row_html, flags=re.S | re.I)
        if not cells:
            continue
        values = [_strip_html(cell) for cell in cells]
        if header is None:
            header = values
        else:
            rows.append(values)
    if header is None:
        return None
    columns = [col.strip() or f"col_{idx}" for idx, col in enumerate(header)]
    dict_rows: List[dict[str, str]] = []
    for values in rows:
        record: dict[str, str] = {}
        for idx, column in enumerate(columns):
            record[column] = values[idx] if idx < len(values) else ""
        dict_rows.append(record)
    return MiniDataFrame(dict_rows, columns)


def _read_tables(source) -> List[MiniDataFrame]:
    if hasattr(source, "read"):
        html = source.read()
    else:
        html = source
    if not isinstance(html, str):
        html = html.decode("utf-8", errors="ignore")
    tables_html = re.findall(r"<table[^>]*>(.*?)</table>", html, flags=re.S | re.I)
    frames: List[MiniDataFrame] = []
    for table_html in tables_html:
        df = _parse_table(table_html)
        if df is not None:
            frames.append(df)
    if not frames:
        raise ValueError("No tables found")
    return frames


class _CompatPandas:
    def read_html(self, source) -> List[MiniDataFrame]:
        return _read_tables(source)

    def isna(self, value) -> bool:
        if value is None:
            return True
        if isinstance(value, float):
            return math.isnan(value)
        return False


pd = _CompatPandas()
