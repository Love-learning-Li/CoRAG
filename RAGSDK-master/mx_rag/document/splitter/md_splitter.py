#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------
This file is part of the RAGSDK project.
Copyright (c) 2025 Huawei Technologies Co.,Ltd.

RAGSDK is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:

         http://license.coscl.org.cn/MulanPSL2

THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
-------------------------------------------------------------------------
"""


from typing import List, Dict

from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter

from mx_rag.utils.common import validate_params, HEADER_MARK, MAX_SPLIT_SIZE


class MarkdownTextSplitter(RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter):
    """Text splitter for Markdown documents with hierarchical header support."""
    _HEADERS_TO_SPLIT_ON = [
              ("#", "Header 1"),
              ("##", "Header 2"),
              ("###", "Header 3"),
              ("####", "Header 4"),
              ("#####", "Header 5"),
              ("######", "Header 6")
    ]

    @validate_params(
        chunk_size=dict(validator=lambda x: isinstance(x, int) and x > 0,
                        message="chunk_size must be a positive integer"),
        chunk_overlap=dict(validator=lambda x: isinstance(x, int) and x >= 0,
                           message="chunk_overlap must be a non-negative integer"),
        header_level=dict(validator=lambda x: isinstance(x, int) and 0 <= x <= 6,
                          message="header_level must be an integer between 0 and 6")
    )
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 50, header_level: int = 3, **kwargs):
        """Initialize the text splitter with chunk size and overlap settings."""
        RecursiveCharacterTextSplitter.__init__(
            self,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            **kwargs
        )

        headers_to_split_on = self._HEADERS_TO_SPLIT_ON[:header_level]
        MarkdownHeaderTextSplitter.__init__(
            self,
            headers_to_split_on=headers_to_split_on
        )

    @validate_params(
        text=dict(validator=lambda x: isinstance(x, str) and len(x) <= MAX_SPLIT_SIZE,
                  message="text must be a string, length range [0, 100 * 1024 * 1024]")
    )
    def split_text(self, text: str) -> List[str]:
        """Split the input text into chunks according to size and header structure."""
        if len(text) <= self._chunk_size:
            return [text]

        # First split by headers
        header_chunks = MarkdownHeaderTextSplitter.split_text(self, text)

        result_chunks = []
        i = 0

        while i < len(header_chunks):
            current_chunk = header_chunks[i]

            # Merge small chunks with following chunks
            if len(current_chunk.page_content) <= self._chunk_size:
                merged_content = current_chunk.page_content
                merged_metadata = current_chunk.metadata.copy()
                j = i + 1

                # Merge until reaching chunk size or no more chunks
                while (j < len(header_chunks) and
                       len(merged_content) + len(header_chunks[j].page_content) <= self._chunk_size):
                    merged_content += "\n\n" + header_chunks[j].page_content
                    merged_metadata = self._merge_metadata(merged_metadata, header_chunks[j].metadata)
                    j += 1

                final_content = self._format_content_with_headers(merged_metadata, merged_content)
                result_chunks.append(final_content)
                i = j

            else:
                # Recursively split large chunks
                sub_chunks = RecursiveCharacterTextSplitter.split_text(
                    self, current_chunk.page_content
                )

                for sub_chunk in sub_chunks:
                    formatted_content = self._format_content_with_headers(
                        current_chunk.metadata, sub_chunk
                    )
                    result_chunks.append(formatted_content)

                i += 1

        return result_chunks

    def _format_content_with_headers(self, metadata: Dict[str, str], content: str) -> str:
        """Format content by combining header metadata and content text."""
        header_lines = []

        for header_name in [header[1] for header in self._HEADERS_TO_SPLIT_ON]:
            if header_name in metadata and metadata[header_name]:
                level_num = int(header_name.split()[-1])
                header_level = HEADER_MARK * level_num
                header_lines.append(f"{header_level} {metadata[header_name]}")

        if header_lines:
            headers_content = "\n".join(header_lines)
            return f"{headers_content}\n\n{content}"
        return content

    def _merge_metadata(self, metadata1: Dict[str, str], metadata2: Dict[str, str]) -> Dict[str, str]:
        """Merge two metadata dictionaries handling header conflicts."""
        merged = {}

        common_keys = [key for key in metadata1 if key in metadata2]
        for key in common_keys:
            if metadata1[key] == metadata2[key]:
                merged[key] = metadata1[key]
        return merged