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


import os
import re
import stat
from pathlib import Path
from typing import Iterator

from bs4 import BeautifulSoup
from langchain_community.document_loaders.base import BaseLoader
from langchain_core.documents import Document
from loguru import logger
from unstructured.documents.elements import (
    ElementType, Element
)


from mx_rag.document.loader.base_loader import BaseLoader as mxBaseLoader
from mx_rag.llm import Img2TextLLM
from mx_rag.utils.common import validate_params, MAX_PAGE_CONTENT, IMAGE_TYPE, HEADER_MARK
from mx_rag.utils.file_check import SecFileCheck


class MarkdownLoader(BaseLoader, mxBaseLoader):
    """Loader for loading and processing Markdown documents."""
    EXTENSION = (".md", ".markdown")

    @validate_params(
        vlm=dict(validator=lambda x: isinstance(x, Img2TextLLM) or x is None,
                 message="param must be instance of Img2TextLLM or None"),
        process_images_separately=dict(validator=lambda x: isinstance(x, bool),
                                       message="process_images_separately must be boolean")
    )
    def __init__(self, file_path: str, vlm: Img2TextLLM = None, process_images_separately: bool = False):
        """Initialize the markdown loader with file path, VLM and process_images_separately options."""
        super().__init__(file_path)
        self.vlm = vlm
        self.process_images_separately = process_images_separately
        self.table_index = 0

    def lazy_load(self) -> Iterator[Document]:
        """Lazily load and process the Markdown document."""
        from unstructured.partition.md import partition_md

        self._is_document_valid()
        elements = partition_md(filename=self.file_path)
        logger.info(f"Total markdown elements count: {len(elements)}")

        content_parts = []
        for element in elements:
            if element.category == ElementType.TABLE:
                table_text = self._handle_table(element)
                content_parts.append(table_text)
            elif element.category == ElementType.IMAGE:
                img_base64, image_text = self._handle_image(element)
                if (self.process_images_separately):
                    yield Document(page_content=image_text,
                                   metadata={
                                       "source": os.path.basename(self.file_path),
                                       "image_base64": img_base64,
                                       "type": "image"}
                                   )
                else:
                    content_parts.append(image_text)
            elif element.category == ElementType.TITLE:
                header_level = HEADER_MARK * (element.metadata.category_depth + 1)
                content_parts.append(f"{header_level} {element.text}")
            else:
                content_parts.append(element.text)

        full_content = "\n\n".join(content_parts) if content_parts else ""

        if full_content:
            yield Document(
                page_content=full_content,
                metadata={
                    "source": os.path.basename(self.file_path),
                    "type": "text"
                }
            )

    def _handle_table(self, element: Element) -> str:
        """Process table element and serialize it into text format."""
        logger.info(f"Processing markdown table {self.table_index}")
        self.table_index += 1

        table_html = element.metadata.text_as_html
        soup = BeautifulSoup(table_html, 'html.parser')
        rows = soup.find_all('tr')
        if not rows:
            logger.warning("The table is empty.")
            return ""

        header_cells = rows[0].find_all(['td', 'th'])
        headers = [cell.get_text(strip=True) for cell in header_cells]

        serialized_rows = []
        for row in rows[1:]:
            cells = row.find_all(['td', 'th'])
            row_data = [cell.get_text(strip=True) for cell in cells]
            kv_pairs = []
            for i, header in enumerate(headers):
                if i < len(row_data):
                    kv_pairs.append(f"{header}:{row_data[i]}")
            if kv_pairs:
                serialized_rows.append(",".join(kv_pairs))

        serialized_table = ";".join(serialized_rows)
        if serialized_table:
            serialized_table += "。"

        return serialized_table

    def _handle_image(self, element: Element) -> tuple[str, str]:
        img_path = element.metadata.image_url
        if img_path and self.vlm:
            try:
                if not os.path.isabs(img_path):
                    img_path = os.path.realpath(os.path.join(os.path.dirname(self.file_path), img_path))
                SecFileCheck(img_path, MAX_PAGE_CONTENT).check()
                if Path(img_path).suffix not in IMAGE_TYPE:
                    raise TypeError(f"type '{Path(img_path).suffix}' is not support")

                R_FLAGS = os.O_RDONLY
                MODES = stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH
                with os.fdopen(os.open(img_path, R_FLAGS, MODES), 'rb') as f:
                    logger.debug(f"Processing markdown image: {img_path}")
                    image_data = f.read()
                    img_base64, image_summary = self._interpret_image(image_data, self.vlm)
                    return img_base64, re.sub(r'^#+\s*', '', image_summary, flags=re.MULTILINE) if image_summary else element.text
            except FileNotFoundError as fnf_error:
                logger.warning(f"Image file not found: {str(fnf_error)}")
                return "", element.text
            except TypeError as type_error:
                logger.warning(f"Unsupported image type: {str(type_error)}")
                return "", element.text
            except Exception as e:
                logger.warning(f"An unexpected error occurred: {str(e)}")
                return "", element.text
        else:
            return "", element.text


    def _is_document_valid(self):
        """Validate the Markdown document for security and format."""
        SecFileCheck(self.file_path, self.MAX_SIZE).check()
        if not self.file_path.endswith(tuple(self.EXTENSION)):
            raise TypeError(f"Unsupported file type '{Path(self.file_path).suffix}'")