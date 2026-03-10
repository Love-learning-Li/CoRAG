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



__all__ = [
    "DocxLoader",
    "ExcelLoader",
    "PdfLoader",
    "PowerPointLoader",
    "ImageLoader",
    "MarkdownLoader",
    "BaseLoader"
]

from mx_rag.document.loader.docx_loader import DocxLoader
from mx_rag.document.loader.md_loader import MarkdownLoader
from mx_rag.document.loader.pdf_loader import PdfLoader
from mx_rag.document.loader.excel_loader import ExcelLoader
from mx_rag.document.loader.ppt_loader import PowerPointLoader
from mx_rag.document.loader.image_loader import ImageLoader
from mx_rag.document.loader.base_loader import BaseLoader
