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


from typing import Iterator

from pathlib import Path
import fitz
import cv2
import numpy as np
from loguru import logger
from tqdm import tqdm

from paddleocr import PPStructure
from paddleocr.ppstructure.recovery.recovery_to_doc import sorted_layout_boxes
from PIL import Image

from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader

from mx_rag.document.loader.base_loader import BaseLoader as mxBaseLoader
from mx_rag.llm import Img2TextLLM
from mx_rag.utils.file_check import SecFileCheck
from mx_rag.utils.common import validate_params, BOOL_TYPE_CHECK_TIP, Lang


class PdfLoader(BaseLoader, mxBaseLoader):
    EXTENSION = (".pdf",)

    @validate_params(
        vlm=dict(validator=lambda x: isinstance(x, Img2TextLLM) or x is None,
                 message="param must be instance of Img2TextLLM or None"),
        lang=dict(validator=lambda x: isinstance(x, Lang), message="param must be instance of Lang"),
        enable_ocr=dict(validator=lambda x: isinstance(x, bool), message=BOOL_TYPE_CHECK_TIP)
    )
    def __init__(self, file_path: str, vlm: Img2TextLLM = None,
                 lang: Lang = Lang.CH, enable_ocr: bool = False):
        super().__init__(file_path)
        self.enable_ocr = enable_ocr
        self.ocr_engine = None
        self.lang = lang
        self.vlm = vlm

    @staticmethod
    def _reconstruct_line(line, pdf_content):
        line.pop('img')
        for res in line['res']:
            if 'text' in res:
                pdf_content.append(res['text'])
        pdf_content.append("\n")

    def lazy_load(self) -> Iterator[Document]:
        self._check()
        return self._parser() if self.enable_ocr else self._plain_parser()

    def _layout_recognize(self, pdf_document):
        logger.info(f"Processing PDF with {pdf_document.page_count} pages using layout recognition...")
        for page_num in tqdm(range(pdf_document.page_count), desc="Converting PDF pages to images",
                             total=pdf_document.page_count, disable=pdf_document.page_count < 5):
            page = pdf_document.load_page(page_num)
            image_list = page.get_images(full=True) if self.vlm else []

            mat = fitz.Matrix(2, 2)
            # 获取页面的宽高（未放大）
            rect = page.rect
            estimated_width = int(rect.width * 2)  # 放大两倍后的宽度
            estimated_height = int(rect.height * 2)  # 放大两倍后的高度
            # 判断放大两倍后是否超出4096像素
            if estimated_width > 4096 or estimated_height > 4096:
                raise ValueError(f"Page {page_num} 2*size exceed limit 4096 : width={rect.width},"
                                 f" height={rect.height}")

            pm = page.get_pixmap(matrix=mat, alpha=False)

            img = Image.frombytes("RGB", [pm.width, pm.height], pm.samples)
            img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

            yield {
                'image_list': image_list,
                'img': img
            }

    def _parser(self):
        if self.ocr_engine is None:
            try:
                self.ocr_engine = PPStructure(table=True, ocr=True, lang=self.lang.value, layout=True, show_log=False)
            except AssertionError as e:
                logger.error(f"Assertion error: {e}")
                self.ocr_engine = None
                yield Document(page_content="", metadata={"source": self.file_path})
            except Exception as e:
                logger.error(f"paddleOcr init failed, {e}")
                self.ocr_engine = None
                yield Document(page_content="", metadata={"source": self.file_path})
        with fitz.open(self.file_path) as pdf_document:
            pdf_content = []
            for item in self._layout_recognize(pdf_document):
                img = item['img']
                try:
                    ocr_res = self.ocr_engine(img)
                    result = sorted_layout_boxes(ocr_res, img.shape[1])
                except ValueError as e:
                    result = []
                    logger.warning(f"Value error occurred: {str(e)}")
                except Exception as e:
                    result = []
                    logger.warning(f"Failed to process: {str(e)}")
                page_content = []
                for line in result:
                    PdfLoader._reconstruct_line(line, page_content)
                pdf_content.extend(page_content)

                if self.vlm:
                    image_list = item['image_list']
                    yield from self._process_image(pdf_document, image_list)

            one_text = " ".join(pdf_content)
            yield Document(page_content=one_text, metadata={"source": self.file_path,
                                                            "page_count": self._get_pdf_page_count(),
                                                            "type": "text"
                                                            })

    def _plain_parser(self):
        pdf_content, img_base64_list, image_summaries = [], [], []

        pdf_document = fitz.open(self.file_path)
        logger.info(f"Extracting text from PDF with {pdf_document.page_count} pages...")
        for page_num in tqdm(range(pdf_document.page_count), desc="Extracting text"):
            try:
                page = pdf_document.load_page(page_num)
                pdf_content.append(page.get_text("text"))

                if self.vlm:
                    image_list = page.get_images(full=True)
                    yield from self._process_image(pdf_document, image_list)
            except (PermissionError, NotImplementedError) as e:
                logger.warning(f"Page {page_num + 1} access denied: {str(e)}")
            except (ValueError, RuntimeError) as e:
                logger.warning(f"Page {page_num + 1} corrupted: {str(e)}")
            except Exception as e:
                logger.warning(f"Failed to extract text from page {page_num + 1}: {str(e)}")
        pdf_document.close()

        one_text = " ".join(pdf_content)
        one_text and (yield Document(page_content=one_text, metadata={"source": self.file_path,
                                                                      "page_count": self._get_pdf_page_count(),
                                                                      "type": "text"
                                                                      }))

    def _process_image(self, pdf_document, image_list):
        """
        处理图像和 OCR 逻辑的公共函数。
        """
        for _, img in enumerate(image_list):
            xref = img[0]
            base_image = pdf_document.extract_image(xref)
            image_data = base_image["image"]
            img_base64, img_sumy = self._interpret_image(image_data, self.vlm)
            if img_base64 and img_sumy:
                yield Document(
                    page_content=img_sumy,
                    metadata={
                        "source": self.file_path,
                        "image_base64": img_base64,
                        "type": "image"
                    }
                )

    def _get_pdf_page_count(self):
        try:
            with fitz.open(self.file_path) as pdf_document:
                pdf_page_count = pdf_document.page_count
                return pdf_page_count
        except fitz.FileDataError:
            logger.error(f"Invalid or corrupted PDF file: {self.file_path}")
            return 0
        except fitz.PermissionError:
            logger.error(f"PDF is encrypted/protected, cannot read page count: {self.file_path}")
            return 0
        except Exception as e:
            logger.error(f"Failed to get PDF page count: {e}")
            return 0

    def _check(self):
        SecFileCheck(self.file_path, self.MAX_SIZE).check()
        if not self.file_path.endswith(PdfLoader.EXTENSION):
            raise TypeError(f"type '{Path(self.file_path).suffix}' is not support")
        _pdf_page_count = self._get_pdf_page_count()
        if _pdf_page_count > self.MAX_PAGE_NUM:
            raise ValueError(f"PDF has {_pdf_page_count} pages, "
                             f"which exceeds the maximum limit of {self.MAX_PAGE_NUM} pages")
        logger.info(f"Starting to process PDF file: {self.file_path}")
