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

import json
import unittest
from typing import Dict
from unittest.mock import patch, Mock

from mx_rag.embedding.service.clip_embedding import CLIPEmbedding
from mx_rag.utils import ClientParam
from mx_rag.utils.url import Result


def mock_post(url: str, body: str, headers: Dict):
    resp_data = {'data': [{"embedding": [0.1, 0.2, 0.3]} for _ in range(2)]}
    return Result(True, json.dumps(resp_data))


class TestCLIPEmbedding(unittest.TestCase):
    def test_embed_documents_success(self, ):
        with patch('mx_rag.utils.url.RequestUtils.post', Mock(side_effect=mock_post)):
            clip_embedding = CLIPEmbedding(
                url='http://valid-url.com',
                client_param=ClientParam(use_http=True)
            )
            texts = ["Sample text 1", "Sample text 2"]
            embeddings = clip_embedding.embed_documents(texts)
            self.assertEqual(len(embeddings), 2)
            self.assertEqual(embeddings, [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]])

    def test_embed_images_success(self, ):
        with patch('mx_rag.utils.url.RequestUtils.post', Mock(side_effect=mock_post)):
            clip_embedding = CLIPEmbedding(
                url='http://valid-url.com',
                client_param=ClientParam(use_http=True)
            )
            images = ["imagedata1", "imagedata2"]
            embeddings = clip_embedding.embed_images(images)
            self.assertEqual(len(embeddings), 2)
            self.assertEqual(embeddings, [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]])

    def test_embed_images_failure(self):
        with patch('mx_rag.utils.url.RequestUtils.post', Mock(side_effect=mock_post)):
            clip_embedding = CLIPEmbedding(
                url='http://valid-url.com',
                client_param=ClientParam(use_http=True)
            )
            images = ["^image data 1", "image data 2"]
            with self.assertRaises(ValueError):
                _ = clip_embedding.embed_images(images)


if __name__ == '__main__':
    unittest.main()
