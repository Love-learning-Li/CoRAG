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


import unittest
from unittest.mock import MagicMock

from mx_rag.compress.rerank_compressor import RerankCompressor
from mx_rag.reranker import Reranker


class RerankCompressorTestCase(unittest.TestCase):

    def test_success(self):
        mock_reranker = MagicMock(spec=Reranker)
        mock_reranker.rerank.return_value = [0.11096191, 0.08996582, 0.07006836]
        compressor = RerankCompressor(reranker=mock_reranker)
        context = """近平指出，巴林是中国在海湾地区的好朋友、好伙伴。两国虽然国情不同，但始终以诚相待、友好相处。近年来，在我们共同引领下，中巴关系平稳健康发展。今年是中巴建交35周年，双方一致同意将中巴关系提升为全面战
        略伙伴关系，这是中巴关系史上新的里程碑。中方愿同巴方一道，发展好中巴全面战略伙伴关系，更好造福两国人民。
        习近平强调，中方坚定支持巴方维护国家主权、安全、稳定，支持巴方走独立自主发展道路，支持巴林“2030经济发展愿景”和多元化发展战略，愿同巴方加强能源、投资、交通、新能源、数字经济等领域合作，推动高质量共建“一带
        取得更多成果。双方要密切人文交流和人员往来，持续巩固中巴友好民意基础。中方主张不同制度、不同文明国家相互尊重、和平共处，支持中东地区国家增进团结协作，实现和平和解，促进发展振兴，愿同包括巴林在内的地区国家一道，推动中国
        同海合会国家关系取得更大发展，办好第二届中阿峰会，加快推进中阿命运共同体建设。加强在联合国等多边平台沟通协调，践行真正的多边主义，维护广大发展中国家共同利益。
        哈马德表示，中国是伟大的国家，为巴林国家建设提供了大量支持，巴方深表感谢。巴方希望以建立全面战略伙伴关系为契机，对接两国发展战略，密切各领域务实合作，助力巴林实现多元化发展。巴方高度赞赏并完全赞同中方秉持的高尚
        念和理性智慧的政策主张。中国发展好了，其他发展中国家才能发展好，世界多极化进程才能持续推进。巴方坚信，中国必将实现中华民族伟大复兴，并为世界和平繁荣作出更大贡献。巴方恪守一个中国原则，支持中国实现和平统一。愿同中方密切多边协作，
        更好保障各国人民享有平等的生存权和发展权。巴方愿同中方一道，推动尽早达成海合会－中国自由贸易协定，发扬阿中友好精神，携手构建面向新时代的阿中命运共同体。
        哈马德通报了近期第三十三届阿盟首脑会议情况，特别是阿拉伯国家在巴勒斯坦问题上的立场以及为推动尽快结束加沙冲突所作的努力，赞赏并感谢中方始终秉持正义立场，期待中方发挥更大作用。习近平强调，中巴双方在巴勒斯坦问题上
        致。中方赞赏阿盟首脑会议就巴以问题发出阿拉伯国家共同声音，愿同巴林和其他阿拉伯国家一道努力，推动巴勒斯坦问题早日得到全面、公正、持久解决。
        会谈后，两国元首共同见证签署关于投资、绿色低碳、电子商务、数字经济等领域多项双边合作文件。"""
        question = "请给上述内容起一个标题？"
        res = compressor.compress_texts(context, question, 0.4)
        self.assertLess(len(res), len(context))

    def test_init_fail(self):
        with self.assertRaises(ValueError):
            RerankCompressor(reranker="")

        with self.assertRaises(ValueError):
            RerankCompressor(reranker=None)

        with self.assertRaises(ValueError):
            mock_reranker = MagicMock(spec=Reranker)
            RerankCompressor(reranker=mock_reranker, splitter="")

    def test_compress_texts_fail(self):
        mock_reranker = MagicMock(spec=Reranker)
        mock_reranker.rerank.return_value = [0.11096191, 0.08996582, 0.07006836]
        compressor = RerankCompressor(reranker=mock_reranker)
        context = "近平指出，巴林是中国在海湾地区的好朋友、好伙伴。"
        question = "请给上述内容起一个标题？"
        with self.assertRaises(ValueError):
            compressor.compress_texts("", question, 0.4)

        with self.assertRaises(ValueError):
            compressor.compress_texts(context, "", 0.4)

        with self.assertRaises(ValueError):
            compressor.compress_texts(context, question, 1.1)

        with self.assertRaises(ValueError):
            compressor.compress_texts(context, question, -0.1)

        with self.assertRaises(ValueError):
            compressor.compress_texts(context, question, "0.4")

        with self.assertRaises(ValueError):
            compressor.compress_texts(context, question, context_reorder="")

if __name__ == '__main__':
    unittest.main()
