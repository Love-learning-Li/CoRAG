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

TRIPLE_INSTRUCTIONS_EN = {
    "entity_relation": """Given a passage, summarize all the important entities and the relations between them in 
    a concise manner. Relations should briefly capture the connections between entities, without repeating information 
    from the head and tail entities. The entities should be as specific as possible. Exclude pronouns from 
    being considered as entities. The output should strictly adhere to the following JSON format:
    [
        {
            "Head": "{a noun}",
            "Relation": "{a verb}",
            "Tail": "{a noun}",
        },
        {
            "Head": "China",
            "Relation": "Capital",
            "Tail": "Beijing",
        },
        {
            "Head": "Dog",
            "Relation": "like",
            "Tail": "bone",
        },
        {
            "Head": "Mao Zedong",
            "Relation": "Father",
            "Tail": "Mao Anying",
        },
        {
            "Head": "China Shipbuilding Materials Yungui Co., Ltd.",
            "Relation": "Established",
            "Tail": "May 31, 1990",
        },
        {
            "Head": "Company",
            "Relation": "Address",
            "Tail": "Kunming City, Yunnan Province",
        },
        {
            "Head": "Company",
            "Relation": "Operation",
            "Tail": "Electronics",
        },
        {
            "Head": "Year 1999",
            "Relation": "Before",
            "Tail": "Year 2000",
        },
        {
            "Head": "Year 2001",
            "Relation": "After",
            "Tail": "Year 2000",
        },
    ]
    Here is the passage:\n""",

    "event_entity": """Please analyze and summarize the participation relations between the events and entities 
    in the given paragraph. Each event is a single independent sentence. Additionally, identify all the entities 
    that participated in the events. Do not use ellipses. Please strictly output in the following JSON format:
    [
        {
            "Event": "{a simple sentence describing an event}",
            "Entity": ["entity 1", "entity 2", "..."]
        }...
    ]
    Here is the passage:\n""",

    "event_relation": """Please analyze and summarize the relationships between the events in the paragraph. 
    Each event is a single independent sentence. Identify temporal and causal relationships between the events using the following types: before, after, at the same time, because, and as a result. Each extracted triple should be specific, meaningful, and able to stand alone.  Do not use ellipses.  The output should strictly adhere to the following JSON format:
    [
        {
            "Head": "{a simple sentence describing the event 1}",
            "Relation": "{temporal or causality relation between the events}",
            "Tail": "{a simple sentence describing the event 2}"
        }...
    ]
    Here is the passage:\n"""

}

EVENT_PROMPT_EN = '''I will give you an EVENT. You need to give several phrases containing 1-2 words for the 
            ABSTRACT EVENT of this EVENT.
            You must return your answer in the following format: phrases1, phrases2, phrases3,...
            You can't return anything other than answers.
            These abstract event words should fulfill the following requirements.
            1. The ABSTRACT EVENT phrases can well represent the EVENT, and it could be the type of the EVENT or the related concepts of the EVENT.    
            2. Strictly follow the provided format, do not add extra characters or words.
            3. Write at least 3 or more phrases at different abstract level if possible.
            4. Do not repeat the same word and the input in the answer.
            5. Stop immediately if you can't think of any more phrases, and no explanation is needed.

            EVENT: A man retreats to mountains and forests
            Your answer: retreat, relaxation, escape, nature, solitude
            
            EVENT: A cat chased a prey into its shelter
            Your answer: hunting, escape, predation, hidding, stalking

            EVENT: Sam playing with his dog
            Your answer: relaxing event, petting, playing, bonding, friendship

            EVENT: [EVENT]
            Your answer:
            '''

ENTITY_PROMPT_EN = '''I will give you an ENTITY. You need to give several phrases containing 1-2 words for the 
            ABSTRACT ENTITY of this ENTITY.
            You must return your answer in the following format: phrases1, phrases2, phrases3,...
            You can't return anything other than answers.
            These abstract intention words should fulfill the following requirements.
            1. The ABSTRACT ENTITY phrases can well represent the ENTITY, and it could be the type of the ENTITY or 
            the related concepts of the ENTITY.
            2. Strictly follow the provided format, do not add extra characters or words.
            3. Write at least 3 or more phrases at different abstract level if possible.
            4. Do not repeat the same word and the input in the answer.
            5. Stop immediately if you can't think of any more phrases, and no explanation is needed.

            ENTITY: Soul
            CONTEXT: premiered BFI London Film Festival, became highest-grossing Pixar release
            Your answer: movie, film

            ENTITY: Thinkpad X60
            CONTEXT: Richard Stallman announced he is using Trisquel on a Thinkpad X60
            Your answer: Thinkpad, laptop, machine, device, hardware, computer, brand

            ENTITY: Harry Callahan
            CONTEXT: bluffs another robber, tortures Scorpio
            Your answer: person, Amarican, character, police officer, detective

            ENTITY: Black Mountain College
            CONTEXT: was started by John Andrew Rice, attracted faculty
            Your answer: college, university, school, liberal arts college

            EVENT: 1st April
            CONTEXT: Utkal Dibas celebrates
            Your answer: date, day, time, festival

            ENTITY: [ENTITY]
            CONTEXT: [CONTEXT]
            Your answer:
            '''

RELATION_PROMPT_EN = '''I will give you an RELATION. You need to give several phrases containing 1-2 words for 
            the ABSTRACT RELATION of this RELATION.
            You must return your answer in the following format: phrases1, phrases2, phrases3,...
            You can't return anything other than answers.
            These abstract intention words should fulfill the following requirements.
            1. The ABSTRACT RELATION phrases can well represent the RELATION, and it could be the type of the RELATION 
            or the simplest concepts of the RELATION.
            2. Strictly follow the provided format, do not add extra characters or words.
            3. Write at least 3 or more phrases at different abstract level if possible.
            4. Do not repeat the same word and the input in the answer.
            5. Stop immediately if you can't think of any more phrases, and no explanation is needed.
            
            RELATION: participated in
            Your answer: become part of, attend, take part in, engage in, involve in

            RELATION: be included in
            Your answer: join, be a part of, be a member of, be a component of

            RELATION: [RELATION]
            Your answer:
            '''

TRIPLE_INSTRUCTIONS_CN = {
    "entity_relation": """
## 目标
请从以下文本中提取所有重要实体及其关系，并严格遵守以下规则：

## 要求
1. 实体必须为名词，尽量简洁；  
2. 关系必须为一个动词，准确描述“头实体”与“尾实体”之间的具体联系，且不得重复头、尾实体的字面信息；  
3. 头实体与尾实体均不得为“是”，不得使用代词；
4. 实体和关系不能为空字符串，不能为仅包含标点符号的字符串；
4. 输出必须采用下列 JSON 格式，禁止添加、删除或修改任何字段：

[
    {
        "头实体": "{名词}",
        "关系": "{动词}",
        "尾实体": "{名词}"
    }
]

## 示例
[
    {
        "头实体": "中国",
        "关系": "首都",
        "尾实体": "北京",
    },
    {
        "头实体": "小狗",
        "关系": "喜欢",
        "尾实体": "骨头",
    },
    {
        "头实体": "毛泽东",
        "关系": "父亲",
        "尾实体": "毛岸英",
    },
    {
        "头实体": "中国船舶工业物资云贵有限公司",
        "关系": "成立",
        "尾实体": "1990月05月31日",
    },
    {
        "头实体": "公司",
        "关系": "地址",
        "尾实体": "云南省昆明市",
    },
    {
        "头实体": "公司",
        "关系": "经营",
        "尾实体": "电子器件",
    },
    {
        "头实体": "1999年",
        "关系": "早于",
        "尾实体": "2000年",
    },
    {
        "头实体": "2001年",
        "关系": "晚于",
        "尾实体": "2000年",
    }
]
## 待分析文本
""",

    "event_entity": """
## 目标
请对以下段落逐句进行事件抽取，并识别每个事件所涉及的全部实体。

## 要求
1. 一句视为一个独立事件，保留原句，不做任何省略；
2. 列出每个事件直接参与的所有实体，不重复、不遗漏；
3. 输出严格使用下方 JSON 格式，不允许添加或删减字段。

## JSON 格式
[
    {
        "事件": "{原句}",
        "实体": ["实体1", "实体2", "..."]
    }
]

## 待分析文本
""",

    "event_relation": """
## 目标
请对以下段落逐句抽取事件，并识别它们之间的时间或因果关系。  

## 要求  
1. 一句视为一个独立事件，保留原句，不做任何省略。  
2. 仅使用指定关系类型：在之前、在之后、同时、因为、结果。  
3. 每个三元组中的“头事件”与“尾事件”均须为段落中完整原句，且语义对应具体、可独立理解。
4. “头事件”和“尾事件”不能为空字符串，且不能重叠；
4. 关系不能为空字符串；
5. 输出严格使用下方 JSON 格式，不允许添加、删减或省略任何字段。  

## JSON 格式  
[
    {
        "头事件": "{事件1完整原句}",
        "关系": "{在之前|在之后|同时|因为|结果}",
        "尾事件": "{事件2完整原句}"
    }
]

## 待分析文本
""",
}

EVENT_PROMPT_CN = '''
## 目标
给定一个事件，提供多个短语来表示该事件的抽象概念。 

## 输出格式
短语1, 短语2, 短语3,...

## 要求
* 短语应准确代表事件，可以是其类型或相关概念。
* 短语包含1-2个词。
* 短语不能包含空格和换行符，不能包含标点符号，但可以包含连字符。
* 严格遵循输出格式，不添加任何额外字符。
* 尽可能提供3到10个不同抽象层次的短语。
* 不重复使用与事件或已有短语相同的词语。
* 如果无法生成更多短语，请立即停止。

## 示例

事件：一名男子隐居于山林
概念：隐居, 放松, 逃避, 自然, 孤独

事件：一只猫追逐猎物进入它的藏身之处
概念：狩猎, 逃避, 捕食, 躲藏, 潜行

事件：山姆和他的狗玩耍
概念：放松的活动, 爱抚, 玩耍, 联结, 友谊

事件：中国船舶工业物资云贵有限公司，成立日期：1990年3月31日成立，住所：云南省昆明市
概念：公司成立, 公司地点, 公司住所, 省, 市, 成立时间, 年月日

## 待分析文本
事件：[EVENT]
概念：
'''

ENTITY_PROMPT_CN = '''
## 目标
给定一个实体及其背景，提供多个短语来表示该实体的抽象概念。

## 输出格式
短语1, 短语2, 短语3,...

## 要求
* 短语应准确代表实体，可以是其类型或相关概念。
* 短语包含1-2个词。
* 短语不能包含空格和换行符，不能包含标点符号，但可以包含连字符。
* 严格遵循输出格式，不添加任何额外字符。
* 尽可能提供3到10个不同抽象层次的短语。
* 不重复使用与实体或已有短语相同的词语。
* 如果无法生成更多短语，请立即停止。

## 示例
实体：灵魂
背景：在BFI伦敦电影节首映，成为皮克斯票房最高的影片
概念：电影, 影片

实体：Thinkpad X60
背景：理查德·斯托曼宣布他在Thinkpad X60上使用Trisquel
概念：Thinkpad, 笔记本, 机器, 设备, 硬件, 电脑, 品牌

实体：哈里·卡拉汉
背景：欺骗另一个抢劫犯，折磨天蝎座
概念：人, 美国人, 角色, 警察, 侦探

实体：黑山学院
背景：由约翰·安德鲁·赖斯创立，吸引了教师
概念：学院, 大学, 学校, 文理学院

## 待分析文本
实体：[ENTITY]
背景：[CONTEXT]
概念：
'''

RELATION_PROMPT_CN = '''
## 目标
给定一个关系，提供多个短语来表示该关系的抽象概念。

## 输出格式
短语1, 短语2, 短语3,...

## 要求
* 短语应准确代表关系，可以是其类型或相关概念。
* 短语包含1-2个词。
* 短语不能包含空格和换行符，不能包含标点符号，但可以包含连字符。
* 严格遵循输出格式，不添加任何额外字符。
* 尽可能提供3到10个不同抽象层次的短语。
* 不重复使用与关系或已有短语相同的词语。
* 如果无法生成更多短语，请立即停止。

## 示例

关系：参与
概念：成为一部分, 参加, 投入, 涉及

关系：被包括在内
概念：加入, 成为一部分, 成为成员, 成为组成部分

## 待分析文本
关系：[RELATION]
概念：
'''
