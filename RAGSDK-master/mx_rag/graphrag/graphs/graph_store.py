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


from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union, Callable


class GraphStore(ABC):
    @abstractmethod
    def add_node(self, node: str, **attrs: Any) -> None:
        pass

    @abstractmethod
    def add_nodes_from(self, nodes: Iterable[str], **attrs: Any) -> None:
        pass

    @abstractmethod
    def remove_node(self, node: str) -> None:
        pass

    @abstractmethod
    def has_node(self, node: str) -> bool:
        pass

    @abstractmethod
    def get_node_attributes(self, node: str, key: Optional[str] = None) -> Union[Dict[str, Any], Any]:
        pass

    @abstractmethod
    def set_node_attributes(self, attributes: Dict[str, Dict[str, Any]], name: str) -> None:
        pass

    @abstractmethod
    def update_node_attribute(self, node: str, key: str, value: Any, append: bool = False) -> None:
        pass

    @abstractmethod
    def update_node_attributes_batch(
            self, node_updates: List[Tuple[str, Dict[str, Any]]], batch_size: int = 1024, append: bool = False
    ) -> None:
        pass

    @abstractmethod
    def get_nodes(self, with_data: bool = True) -> Union[List[str], List[Tuple[str, Dict[str, Any]]]]:
        pass

    @abstractmethod
    def get_nodes_by_attribute(self, key: str, value: Any) -> List[Any]:
        pass

    @abstractmethod
    def get_nodes_containing_attribute_value(self, key: str, value: str) -> List[Any]:
        pass

    @abstractmethod
    def add_edge(self, u: str, v: str, **attr: Any) -> None:
        pass

    @abstractmethod
    def add_edges_from(self, edges: Iterable[Tuple[str, str]]) -> None:
        pass

    @abstractmethod
    def remove_edge(self, u: str, v: str) -> None:
        pass

    @abstractmethod
    def has_edge(self, u: str, v: str) -> bool:
        pass

    @abstractmethod
    def get_edge_attributes(
            self, u: str, v: str, key: Optional[str] = None
    ) -> Union[Dict[str, Any], Any]:
        pass

    @abstractmethod
    def update_edge_attributes_batch(
            self, edge_updates: List[Tuple[str, str, Dict[str, Any]]], batch_size: int = 1024, append: bool = False
    ) -> None:
        pass

    @abstractmethod
    def update_edge_attribute(
            self, u: str, v: str, key: str, value: Any, append: bool = False
    ) -> None:
        pass

    @abstractmethod
    def get_edges(
            self, with_data: bool = True
    ) -> Union[List[Tuple[str, str]], List[Tuple[str, str, Dict[str, Any]]]]:
        pass

    @abstractmethod
    def get_edge_attribute_values(self, key: str) -> List[str]:
        pass

    @abstractmethod
    def in_degree(self, node: str) -> int:
        pass

    @abstractmethod
    def out_degree(self, node: str) -> int:
        pass

    @abstractmethod
    def neighbors(self, node: str) -> Iterable[str]:
        pass

    @abstractmethod
    def successors(self, node: str) -> Iterable[str]:
        pass

    @abstractmethod
    def predecessors(self, node: str) -> Iterable[str]:
        pass

    @abstractmethod
    def number_of_nodes(self) -> int:
        pass

    @abstractmethod
    def number_of_edges(self) -> int:
        pass

    @abstractmethod
    def density(self) -> float:
        pass

    @abstractmethod
    def connected_components(self) -> Iterable[Set[str]]:
        pass

    @abstractmethod
    def subgraph(self, nodes: Iterable[str]):
        pass

    @abstractmethod
    def get_subgraph_edges(
            self, nodes: Iterable[str], with_data: bool = True
    ) -> Union[List[Tuple[str, str]], List[Tuple[str, str, Dict[str, Any]]]]:
        pass

    @abstractmethod
    def save(self, output_path: str, encrypt_fn: Callable = None):
        pass
