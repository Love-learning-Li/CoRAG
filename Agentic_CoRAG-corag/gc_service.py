#currently only preloading index is supported, it is expected to allow load and manage multiple gc sources

import gc
import argparse
import uvicorn
import json 

from fastapi import FastAPI
from pydantic import BaseModel

from kgc.rag.graphrag_simireranker import GraphRAGModel
from kgc.scheme_embedding import EMBWorker
from vdb.faiss_vdb import FaissVector
from gdb.networkx_gdb import NetworkxGraph

import logging


logger = logging.getLogger(__name__)


class RetrieveInput(BaseModel):
    query: str
    # depth: int
    # max_passages: int



class RAG_Service():
    def __init__(self):
        self.initialized = False
        self.RAG = None
        self.emb = None
    
    def load_index(self, args):
        if not self.initialized:
            self.args = args

            g0 = NetworkxGraph(path_or_name=args.gdb_path)
            v0 = FaissVector(args.embedding_dim, args.vdb_path, type="IndexFlatIP")
            v1 = FaissVector(args.embedding_dim, args.vdb_triples_path, type="IndexFlatIP") # will be created if path not existed
            v2 = FaissVector(args.embedding_dim, args.vdb_passages_path, type="IndexFlatIP")
            self.RAG = GraphRAGModel(args, None, None, g0, v0, v1, v2)
            self.initialized = True


    def clear_index(self):
        del self.RAG
        self.initialized = False
        gc.collect()

    def retrieve(self, query, max_passages = 10, depth = 2):
        self.RAG.subgraph = NetworkxGraph(is_digraph=True)
        top_k_triples,top_k_triple_elements, distance = self.RAG.query_to_triples(query, top_k=10)
        logging.info("Query: %s", query)
        logging.info("Shot Triples: %s", top_k_triples)
        retrieved_passages, _ = self.RAG.get_passages_by_ppr(query, top_k_triples, top_k_triple_elements, distance, depth)
        self.RAG.reset_subgraph()


        return retrieved_passages[:max_passages]

def getargs():
    global args
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast_retrieval", type=bool, default=False)
    parser.add_argument("-m", "--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--model_path", type=str, default="/data0/xyao/models/Qwen2.5-7B-Instruct")
    parser.add_argument("--model_url", type=str, default='http://10.170.30.9/v1/infers/f13f111a-7b8f-4f35-89ae-0fd3f49b1304/v1/chat/completions?endpoint=infer-app-modelarts-cn-southwest-2')
    parser.add_argument("--model_name", type=str, default="qwen2.5-72b")
    parser.add_argument("--server_type", type=str, default="vllm-ascend")  # mindie or vllm-ascend [vllm format, openai format]
    parser.add_argument('--headers', type=str, 
                    default='{"Content-Type": "application/json", "X-Apig-AppCode":"7edfd51d270b49aea844be9d4ff5d2fa266ebf3fc33f41b1ad2398279d2e8595"}', 
                    help='Headers in JSON format')
    parser.add_argument("--model_online", type=bool, default=True)
    parser.add_argument("--llmworker_online", type=bool, default=True)

    # emb service parameters
    parser.add_argument('--emb_model_url', type=str, default='http://7.242.107.236:9613/v1/embeddings')  # vllm-ascend--> output:['data'][0]['embedding']
    parser.add_argument('--emb_model_headers', type=str, default='{"Content-Type": "application/json"}') # service header
    parser.add_argument("--emb_model_name", type=str, default="bge-large-zh-v1.5")  # bge-large-zh(en)-v1.5 support max token len: 512
    parser.add_argument('--emb_model_max_len', type=int, default=500)  # maximum length of characters
    parser.add_argument("--embedding_dim", type=int, default=1024) # dimension of the embedded model
    parser.add_argument("--reranker_model", type=str, default='/data/linquan/models/Qwen3-Reranker-0.6B') #input None to disable reranker process.
    parser.add_argument('--num_workers_per_server', type=int, default=8)  # default 8 threads, depends on #CPUcores


    parser.add_argument("--preload_index", type=bool, default=True)
    parser.add_argument("--gdb_path", type=str, default=f"data_ksc/merged_graph/2wiki.pkl") # path of KG
    parser.add_argument("--vdb_path", type=str, default=f"data_ksc/merged_graph/vdb_2wiki_bge.faiss") # path of the vecDB storing all node
    parser.add_argument("--vdb_triples_path", type=str, default=f"data_ksc/merged_graph/vdb_2wiki_triples.faiss") # path of the vecDB storing triples node
    parser.add_argument("--vdb_passages_path", type=str, default=f"data_ksc/merged_graph/vdb_2wiki_passages.faiss") # path of the vecDB storing passages node
    parser.add_argument("--image_path", type=str, default=f"./dataset/vdb/image_result")


    # parser.add_argument("--gdb_path", type=str, default=f"/data0/j84/dbs/serhii/data_ksc/merged_graph/tmp_20250724_10.pkl")
    # parser.add_argument("--vdb_path", type=str, default=f"/data0/j84/dbs/serhii/data_ksc/merged_graph/vdb_tmp_20250724_10.faiss")
    # parser.add_argument("--vdb_triples_path", type=str, default=f"/data0/j84/dbs/serhii/data_ksc/merged_graph/vdb_tmp_20250724_10_triples.faiss")
    # parser.add_argument("--vdb_passages_path", type=str, default=f"/data0/j84/dbs/serhii/data_ksc/merged_graph/vdb_tmp_20250724_10_passages.faiss")
    # parser.add_argument("--image_path", type=str, default=f"/data0/j84/dbs/serhii/data_ksc/image_result")


    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--metric", type=str, choices=["choice", "generation"], default="generation")

    # rag parameters
    parser.add_argument("--chunk_size", type=int, default=1000)
    parser.add_argument("--chunk_overlap", type=int, default=100)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--subgraph_depth", type=int, default=2)
    parser.add_argument("--text", type=bool, default=True)

    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.01)
    parser.add_argument("--top_p", type=float, default=0.9)

    args = parser.parse_args()
    
app = FastAPI()
rag = RAG_Service()
args = None

@app.on_event("startup")
async def startup():
    getargs()
    if args.preload_index:
        rag.load_index(args)

@app.get("/")
async def root():
    return {"status": f"{'ok' if rag.initialized else 'index uninitalized'}"}

@app.post("/retrieve")
async def retrieve(input: RetrieveInput):
    logger.info(f"Retrieve: {input.query}")
    response = rag.retrieve(input.query)
    logger.info(f"context: {json.dumps(response)}")
    return response


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default = 8035)
    uv_args = parser.parse_args()
    uvicorn.run("gc_service:app", host=uv_args.host, port=uv_args.port, reload=False)
