import pyterrier as pt
pt.init()
import gzip
import torch
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert.data import Queries, Collection
from colbert import Indexer, LadrSearcher
from time import time
from pyterrier_pisa import PisaIndex
from pyterrier_adaptive import CorpusGraph
import ir_datasets
#from argparse import ArgumentParser
#parser = ArgumentParser()
#parser.add_argument('--num_results', '-n', default=1000)
#parser.add_argument('--num_neighbours', '-k', default=128)
#parser.add_argument('--num_hops', '-h', default=1)
#args = parser.parse_args()
checkpoint = 'downloads/colbertv2.0'
nbits = 2
index_name = f'{nbits}bits'
index = PisaIndex.from_dataset('msmarco_passage', threads=1)
bm25 = index.bm25(num_results=1000)
graph = CorpusGraph.load('/home/sean/data/indices/msmarco-passage.gbm25.128')
with Run().context(RunConfig(experiment='notebook')):
    searcher = LadrSearcher(index=index_name, first_pass=bm25, corpus_graph=graph, proactive_steps=0)

ds = 'dl20'
if ds == 'dl19':
  queries = {q.query_id: q.text for q in ir_datasets.load('msmarco-passage/trec-dl-2019/judged').queries}
if ds == 'dl20':
  queries = {q.query_id: q.text for q in ir_datasets.load('msmarco-passage/trec-dl-2020/judged').queries}
elif ds == 'devs':
  queries = {q.query_id: q.text for q in ir_datasets.load('msmarco-passage/dev/small').queries}

print('encoding')
enc_Q = searcher.encode(list(queries.values()))
print('starting')

torch.set_num_threads(1)

#for num_results in [10, 100, 200, 500, 1000, 2000, 5000, 10000]:
for num_results in [100, 500, 1000]:
  for num_neighbours in [64, 128]:
    for depth in [10, 20, 50, 100]:
#for num_results in [1]:
#  for num_neighbours in [0]:
#    for depth in [0]:
      searcher.first_pass = index.bm25(num_results=num_results)
      searcher.adaptive_depth = depth
      searcher.graph = graph.to_limit_k(num_neighbours)
      t0 = time()
      results = searcher.search_all_Q(queries, enc_Q, k=1000)
      t1 = time()
      res_dict = results.todict()
      with gzip.open(f'results/{ds}.ladr.num_results-{num_results}.num_neighbours-{num_neighbours}.depth-{depth}.run.gz', 'wt') as fout:
        for qid in res_dict.keys():
          for did, rank, score in res_dict[qid]:
            fout.write(f'{qid} 0 {did} {rank} {score} run\n')
      with open(f'results/{ds}.ladr.num_results-{num_results}.num_neighbours-{num_neighbours}.depth-{depth}.time', 'wt') as fout:
        fout.write(f'{(t1-t0)/len(queries)*1000}')
      print(f'{num_results=} {num_neighbours=} {depth=} time={(t1-t0)/len(queries)*1000}')
