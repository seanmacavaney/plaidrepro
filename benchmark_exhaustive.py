import os
import gzip
import torch
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert.data import Queries, Collection
from colbert import Indexer, Searcher
from time import time
import ir_datasets
checkpoint = 'downloads/colbertv2.0'
nbits = 2
index_name = f'{nbits}bits'
with Run().context(RunConfig(experiment='notebook')):
    searcher = Searcher(index=index_name)

for ds in ['test', 'dl19', 'devs', 'dl20']:
  print(ds, 'started')
  if ds == 'test':
    queries = {'0': 'test'}
  elif ds == 'dl19':
    queries = {q.query_id: q.text for q in ir_datasets.load('msmarco-passage/trec-dl-2019/judged').queries}
  elif ds == 'dl20':
    queries = {q.query_id: q.text for q in ir_datasets.load('msmarco-passage/trec-dl-2020/judged').queries}
  elif ds == 'devs':
    queries = {q.query_id: q.text for q in ir_datasets.load('msmarco-passage/dev/small').queries}

  print('encoding')
  enc_Q = searcher.encode(list(queries.values()))
  print('starting')

  torch.set_num_threads(28)

  path = f'results/{ds}.exhuastive.run.gz'
  if os.path.exists(path):
    continue
  print('searching')
  results = searcher.search_exhaustive_Q(queries, enc_Q, k=1000)
  res_dict = results.todict()
  with gzip.open(path, 'wt') as fout:
    for qid in res_dict.keys():
      for rank, (score, did) in enumerate(res_dict[qid]):
        fout.write(f'{qid} 0 {did} {rank} {score} run\n')
  print(ds, 'finished')
