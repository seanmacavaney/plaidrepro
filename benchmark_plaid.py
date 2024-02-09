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

#for ds in ['dl19', 'devs', 'dl20']:
for ds in ['dl19-first']:
  if ds == 'dl19':
    queries = {q.query_id: q.text for q in ir_datasets.load('msmarco-passage/trec-dl-2019/judged').queries}
  if ds == 'dl20':
    queries = {q.query_id: q.text for q in ir_datasets.load('msmarco-passage/trec-dl-2020/judged').queries}
  elif ds == 'devs':
    queries = {q.query_id: q.text for q in ir_datasets.load('msmarco-passage/dev/small').queries}
  if ds == 'dl19-first':
    queries = {q.query_id: q.text for q in list(ir_datasets.load('msmarco-passage/trec-dl-2019/judged').queries)[:1]}

  print('encoding')
  enc_Q = searcher.encode(list(queries.values()))
  print('starting')

  torch.set_num_threads(1)

#  for ncells in [1, 2, 4, 8, 16]:
  for ncells in [4]:
    for centroid_score_threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
#      for ndocs in [256, 512, 1024, 2048, 4096, 8192]:
      for ndocs in [256, 1024, 4096, 8192]:
          path = f'results/{ds}.plaid.ncells-{ncells}.cst-{centroid_score_threshold}.ndocs-{ndocs}.run.gz'
          if os.path.exists(path):
              continue
          searcher.config.ncells = ncells
          searcher.config.centroid_score_threshold = centroid_score_threshold
          searcher.config.ndocs = ndocs
          t0 = time()
          results = searcher.search_all_Q(queries, enc_Q, k=1000)
          t1 = time()
          res_dict = results.todict()
          with gzip.open(path, 'wt') as fout:
            for qid in res_dict.keys():
              for did, rank, score in res_dict[qid]:
                fout.write(f'{qid} 0 {did} {rank} {score} run\n')
          with open(f'results/{ds}.plaid.ncells-{ncells}.cst-{centroid_score_threshold}.ndocs-{ndocs}.time', 'wt') as fout:
            fout.write(f'{(t1-t0)/len(queries)*1000}')
          print(f'{ncells=} {centroid_score_threshold=} {ndocs=} time={(t1-t0)/len(queries)*1000}')
