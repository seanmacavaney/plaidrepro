import gzip
import json
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

dataset = ir_datasets.load('msmarco-passage/trec-dl-2019/judged')
query = next(iter(dataset.queries))

print('encoding')
enc_Q = searcher.encode([query.text])
print('starting')

for threads in [16, 8, 4, 2, 1]:
  torch.set_num_threads(threads)
  #for count in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]:
  for count in [32768]:
    times = []
    for i in range(100):
      pids = torch.randint(len(dataset.docs), (count,), dtype=torch.int32)
      t0 = time()
      searcher.ranker.score_pids(searcher.config, enc_Q, pids)
      t1 = time()
      print(f'{threads=} {count=} {i=} time={(t1-t0)*1000}')
      times.append((t1-t0)*1000)
    with open('scorer.jsonl', 'at') as fout:
      json.dump({'count': count, 'threads': threads, 'times': times}, fout)
      fout.write('\n')

