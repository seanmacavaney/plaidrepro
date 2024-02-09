import pandas as pd
import os
import gzip
import torch
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert.data import Queries, Collection
from colbert import Indexer, Searcher
from time import time
import ir_datasets
import json
import pyterrier as pt ; pt.init() ; from pyterrier_pisa import PisaIndex
from ir_measures import SetR

topics = pt.get_dataset('irds:msmarco-passage/dev/small').get_topics()

checkpoint = 'downloads/colbertv2.0'
nbits = 2
index_name = f'{nbits}bits'
with Run().context(RunConfig(experiment='notebook')):
    searcher = Searcher(index=index_name)


for num_threads in [16,16,8,4,2,1]:
  torch.set_num_threads(num_threads)
  times = []
  for query in pt.tqdm(topics['query']):
    t_start = time()
    enc_Q = searcher.encode([query])
    t_end = time()
    times.append(t_end - t_start)
  with open('colbert_query_encoder.jsonl', 'at') as fout:
    json.dump({
      'num_threads': num_threads,
      'latency': times,
    }, fout)
    fout.write('\n')
  #print(num_threads, pd.Series(times).describe())
