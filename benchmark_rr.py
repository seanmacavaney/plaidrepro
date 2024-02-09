from collections import Counter
import os
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
from transformers import AutoTokenizer
#from argparse import ArgumentParser
#parser = ArgumentParser()
#parser.add_argument('--num_results', '-n', default=1000)
#parser.add_argument('--num_neighbours', '-k', default=128)
#parser.add_argument('--num_hops', '-h', default=1)
#args = parser.parse_args()
checkpoint = 'downloads/colbertv2.0'
nbits = 2
index_name = f'{nbits}bits'
index_n = 'lexical'

class BertTok(pt.Transformer):
  def __init__(self):
    self.tok = AutoTokenizer.from_pretrained('bert-base-uncased', use_fast=True)
  def transform(self, inp):
    out = inp
    if 'query' in inp.columns:
      out = out.assign(query_toks=out['query'].apply(self.tokenize))
    if 'text' in inp.columns:
      out = out.assign(toks=out['text'].apply(self.tokenize))
    return out

  def tokenize(self, value):
    return dict(Counter(self.tok.convert_ids_to_tokens(self.tok(value)['input_ids'], skip_special_tokens=True)))

if index_n == 'lexical':
  index = PisaIndex.from_dataset('msmarco_passage', threads=1)
  bm25 = index.bm25(num_results=1000)
  retr = bm25
elif index_n == 'bert':
  index = PisaIndex('/home/sean/data/indices/msmarco-passage.bert.pisa')
  bm25 = index.bm25(query_weighted=True)
  retr = BertTok() >> bm25

with Run().context(RunConfig(experiment='notebook')):
    searcher = LadrSearcher(index=index_name, first_pass=retr, corpus_graph=None, proactive_steps=0)

ds = 'dl19'
if ds == 'dl19':
  queries = {q.query_id: q.text for q in ir_datasets.load('msmarco-passage/trec-dl-2019/judged').queries}
elif ds == 'devs':
  queries = {q.query_id: q.text for q in ir_datasets.load('msmarco-passage/dev/small').queries}

print('encoding')
enc_Q = searcher.encode(list(queries.values()))
print('starting')

torch.set_num_threads(1)

for num_results in [10, 100, 200, 500, 1000, 2000, 5000, 10000]:
#for num_results in [1]:
#  for num_neighbours in [0]:
#    for depth in [0]:
      path = f'results/{ds}.rr.{index_n}.num_results-{num_results}.run.gz'
      if os.path.exists(path):
        continue
      bm25.num_results = num_results
      t0 = time()
      results = searcher.search_all_Q(queries, enc_Q, k=1000)
      t1 = time()
      res_dict = results.todict()
      with gzip.open(path, 'wt') as fout:
        for qid in res_dict.keys():
          for did, rank, score in res_dict[qid]:
            fout.write(f'{qid} 0 {did} {rank} {score} run\n')
      with open(f'results/{ds}.rr.{index_n}.num_results-{num_results}.time', 'wt') as fout:
        fout.write(f'{(t1-t0)/len(queries)*1000}')
      print(f'{num_results=} time={(t1-t0)/len(queries)*1000}')
