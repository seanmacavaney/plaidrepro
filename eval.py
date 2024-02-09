import random
import pandas as pd
import json
from collections import defaultdict
import gzip
import os
import ir_measures
from ir_measures import *
import ir_datasets
import argparse
from tqdm import tqdm

def rbo(a, p=0.99):
  a = a.sort_values(by=['query_id', 'score'], ascending=False)
  a = dict(iter(a.groupby('query_id')))
  def inner(qrels, b):
    # adapted from https://github.com/terrierteam/ir_measures/blob/main/ir_measures/providers/compat_provider.py
    res = {}
    b = b.sort_values(by=['query_id', 'score'], ascending=False)
    b = dict(iter(b.groupby('query_id')))
    for qid in set(a.keys()) | set(b.keys()):
      ranking = list(a[qid].doc_id) if qid in a else []
      ideal = list(b[qid].doc_id) if qid in b else []
      ranking_set = set()
      ideal_set = set()
      score = 0.0
      normalizer = 0.0
      weight = 1.0
      for i in range(1000):
        if i < len(ranking):
            ranking_set.add(ranking[i])
        if i < len(ideal):
            ideal_set.add(ideal[i])
        score += weight*len(ideal_set.intersection(ranking_set))/(i + 1)
        normalizer += weight
        weight *= p
      res[qid] = score/normalizer
    return res.items()
  return inner

parser = argparse.ArgumentParser()
parser.add_argument('dataset')
parser.add_argument('files', nargs='+')
args = parser.parse_args()

measures = {
  'msmarco-passage/trec-dl-2019': [nDCG@10, nDCG@100, nDCG@1000, R(rel=2)@1000, RR(rel=2), RR(rel=3)],
  'msmarco-passage/trec-dl-2020': [nDCG@10, nDCG@100, nDCG@1000, R(rel=2)@1000, RR(rel=2), RR(rel=3)],
  'msmarco-passage/dev/small': [RR@10, R@1000],
}[args.dataset]

exh = pd.DataFrame(list(ir_measures.read_trec_run('results/' + {
  'msmarco-passage/trec-dl-2019': 'dl19',
  'msmarco-passage/trec-dl-2020': 'dl20',
  'msmarco-passage/dev/small': 'devs',
}[args.dataset] + '.exhuastive.run.gz')))

measures += [
  ir_measures.define(rbo(exh, p=0.999), name='RBO(p=0.999)'),
  ir_measures.define(rbo(exh, p=0.99), name='RBO(p=0.99)'),
  ir_measures.define(rbo(exh, p=0.9), name='RBO(p=0.9)'),
]

eval = ir_measures.evaluator(measures, ir_datasets.load(args.dataset).qrels)

random.shuffle(args.files)

for file in tqdm(args.files):
  outf = f'{file}.measures.json.gz'
  if not os.path.exists(outf):
    res = list(eval.iter_calc(ir_measures.read_trec_run(file)))
    out = {
      'avg': {},
      'perq': {},
    }
    lookup = defaultdict(list)
    for r in res:
      lookup[r.measure].append(r)
    for measure in lookup:
      out['avg'][str(measure)] = sum(r.value for r in lookup[measure]) / len(lookup[measure])
      out['perq'][str(measure)] = {r.query_id: r.value for r in lookup[measure]}
    with gzip.open(outf, 'wt') as fout:
      json.dump(out, fout)
