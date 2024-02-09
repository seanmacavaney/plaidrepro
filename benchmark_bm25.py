import json
import pyterrier as pt ; pt.init() ; from pyterrier_pisa import PisaIndex
from ir_measures import SetR
idx = PisaIndex.from_dataset('msmarco_passage', threads=1)

topics = pt.get_dataset('irds:msmarco-passage/dev/small').get_topics()
qrels = pt.get_dataset('irds:msmarco-passage/dev/small').get_qrels().rename(columns={'qid': 'query_id', 'docno': 'doc_id', 'label': 'relevance'})

for num_results in [1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768]:
  bm25 = idx.bm25(num_results=num_results, include_latency=True)
  res = bm25(topics)
  latency = (res[res['rank']==0].latency)
  recall = [m.value for m in SetR.iter_calc(qrels, res.rename(columns={'qid': 'query_id', 'docno': 'doc_id'}))]
  with open('bm25.jsonl', 'at') as fout:
    json.dump({
      'num_results': num_results,
      'latency': latency.tolist(),
      'recall': recall,
    }, fout)
    fout.write('\n')
  print(num_results, latency.describe())
