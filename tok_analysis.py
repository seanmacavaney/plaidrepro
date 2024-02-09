import pandas as pd
import torch
import json
from collections import Counter, defaultdict
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('downloads/colbertv2.0')

chunk_idx = -1
codes = []

code2tok = defaultdict(Counter)
tok2code = defaultdict(Counter)

with open('ids.jsonl') as fin:
  id_it = iter(fin)
  next(id_it)
  for ids in id_it:
    ids = json.loads(ids)
    if len(codes) == 0:
      chunk_idx += 1
      print(f'chunk {chunk_idx}')
      codes = torch.load(f'experiments/notebook/indexes/2bits/{chunk_idx}.codes.pt').cpu().tolist()
    for id, code in zip(ids, codes):
      code2tok[code][id] += 1
      tok2code[id][code] += 1
    codes = codes[len(ids):]
code2count = Counter({k: len(v) for k, v in code2tok.items()})
code2total = Counter({k: sum(v.values()) for k, v in code2tok.items()})
code2toppct = Counter({k: v.most_common(1)[0][1] / code2total[k] for k, v in code2tok.items()})
tok2count = Counter({k: len(v) for k, v in tok2code.items()})
tok2total = Counter({k: sum(v.values()) for k, v in tok2code.items()})
tok2toppct = Counter({k: v.most_common(1)[0][1] / tok2total[k] for k, v in tok2code.items()})
print('code2count', pd.Series(code2count.values()).describe())
print('code2total', pd.Series(code2total.values()).describe())
print('code2tooppct', pd.Series(code2toppct.values()).describe())
print('tok2count', pd.Series(tok2count.values()).describe())
print('tok2total', pd.Series(tok2total.values()).describe())
print('tok2tooppct', pd.Series(tok2toppct.values()).describe())
import pdb; pdb.set_trace()
code2tok, tok2code
