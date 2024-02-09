import numpy as np
import pandas as pd
import os
import torch

from tqdm import tqdm
from typing import Union

from colbert.data import Collection, Queries, Ranking

from colbert.modeling.checkpoint import Checkpoint
from colbert.search.index_storage import IndexScorer

from colbert.infra.provenance import Provenance
from colbert.infra.run import Run
from colbert.infra.config import ColBERTConfig, RunConfig
from colbert.infra.launcher import print_memory_stats

import time

TextQueries = Union[str, 'list[str]', 'dict[int, str]', Queries]


class Searcher:
    def __init__(self, index, checkpoint=None, collection=None, config=None):
        print_memory_stats()

        initial_config = ColBERTConfig.from_existing(config, Run().config)

        default_index_root = initial_config.index_root_
        self.index = os.path.join(default_index_root, index)
        self.index_config = ColBERTConfig.load_from_index(self.index)

        self.checkpoint = checkpoint or self.index_config.checkpoint
        self.checkpoint_config = ColBERTConfig.load_from_checkpoint(self.checkpoint)
        self.config = ColBERTConfig.from_existing(self.checkpoint_config, self.index_config, initial_config)

        self.collection = Collection.cast(collection or self.config.collection)
        self.configure(checkpoint=self.checkpoint, collection=self.collection)

        self.checkpoint = Checkpoint(self.checkpoint, colbert_config=self.config)
        use_gpu = self.config.total_visible_gpus > 0
        if use_gpu:
            self.checkpoint = self.checkpoint.cuda()
        self.ranker = IndexScorer(self.index, use_gpu)

        print_memory_stats()

    def configure(self, **kw_args):
        self.config.configure(**kw_args)

    def encode(self, text: TextQueries, full_length_search=False):
        queries = text if type(text) is list else [text]
        bsize = 128 if len(queries) > 128 else None

        self.checkpoint.query_tokenizer.query_maxlen = self.config.query_maxlen
        Q = self.checkpoint.queryFromText(queries, bsize=bsize, to_cpu=True, full_length_search=full_length_search)

        return Q

    def search(self, text: str, k=10, filter_fn=None, full_length_search=False):
        Q = self.encode(text, full_length_search=full_length_search)
        return self.dense_search(Q, k, filter_fn=filter_fn)

    def search_all(self, queries: TextQueries, k=10, filter_fn=None, full_length_search=False):
        queries = Queries.cast(queries)
        queries_ = list(queries.values())

        Q = self.encode(queries_, full_length_search=full_length_search)

        return self._search_all_Q(queries, Q, k, filter_fn=filter_fn)

    def search_all_Q(self, queries: TextQueries, Q, k=10, filter_fn=None, full_length_search=False):
        queries = Queries.cast(queries)
        queries_ = list(queries.values())

        return self._search_all_Q(queries, Q, k, filter_fn=filter_fn)

    def search_exhaustive_Q(self, queries: TextQueries, Q, k=10, filter_fn=None, full_length_search=False):
        queries = Queries.cast(queries)
        queries_ = list(queries.values())
        all_scored_pids = [list(zip(*self.ranker.exhaustive(self.config, Q[query_idx:query_idx+1], k)))
                           for query_idx in tqdm(range(Q.size(0)))]

        data = {qid: val for qid, val in zip(queries.keys(), all_scored_pids)}

        provenance = Provenance()
        provenance.source = 'Searcher::exhaustive'
        provenance.queries = queries.provenance()
        provenance.config = self.config.export()
        provenance.k = k

        return Ranking(data=data, provenance=provenance)

    def _search_all_Q(self, queries, Q, k, filter_fn=None):
        all_scored_pids = [list(zip(*self.dense_search(Q[query_idx:query_idx+1], k, filter_fn=filter_fn)))
                           for query_idx in tqdm(range(Q.size(0)))]

        data = {qid: val for qid, val in zip(queries.keys(), all_scored_pids)}

        provenance = Provenance()
        provenance.source = 'Searcher::search_all'
        provenance.queries = queries.provenance()
        provenance.config = self.config.export()
        provenance.k = k

        return Ranking(data=data, provenance=provenance)

    def dense_search(self, Q: torch.Tensor, k=10, filter_fn=None):
        if k <= 10:
            if self.config.ncells is None:
                self.configure(ncells=1)
            if self.config.centroid_score_threshold is None:
                self.configure(centroid_score_threshold=0.5)
            if self.config.ndocs is None:
                self.configure(ndocs=256)
        elif k <= 100:
            if self.config.ncells is None:
                self.configure(ncells=2)
            if self.config.centroid_score_threshold is None:
                self.configure(centroid_score_threshold=0.45)
            if self.config.ndocs is None:
                self.configure(ndocs=1024)
        else:
            if self.config.ncells is None:
                self.configure(ncells=4)
            if self.config.centroid_score_threshold is None:
                self.configure(centroid_score_threshold=0.4)
            if self.config.ndocs is None:
                self.configure(ndocs=max(k * 4, 4096))

        pids, scores = self.ranker.rank(self.config, Q, filter_fn=filter_fn)

        return pids[:k], list(range(1, k+1)), scores[:k]


class LadrSearcher(Searcher):
    def __init__(self, index, checkpoint=None, collection=None, config=None, first_pass=None, corpus_graph=None, proactive_steps=0, adaptive_depth=0):
        super().__init__(index=index, checkpoint=checkpoint, collection=collection, config=config)
        self.first_pass = first_pass
        self.corpus_graph = corpus_graph
        self.proactive_steps = proactive_steps
        self.adaptive_depth = adaptive_depth

    def search_all(self, queries: TextQueries, k=10, filter_fn=None, full_length_search=False):
        queries = Queries.cast(queries)
        queries_ = list(queries.values())
        df_queries = pd.DataFrame(queries.items(), columns=['qid', 'query'])

        initial_res = dict(iter(self.first_pass(df_queries).groupby('qid')))

        Q = self.encode(queries_, full_length_search=full_length_search)

        return self._search_all_Q(queries, Q, k, filter_fn=filter_fn, initial_res=initial_res)

    def search_all_Q(self, queries, Q, k, filter_fn=None):
        queries = Queries.cast(queries)
        df_queries = pd.DataFrame(queries.items(), columns=['qid', 'query'])
        initial_res = dict(iter(self.first_pass(df_queries).groupby('qid')))
        return self._search_all_Q(queries, Q, k, filter_fn=filter_fn, initial_res=initial_res)

    def _search_all_Q(self, queries, Q, k, filter_fn=None, initial_res=None):
        qids = list(queries.keys())
        all_scored_pids = [list(zip(*self.dense_search(Q[query_idx:query_idx+1], k, filter_fn=filter_fn, initial_res=initial_res.get(qids[query_idx], pd.DataFrame(columns=['docno'])))))
                           for query_idx in tqdm(range(Q.size(0)))]

        data = {qid: val for qid, val in zip(queries.keys(), all_scored_pids)}

        provenance = Provenance()
        provenance.source = 'Searcher::search_all'
        provenance.queries = queries.provenance()
        provenance.config = self.config.export()
        provenance.k = k

        return Ranking(data=data, provenance=provenance)

    def dense_search(self, Q: torch.Tensor, k=10, filter_fn=None, initial_res=None):
        assert initial_res is not None
        if k <= 10:
            if self.config.ncells is None:
                self.configure(ncells=1)
            if self.config.centroid_score_threshold is None:
                self.configure(centroid_score_threshold=0.5)
            if self.config.ndocs is None:
                self.configure(ndocs=256)
        elif k <= 100:
            if self.config.ncells is None:
                self.configure(ncells=2)
            if self.config.centroid_score_threshold is None:
                self.configure(centroid_score_threshold=0.45)
            if self.config.ndocs is None:
                self.configure(ndocs=1024)
        else:
            if self.config.ncells is None:
                self.configure(ncells=4)
            if self.config.centroid_score_threshold is None:
                self.configure(centroid_score_threshold=0.4)
            if self.config.ndocs is None:
                self.configure(ndocs=max(k * 4, 4096))

        docids = initial_res.docno.astype(int).to_numpy()
        ext_docids = [docids]
        for _ in range(self.proactive_steps):
            docids = self.graph.edges_data[docids].reshape(-1)
            ext_docids.append(docids)
        docids = np.unique(np.concatenate(ext_docids).astype(np.int32))
        if self.adaptive_depth == 0:
            docids = torch.from_numpy(docids)
            scores, pids = self.ranker.score_pids(self.config, Q, docids)
            scores_sorter = scores.sort(descending=True)
            pids, scores = pids[scores_sorter.indices].tolist(), scores_sorter.values.tolist()
        else:
            scores = self.ranker.score_pids(self.config, Q, torch.from_numpy(docids))[0].numpy()
            while True:
                if scores.shape[0] > self.adaptive_depth:
                    dids = docids[np.argpartition(scores, -self.adaptive_depth)[-self.adaptive_depth:]]
                else:
                    dids = docids
                neighbour_dids = np.unique(self.graph.edges_data[dids].reshape(-1).astype(np.int32))
                new_neighbour_dids = np.setdiff1d(neighbour_dids, docids, assume_unique=True)
                if new_neighbour_dids.shape[0] == 0:
                    break
                neighbour_scores = self.ranker.score_pids(self.config, Q, torch.from_numpy(new_neighbour_dids))[0].numpy()
                cat_dids = np.concatenate([docids, new_neighbour_dids])
                idxs = np.argsort(cat_dids)
                docids = cat_dids[idxs]
                scores = np.concatenate([scores, neighbour_scores])[idxs]
            # score by score
            pids = docids
            if len(scores) > k:
                idxs = np.argpartition(scores, -k)[-k:]
                scores = scores[idxs]
                pids = pids[idxs]
            idxs = np.argsort(-scores)
            pids = pids[idxs]
            scores = scores[idxs]

        return pids[:k], list(range(1, k+1)), scores[:k]
