import os
import sys
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert.data import Queries, Collection
from colbert import Indexer, Searcher

collection = 'downloads/collection.tsv'
collection = Collection(path=collection)
nbits = 2   # encode each dimension with 2 bits
doc_maxlen = 300   # truncate passages at 300 tokens
checkpoint = 'downloads/colbertv2.0'
index_name = f'{nbits}bits.redo'

with Run().context(RunConfig(nranks=1, experiment='notebook')):  # nranks specifies the number of GPUs to use.
    config = ColBERTConfig(doc_maxlen=doc_maxlen, nbits=nbits)

    indexer = Indexer(checkpoint=checkpoint, config=config)
    indexer.index(name=index_name, collection=collection, overwrite=True)
