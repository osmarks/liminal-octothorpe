from async_lru import alru_cache
import nltk
import umsgpack
import numpy
from dataclasses import dataclass, field
import re

BACKEND = "http://localhost:1706/"
CHUNK_SIZE = 500
def gen_chunks(text, sent_tokenize=True):
    buf = ""
    j = 0
    for part in re.split(r"[\n\r]+", text):
        for sent in (nltk.sent_tokenize(part) if sent_tokenize else [part]):
            sent = sent.strip()
            if sent_tokenize and not sent[-1] in "!.?": sent += "."
            sent += " "
            if len(buf) + len(sent) > CHUNK_SIZE:
                yield buf.strip()
                j += 1
                buf = ""
            buf += sent
    if buf:
        yield buf.strip()

def split_list(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

@alru_cache
async def emb_config(client):
    res = await client.get(BACKEND + "config")
    return umsgpack.unpackb(res.content)

emb_cache = {}

async def use_emb_server(client, query):
    res = await client.post(BACKEND, data=umsgpack.dumps({ "text": query }), timeout=None)
    response = umsgpack.loads(res.content)
    if res.status_code == 200:
        response = [ numpy.frombuffer(x, dtype="float16") for x in response ]
        return response
    else:
        raise Exception(response if res.headers.get("content-type") == "application/msgpack" else res.text)

@dataclass
class VectorIndex:
    vectors: numpy.array
    chunks: list[str]

    def query(self, vec: numpy.array, n):
        n = min(n, len(self.chunks))
        scores = self.vectors @ vec
        ind = numpy.argpartition(scores, -n)[-n:]
        ind = ind[numpy.argsort(scores[ind])]
        return [ self.chunks[i] for i in ind ]

async def embed(client, chunks):
    config = await emb_config(client)
    chunks = { c: emb_cache.get(c) for c in chunks }
    for group in split_list([ c for c, v in chunks.items() if v is None ], config["batch"]):
        for embedding, chunk in zip(await use_emb_server(client, [ x for x in group ]), group):
            chunks[chunk] = embedding 
    return chunks

async def build_vector_index(client, s):
    chunks = await embed(client, list(gen_chunks(s)))
    return VectorIndex(numpy.array(list(chunks.values())), list(chunks.keys()))