import sys
import torch
import time
import threading
from aiohttp import web
import aiohttp
import asyncio
import traceback
import umsgpack
import gc
import collections
import queue
import io
from sentence_transformers import SentenceTransformer
from prometheus_client import Counter, Histogram, REGISTRY, generate_latest

device = torch.device("cpu")
model_name = "BAAI/bge-small-en-v1.5"
model = SentenceTransformer(model_name).to(device)
model.eval()
print("model loaded")

MODELNAME = "bge-small-en-v1.5"
BS = 64

InferenceParameters = collections.namedtuple("InferenceParameters", ["text", "callback"])

items_ctr = Counter("modelserver_total_items", "Items run through model server", ["model", "modality"])
inference_time_hist = Histogram("modelserver_inftime", "Time running inference", ["model", "batch_size"])
batch_count_ctr = Counter("modelserver_batchcount", "Inference batches run", ["model"])

torch.set_grad_enabled(False)
def do_inference(params: InferenceParameters):
    with torch.no_grad():
        try:
            text, callback = params
            batch_size = text["input_ids"].shape[0]
            assert batch_size <= BS, f"max batch size is {BS}"
            items_ctr.labels(MODELNAME, "text").inc(batch_size)
            with inference_time_hist.labels(MODELNAME, batch_size).time():
                features = model(text)["sentence_embedding"]
                features /= features.norm(dim=-1, keepdim=True)
            batch_count_ctr.labels(MODELNAME).inc()
            callback(True, features.cpu().numpy())
        except Exception as e:
            traceback.print_exc()
            callback(False, str(e))
        finally:
            torch.cuda.empty_cache()

q = queue.Queue(10)
def infer_thread():
    while True:
        do_inference(q.get())

app = web.Application(client_max_size=2**26)
routes = web.RouteTableDef()

@routes.post("/")
async def run_inference(request):
    loop = asyncio.get_event_loop()
    data = umsgpack.loads(await request.read())
    event = asyncio.Event()
    results = None
    def callback(*argv):
        nonlocal results
        results = argv
        loop.call_soon_threadsafe(lambda: event.set())
    tokenized = model.tokenize(data["text"])
    tokenized = { k: v.to(device) for k, v in tokenized.items() }
    q.put_nowait(InferenceParameters(tokenized, callback))
    await event.wait()
    body_data = results[1]
    print(".", end="")
    sys.stdout.flush()
    if results[0]:
        status = 200
        body_data = [x.astype("float16").tobytes() for x in body_data]
    else:
        status = 500
        print(results[1])
    return web.Response(body=umsgpack.dumps(body_data), status=status, content_type="application/msgpack")

@routes.get("/config")
async def config(request):
    return web.Response(body=umsgpack.dumps({
        "model": model_name,
        "batch": BS,
        "embedding_size": model.get_sentence_embedding_dimension()
    }), status=200, content_type="application/msgpack")

@routes.get("/")
async def health(request):
    return web.Response(status=204)

@routes.get("/metrics")
async def metrics(request):
    return web.Response(body=generate_latest(REGISTRY))

app.router.add_routes(routes)

async def run_webserver():
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "", 1706)
    print("server starting")
    await site.start()

try:
    th = threading.Thread(target=infer_thread)
    th.start()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(run_webserver())
    loop.run_forever()
except KeyboardInterrupt:
    print("quitting")
    import sys
    sys.exit(0)
