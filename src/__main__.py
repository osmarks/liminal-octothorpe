import json
import httpx
import asyncio
from dataclasses import dataclass, field
from interface import Interface, Finished
from datetime import datetime, timezone, timedelta
import re
import json
import util
import random
import numpy
import math

with open("config.json") as f:
    config = json.load(f)

async def send_request(client: httpx.AsyncClient, messages, stop):
    response = await client.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "HTTP-Referer": "https://osmarks.net/",
            "X-Title": "osmarks.net R&D",
            "Authorization": f"Bearer {config['apikey']}"
        },
        json={
            "model": "mistralai/mixtral-8x7b-instruct",
            "messages": messages,
            "temperature": 0.5,
            "max_tokens": 100,
            "stop": stop
        },
        timeout=30
    )
    assert "choices" in response.json(), response.json()["error"]["message"]
    return response.json()["choices"][0]["message"]["content"]

async def batch(client, n, *args):
    return await asyncio.gather(*[ send_request(client, *args) for _ in range(n) ])

TASK_PROMPT = """You are an advanced AI designed to forecast future events' probability using the internet.
Your training data runs to early 2023. It is now {date}. Search for updated knowledge where necessary.
You run in a loop of Thought, Action, Observation and, when ready, produce an answer consisting of your estimated probability and confidence in that probability.
You consider your actions in a Thought step and then produce an Action, the result of which is returned as an Observation.
Think creatively, be a good Bayesian reasoner and avoid cognitive biases.

Actions may be:
{actions}

Your current question is:
{question}
"""
MANIFOLD = "https://api.manifold.markets/v0/"

def format_question(q):
    return "\n".join("Question: " + l.strip() for l in q.splitlines() if l.strip())

@dataclass
class Context:
    client: httpx.AsyncClient
    question: str
    answers: list[(float, float)]

@dataclass
class RTree:
    interface: Interface
    thought: str = None
    action: str = None
    observation: str = None
    parent: "RTree" = None
    children: list["RTree"] = field(default_factory=list)
    terminal: bool = False

    def context(self):
        out = self.parent.context() if self.parent else []
        if self.thought:
            out.append("Thought: " + self.thought)
        if self.action:
            out.append("Action: " + self.action)
        if self.observation:
            for observation in self.observation:
                out.append("Observation: " + observation)
        return out

async def embed_node(ctx: Context, node: RTree):
    embeddings = (await util.embed(ctx.client, util.gen_chunks("\n".join(node.context()), sent_tokenize=False))).values()
    embedding = sum(embeddings) # something something linearity
    embedding /= numpy.linalg.norm(embedding)
    return embedding

async def expand(ctx: Context, node: RTree):
    results = await batch(ctx.client, 4, [
        {
            "role": "user",
            "content": TASK_PROMPT.format(
                date=datetime.now().strftime("%Y-%m-%d"),
                actions=node.interface.action_descriptions(),
                question=format_question(ctx.question)
            )
        },
        {
            "role": "assistant",
            "content": "\n".join(node.context()) + "\nThought:"
        }
    ], "Observation:")
    unique_answers = set()
    for result in results:
        action = re.search(r"\nAction: ([A-Za-z0-9_-]+)\[([^\n]+)\]\n", result)
        if action:
            thought = result[:action.start()].strip()
            action = action.groups()
        else:
            continue
        unique_answers.add((thought, action))
    async with asyncio.TaskGroup() as tg:
        async def run(thought, action):
            action_name, value = action
            try:
                value = str(json.loads(value))
            except json.JSONDecodeError:
                pass
            #print(thought, action_name, value, sep="\n")
            results = await node.interface.execute(action_name, value)
            if results:
                results, interface = results
                if isinstance(results, Finished):
                    child = RTree(interface, thought, f"{action_name}[{value}]", [], node, terminal=True)
                    mat = re.match(r"""(?:probability\s*=\s*)?"?([\d]+)%"?,\s*(?:confidence\s*=\s*)?"?([\d]+)%"?""", results.result)
                    if mat:
                        prob, confidence = mat.groups()
                        ctx.answers.append((float(prob) / 100, float(confidence) / 100))
                else:
                    child = RTree(interface, thought, f"{action_name}[{value}]", results, node)
                node.children.append(child)
        for thought, action in unique_answers:
            tg.create_task(run(thought, action))

async def select_node_to_expand(ctx: Context, tree: RTree):
    nodes = []
    async def traverse(tree, depth=0):
        nodes.append({ "node": tree, "embedding": await embed_node(ctx, tree), "depth": depth })
        await asyncio.gather(*(traverse(child, depth + 1) for child in (tree.children or [])))
    def weight(nodeinfo):
        similarities = [ float(n["embedding"] @ nodeinfo["embedding"]) for n in nodes if id(n) != id(nodeinfo) ]
        diversity = sum(1 - s for s in similarities) / (len(similarities) or 1)
        if nodeinfo["node"].terminal: return 0
        return max(2 + math.sqrt(nodeinfo["depth"] / 6) + 20 * diversity, 0.1)
    await traverse(tree)
    return random.choices(nodes, [ weight(node) for node in nodes ])[0]["node"]

async def find_market(client: httpx.AsyncClient):
    res = await client.get(f"{MANIFOLD}markets")
    data = res.json()
    market = random.choice([ mkt for mkt in data if mkt["isResolved"] == False and mkt["outcomeType"] == "BINARY" and mkt["mechanism"] == "cpmm-1" ])
    res = await client.get(f"{MANIFOLD}market/{market['id']}")
    return res.json()

async def bet(client: httpx.AsyncClient, amount, probability, market, yes):
    await client.post(f"{MANIFOLD}bet", json={
        "amount": amount,
        "contractId": market,
        "outcome": "YES" if yes else "NO",
        "limitProb": probability,
        "expiresAt": round((datetime.now(tz=timezone.utc) + timedelta(days=1)).timestamp() * 1000)
    }, headers={
        "Authorization": "Key " + config["manifoldkey"]
    })

def clamp(x, min_, max_):
    return max(min(x, max_), min_)

def odds(probability):
    return probability / (1 - probability)

def log_odds(probability):
    return math.log(odds(probability))

async def run():
    async with httpx.AsyncClient(timeout=10) as client:
        while True:
            market = await find_market(client)
            print(market["url"])
            interface = Interface(client)
            t = RTree(interface)
            ctx = Context(client, question=f"""{market['question']}
{market['description']}""", answers=[])
            for _ in range(128):
                node = await select_node_to_expand(ctx, t)
                if len(ctx.answers) > 10 or not node:
                    break
                await expand(ctx, node)
                print(ctx.answers)
            if ctx.answers:
                average_logodds = sum(c * log_odds(clamp(p, 0.01, 0.99)) for p, c in ctx.answers) / sum(c for _, c in ctx.answers)
                probability = (math.tanh(average_logodds / 2) + 1) / 2
                print(probability)
                probability = round(probability * 100) / 100
                size = min(50, market["totalLiquidity"] / 4)
                print("betting", size)
                await bet(client, size, probability, market["id"], probability > market["probability"])
asyncio.run(run())