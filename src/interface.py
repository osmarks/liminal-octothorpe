from dataclasses import dataclass, field
import math
from bs4 import BeautifulSoup
from functools import lru_cache
import subprocess
import json
from datetime import datetime
import httpx
import asyncio
import inspect
from html.parser import HTMLParser
import re
from async_lru import alru_cache
import traceback

from util import VectorIndex, embed, build_vector_index

@dataclass
class ClassExclusionSentinel:
    pass

class HTMLStripper(HTMLParser):
    def __init__(self, exclusions=set()):
        super().__init__()
        self.reset()
        self.strict = False
        self.convert_charrefs = True
        self.text = ""
        self.ignoring = []
        self.in_header = None
        self.title = ""
        self.title_tag = ""
        self.in_title_tag = False
        self.in_table_cell = False
        self.exclusions = exclusions
    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        if tag in {"style", "script", "nav", "iframe", "svg", "math", "table"}:
            self.ignoring.append(tag)
            return
        if self.ignoring != []:
            self.ignoring.append(tag)
            return
        if tag in {"h1", "h2", "h3", "h4", "h5", "h6"} and self.in_header is None:
            self.in_header = tag
        if tag == "title":
            self.in_title_tag = True
        attrs_dict = {}
        for k, v in attrs:
            if v:
                attrs_dict[k] = v
        if tag == "img" and "mwe-math-fallback-image" in attrs_dict.get("class", ""):
            self.text += attrs_dict.get("alt", "")
        if any(s in self.exclusions for s in attrs_dict.get("class", "").split()):
            self.ignoring.append(ClassExclusionSentinel())
            return
        if tag == "td" or tag == "th":
            self.in_table_cell = True
    def handle_endtag(self, tag):
        tag = tag.lower()
        if tag == self.in_header:
            self.in_header = False
        sep = " " if self.in_table_cell else "\n"
        if self.ignoring == [] and tag not in {"span", "sub", "sup", "small", "i", "b", "em", "strong", "strike", "d", "a", "link", "head", "cite", "img", "bdi", "td", "th"}:
            self.text = self.text.strip(" \t")
            if not self.text.endswith(sep * 2):
                self.text += sep
        if self.ignoring == [] and tag in {"td", "th"} and not self.text.endswith("; ") and not self.text.endswith(".") and not self.text.endswith(":"):
            self.text = self.text.strip()
            self.text += "; "
        if self.ignoring != []:
            self.ignoring.pop()
        if tag == "title":
            self.in_title_tag = False
        if tag == "td" or tag == "th":
            self.in_table_cell = False
            if not self.text.endswith(" "): self.text += " "
    def handle_data(self, d):
        d = d.strip("\n")
        if self.ignoring == []:
            self.text += d
            if self.in_header:
                self.title += d
            if self.in_title_tag:
                self.title_tag += d

def strip_tags(html):
    s = HTMLStripper(exclusions={
        "locmap",
        "reference"
    })
    s.feed(str(html))
    return s.text.strip()

parser = "./node_modules/.bin/mercury-parser"

async def parse_url(url):
    proc = await asyncio.subprocess.create_subprocess_exec(parser, url, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise Exception(stderr)
    data = json.loads(stdout)
    if "error" in data:
        print(data["message"])
        return ""
    return strip_tags(data["content"])

def calculate(what):
    #return str(eval(what.replace(",", ""), { "pow": pow, **{ x: getattr(math, x) for x in dir(math) }}))
    proc = subprocess.run(["units", what.replace(",", "")], capture_output=True)
    if proc.returncode != 0: return ["Invalid syntax."]
    return [proc.stdout.decode("utf-8").strip().removeprefix("Definition: ")]

def get_page_obs(page):
    # find all paragraphs
    paragraphs = page.split("\n")
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    # find all sentence
    sentences = []
    for p in paragraphs:
      sentences += p.split('. ')
    sentences = [s.strip() + '.' for s in sentences if s.strip()]
    return ' '.join(sentences[:5])

def construct_lookup_list(page, keyword):
    paragraphs = page.split("\n")
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    # find all sentence
    sentences = []
    for p in paragraphs:
      sentences += p.split('. ')
    sentences = [s.strip() + '.' for s in sentences if s.strip()]

    parts = sentences
    parts = [p for p in parts if keyword.lower() in p.lower()]
    return parts

# TODO: cache should not really be parameterized by client
@alru_cache
async def fetch_wp_page(client: httpx.AsyncClient, q):
    q = q.replace(" ", "+")
    search_url = f"https://en.wikipedia.org/w/index.php?search={q}"
    res = await client.get(search_url, follow_redirects=True)
    response_text = res.text
    soup = BeautifulSoup(response_text, features="lxml")
    result_divs = soup.find_all("div", {"class": "mw-search-result-heading"})
    if result_divs:  # mismatch
        result_titles = [div.get_text().strip() for div in result_divs]
        return False, f"Could not find {q.replace('+', ' ')}. Similar: {result_titles[:5]}."
    else:
        page = [p.get_text().strip() for p in soup.find_all("p") + soup.find_all("ul")]
        if any("may refer to:" in p for p in page):
            return parse_wp_page("[" + q + "]")
        else:
            return True, strip_tags(soup.find("div", {"id": "mw-content-text"})).replace("[edit]", "")

async def parse_infobox(client, page, q):
    soup = fetch_wp_page(client, page)
    infobox = soup.find("table", {"class": "infobox"})
    chunk = ""
    def do_chunk(q, c):
        text = c.replace(" ;", ";").replace(";;", ";").strip().removesuffix(";").removeprefix(";").replace("\xa0", " ")
        if q.lower() in text.lower(): return text
    for row in infobox.find_all("tr"):
        print(row.get("class"))
        header = row.get("class") == ["mergedtoprow"]
        if header:
            if x := do_chunk(q, chunk): return x
            chunk = ""
        nxt = " ".join(c.text.replace(" • ", "").replace("\n", " ") for c in row.children)
        chunk += nxt
        if nxt: chunk += ": " if header else "; "
    if x := do_chunk(q, chunk): return x

async def web_search(client, q, n):
    res = await client.post("https://html.duckduckgo.com/html/", data={"q": q})
    result = res.text
    soup = BeautifulSoup(result, features="lxml")
    xs = []
    for result in soup.find(class_="serp__results").find_all(class_="result")[:n]:
        url = result.find(class_="result__a").get("href")
        if "--ad--" not in result.get("class") and result.find(class_="result__snippet"):
            new_line = result.find(class_="result__snippet").text.strip()
            xs.append((url, new_line))
    return xs

async def google_news(client: httpx.AsyncClient, q, n):
    res = await client.get("https://news.google.com/rss/search", params={
        "hl": "en-US",
        "gl": "US",
        "ceid": "US:en",
        "oc": "11",
        "q": q
    })
    soup = BeautifulSoup(res.text, features="xml")
    results = []
    async def get_real_link(item):
        url = item.find("link").text
        while "news.google.com" in url:
            req = client.build_request("GET", url)
            del req.headers["cookie"]
            req.headers["user-agent"] = "curl/8.5.0"
            res = (await client.send(req))
            if "location" in res.headers:
                url = res.headers["location"]
            else:
                break
            url = url.removeprefix("https://consent.google.com/m?continue=")
        results.append((url, item.find("title").text))
    await asyncio.gather(*(get_real_link(item) for item in soup.find_all("item")[:n]))
    return results

async def parallel_load_and_parse(urls):
    out = {}
    async def run(i, url):
        try:
            out[i] = await parse_url(url)
        except:
            traceback.print_exc()
    async with asyncio.TaskGroup() as tg:
        for i, url in enumerate(urls):
            tg.create_task(run(i, url))
    xs = []
    for i in range(len(urls)):
        if i in out: xs.append(out[i])
    return xs

@dataclass
class Finished:
    result: str

action_descriptions = {
    "Calculate": ("evaluates an arithmetic expression using `units`", "expression"),
    "Clock": ("returns the current date and time", ""),
    "WebSearch": ("runs a web search and opens top results", "query"),
    "NewsSearch": ("search Google News for recent material and opens results", "keyword"),
    "Wikipedia": ("opens a Wikipedia page for reading", "page"),
    "Finish": ("submits estimated probability and confidence", "probability%, confidence%"),
    "Find": ("looks up content in open page(s)", "query")
}

@dataclass
class Interface:
    http_session: httpx.AsyncClient
    open_page: str | None = None
    open_page_index: VectorIndex | None = None

    def actions(self):
        actions = {
            "Calculate": calculate,
            "Clock": lambda *_: [f'Current time is {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'],
            "Wikipedia": self.wikipedia,
            "WebSearch": self.web_search,
            "Finish": Finished,
            "NewsSearch": self.news_search
        }
        if self.open_page:
            actions["Find"] = self.find_in_page
        return actions

    async def find_in_page(self, query):
        query = "query: " + query 
        query_emb = (await embed(self.http_session, [query]))[query]
        top_chunks = self.open_page_index.query(query_emb, 5)
        return top_chunks

    async def wikipedia(self, page):
        ok, content = await fetch_wp_page(self.http_session, page)
        if not ok:
            return [content]
        else:
            index = await build_vector_index(self.http_session, content)
            self = Interface(http_session=self.http_session, open_page=content, open_page_index=index)
            return [index.chunks[0]], self

    async def web_search(self, query):
        res = await web_search(self.http_session, query, 5)
        data = "\n".join(await parallel_load_and_parse([ a for a, b in res ]))
        index = await build_vector_index(self.http_session, data)
        self = Interface(http_session=self.http_session, open_page=data, open_page_index=index)
        return [b for a, b in res], self

    async def news_search(self, query):
        res = await google_news(self.http_session, query, 5)
        data = "\n".join(await parallel_load_and_parse([ a for a, b in res ]))
        index = await build_vector_index(self.http_session, data)
        self = Interface(http_session=self.http_session, open_page=data, open_page_index=index)
        return [b for a, b in res], self

    async def execute(self, action_name, argument):
        actions = self.actions()
        for action, fn in actions.items():
            if action.lower() == action_name.lower():
                result = fn(argument)
                if inspect.isawaitable(result): result = await result
                if not isinstance(result, tuple):
                    return result, self
                else:
                    return result
        return None

    def action_descriptions(self):
        s = ""
        for i, (name, fn) in enumerate(self.actions().items()):
            description, argname = action_descriptions[name]
            s += f"{i + 1}. {name}[{argname}]: {description}"
            s += "\n"
        return s

async def test():
    client = httpx.AsyncClient()
    iface = Interface(client)
    res, iface = await iface.execute("NewsSearch", "generative AI")
    for r in res:
        print(r, end="\n\n")
    print(res)

if __name__ == "__main__":
    asyncio.run(test())