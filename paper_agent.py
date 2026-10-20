# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import re
import json
import threading
import utils
from paper_node import PaperNode
from typing import TYPE_CHECKING
from paper_search import PaperSearch
if TYPE_CHECKING:
    from models import Agent
from datetime   import datetime
from utils      import (
    search_paper_by_title,
    google_search_arxiv_id,
    search_paper_by_arxiv_id,
    search_section_by_arxiv_id
)

class PaperAgent:
    def __init__(
        self,
        user_query:     str,
        crawler:        Agent, # prompt(s) -> response(s)
        selector:       Agent, # prompt(s) -> score(s)
        end_date:       str = datetime.now().strftime("%Y%m%d"),
        prompts_path:   str = "agent_prompt.json",
        expand_layers:  int = 2,
        search_queries: int = 5,
        search_papers:  int = 10, # per query
        expand_papers:  int = 20, # per layer
        threads_num:    int = 20, # number of threads in parallel at the same time
        search_provider: str = None, # auto, parallel, google or none
    ) -> None:
        self.user_query = user_query
        self.crawler    = crawler
        self.selector   = selector
        self.search_tool = PaperSearch(google_search_arxiv_id, utils.GOOGLE_KEY, search_provider)
        self.end_date   = end_date
        self.prompts    = json.load(open(prompts_path))
        self.root       = PaperNode({
            "title": user_query,
            "extra": {
                "touch_ids": [],
                "crawler_recall_papers": [],
                "recall_papers": [],
            }
        })

        # hyperparameters
        self.expand_layers   = expand_layers
        self.search_queries  = search_queries
        self.search_papers   = search_papers
        self.expand_papers   = expand_papers
        self.threads_num     = threads_num
        self.papers_queue    = []
        self.expand_start    = 0
        self.lock            = threading.Lock()
        self.templates       = {
            "cite_template":   r"~\\cite\{(.*?)\}",
            "search_template": r"Search\](.*?)\[",
            "expand_template": r"Expand\](.*?)\["
        }
    
    @staticmethod
    def do_parallel(func, args, num):
        threads = []
        for _ in range(num):
            thread = threading.Thread(target=func, args=args)
            thread.start()
            threads.append(thread)
        for thread in threads:
            thread.join()

    def search_paper(self, queries):
        while True:
            with self.lock:
                if not queries:
                    break
                query, self.root.child[query] = queries.pop(), []
            pre_arxiv_ids, searched_papers = self.search_tool.search(query, self.search_papers, self.end_date), []
            for arxiv_id in pre_arxiv_ids:
                arxiv_id = arxiv_id.split('v')[0]
                self.lock.acquire()
                if arxiv_id not in self.root.extra["touch_ids"]:
                    self.root.extra["touch_ids"].append(arxiv_id)
                    self.lock.release()
                    paper = search_paper_by_arxiv_id(arxiv_id)
                    if paper is not None:
                        searched_papers.append(paper)
                else:
                    self.lock.release()
            
            select_prompts  = [self.prompts["get_selected"].format(title=paper["title"], abstract=paper["abstract"], user_query=self.user_query) for paper in searched_papers]
            scores = self.selector.infer_score(select_prompts)
            with self.lock:
                for score, paper in zip(scores, searched_papers):
                    self.root.extra["crawler_recall_papers"].append(paper["title"])
                    if score > 0.5:
                        self.root.extra["recall_papers"].append(paper["title"])
                    paper_node = PaperNode({
                        "title":        paper["title"],
                        "arxiv_id":     paper["arxiv_id"],
                        "depth":        0,
                        "abstract" :    paper["abstract"],
                        "sections" :    paper["sections"],
                        "source":       "Search " + paper["source"],
                        "select_score": score,
                        "extra":        {}
                    })
                    self.root.child[query].append(paper_node)
                    self.papers_queue.append(paper_node)

    def search(self):
        prompt = self.prompts["generate_query"].format(user_query=self.user_query).strip()
        queries = self.crawler.infer(prompt)
        queries = [q.strip() for q in re.findall(self.templates["search_template"], queries, flags=re.DOTALL)][:self.search_queries]
        PaperAgent.do_parallel(self.search_paper, (queries,), len(queries))

    def get_paper_content(self, new_expand, crawl_prompts, have_full_paper):
        while new_expand:
            with self.lock:
                if new_expand:
                    paper = new_expand.pop(0)
                else:
                    break
            
            if paper.sections == "":
                paper.sections = search_section_by_arxiv_id(paper.arxiv_id, self.templates["cite_template"])
                if not paper.sections:
                    paper.extra["expand"] = "get full paper error"
                    continue
            
            paper.extra["expand"] = "not expand"
            prompt = self.prompts["select_section"].format(user_query=self.user_query, title=paper.title, abstract=paper.abstract, sections=paper.sections.keys()).strip()
            with self.lock:
                have_full_paper.append(paper)
                crawl_prompts.append(prompt)

    def search_ref(self, section_sources_ori, select_prompts, section_sources, lock):
        while section_sources_ori:
            with lock:
                if section_sources_ori:
                    section, title = section_sources_ori.pop(0)
                else:
                    break
            
            searched_paper = search_paper_by_title(title)
            if searched_paper is None:
                continue
            
            arxiv_id = searched_paper["arxiv_id"]
            with lock:
                if arxiv_id not in self.root.extra["touch_ids"]:
                    self.root.extra["touch_ids"].append(arxiv_id)
                else:
                    continue
            prompt = self.prompts["get_selected"].format(title=title, abstract=searched_paper["abstract"], user_query=self.user_query)
            with lock:
                select_prompts.append(prompt)
                section_sources.append([section, searched_paper])

    def do_expand(self, depth, have_full_paper, crawl_results):
        while have_full_paper:
            with self.lock:
                if have_full_paper:
                    paper = have_full_paper.pop(0)
                    crawl_result = crawl_results.pop(0)
                else:
                    break
            crawl_result = re.findall(self.templates["expand_template"], crawl_result, flags=re.DOTALL)
            section_sources_ori = []
            for section in crawl_result:
                section = section.strip()
                if section not in paper.sections:
                    continue
                for ref in paper.sections[section]:
                    section_sources_ori.append([section, ref])
            select_prompts, section_sources, lock = [], [], threading.Lock()
            PaperAgent.do_parallel(self.search_ref, (section_sources_ori, select_prompts, section_sources, lock), self.threads_num * 3)
            scores = self.selector.infer_score(select_prompts)
            for score, (section, ref_paper) in zip(scores, section_sources):
                self.root.extra["crawler_recall_papers"].append(ref_paper["title"])
                if score > 0.5:
                    self.root.extra["recall_papers"].append(ref_paper["title"])
                paper_node = PaperNode({
                    "title":        ref_paper["title"],
                    "depth":        depth + 1,
                    "arxiv_id":     ref_paper["arxiv_id"],
                    "abstract" :    ref_paper["abstract"],
                    "sections" :    ref_paper["sections"],
                    "source":       "Expand " + ref_paper["source"],
                    "select_score": score,
                    "extra":        {}
                })

                with self.lock:
                    if section not in paper.child:
                        paper.child[section] = []
                    paper.child[section].append(paper_node)
                    paper.extra["expand"] = "success"
                    self.papers_queue.append(paper_node)

    def expand(self, depth):
        expand_papers = sorted(self.papers_queue[self.expand_start:], key=PaperNode.sort_paper, reverse=True)
        self.papers_queue = self.papers_queue[:self.expand_start] + expand_papers
        if depth > 0:
            expand_papers = expand_papers[:self.expand_papers]
        self.expand_start = len(self.papers_queue)
        crawl_prompts, have_full_paper = [], []
        PaperAgent.do_parallel(self.get_paper_content, (expand_papers, crawl_prompts, have_full_paper), self.threads_num)
        crawl_results = self.crawler.batch_infer(crawl_prompts)
        PaperAgent.do_parallel(self.do_expand, (depth, have_full_paper, crawl_results), self.threads_num)

    def run(self):
        self.search()
        for depth in range(self.expand_layers):
            self.expand(depth)
