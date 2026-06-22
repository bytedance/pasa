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
"""
Please note that:
1. You need to first apply for a Google Search API key at https://serpapi.com/,
   and replace the 'your google keys' in utils.py before you can use it.
2. The service for searching arxiv and obtaining paper contents is relatively simple. 
   If there are any bugs or improvement suggestions, you can submit pull requests.
   We would greatly appreciate and look forward to your contributions!!
"""
import os
import json
import argparse
from models      import Agent
from paper_agent import PaperAgent
from datetime    import datetime, timedelta

parser = argparse.ArgumentParser()
parser.add_argument('--input_file',     type=str, default="data/RealScholarQuery/test.jsonl")
parser.add_argument('--crawler_path',   type=str, default="checkpoints/pasa-7b-crawler")
parser.add_argument('--selector_path',  type=str, default="checkpoints/pasa-7b-selector")
parser.add_argument('--output_folder',  type=str, default="results")
parser.add_argument('--expand_layers',  type=int, default=2)
parser.add_argument('--search_queries', type=int, default=5)
parser.add_argument('--search_papers',  type=int, default=10)
parser.add_argument('--expand_papers',  type=int, default=20)
parser.add_argument('--threads_num',    type=int, default=20)
args = parser.parse_args()

crawler = Agent(args.crawler_path)
selector = Agent(args.selector_path)

with open(args.input_file) as f:
    for idx, line in enumerate(f.readlines()):
        data = json.loads(line)
        end_date = data['source_meta']['published_time']
        end_date = datetime.strptime(end_date, "%Y%m%d") - timedelta(days=7)
        end_date = end_date.strftime("%Y%m%d")
        paper_agent = PaperAgent(
            user_query     = data['question'], 
            crawler        = crawler,
            selector       = selector,
            end_date       = end_date,
            expand_layers  = args.expand_layers,
            search_queries = args.search_queries,
            search_papers  = args.search_papers,
            expand_papers  = args.expand_papers,
            threads_num    = args.threads_num
        )
        if "answer" in data:
            paper_agent.root.extra["answer"] = data["answer"]
        
        paper_agent.run()
        
        if args.output_folder != "":
            json.dump(paper_agent.root.todic(), open(os.path.join(args.output_folder, f"{idx}.json"), "w"), indent=2)