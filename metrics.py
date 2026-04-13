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
import argparse
import glob
import json
import os

from tqdm import tqdm

from utils import cal_micro, keep_letters

parser = argparse.ArgumentParser()
parser.add_argument('--output_folder',  type=str, default="results")
parser.add_argument('--output_folder_ensemble',  type=str, default=None)
args = parser.parse_args()

pred_files = glob.glob(args.output_folder + "/*.json")

crawler_recalls, precisions, recalls, recalls_100, recalls_50, recalls_20, actions, scores = [], [], [], [], [], [], [], []
for pred_file in tqdm(pred_files):
    paper_root = json.load(open(pred_file))
    crawled_papers, crawled_paper_set, selected_paper_set, queue, action, score = [], set(), set(), [paper_root], 0, []
    answer_paper_set = set([keep_letters(paper) for paper in paper_root["extra"]["answer"]])
    while len(queue) > 0:
        node, queue = queue[0], queue[1:]
        action += len(node["child"])
        total_score = 0
        for _, v in node["child"].items():
            total_score -= 0.1
            for i in v:
                queue.append(i)
                if i["select_score"] > 0.5:
                    selected_paper_set.add(keep_letters(i["title"]))
                    total_score += 1
                if keep_letters(i["title"]) not in crawled_paper_set:
                    crawled_paper_set.add(keep_letters(i["title"]))
                    crawled_papers.append([keep_letters(i["title"]), i["select_score"]])
        score.append(total_score)
    actions.append(action)
    scores.append(sum(score) / len(score) if len(score) > 0 else 0)
    
    # ensemble
    if args.output_folder_ensemble is not None:
        paper_root = json.load(open(os.path.join(args.output_folder_ensemble, pred_file.split("/")[-1])))
        queue = [paper_root]
        while len(queue) > 0:
            node, queue = queue[0], queue[1:]
            for _, v in node["child"].items():
                for i in v:
                    queue.append(i)
                    if i["select_score"] > 0.5:
                        selected_paper_set.add(keep_letters(i["title"]))
                    if keep_letters(i["title"]) not in crawled_paper_set:
                        crawled_paper_set.add(keep_letters(i["title"]))
                        crawled_papers.append([keep_letters(i["title"]), i["select_score"]])
    crawled_papers.sort(key=lambda x: x[1], reverse=True)
    crawled_20, crawled_50, crawled_100 = set(), set(), set()
    for i in range(100):
        if i >= len(crawled_papers):
            break
        if i < 20:
            crawled_20.add(crawled_papers[i][0])
        if i < 50:
            crawled_50.add(crawled_papers[i][0])
        crawled_100.add(crawled_papers[i][0])

    crawled_res = cal_micro(crawled_paper_set, answer_paper_set)
    selected_res = cal_micro(selected_paper_set, answer_paper_set)
    crawled_20_res = cal_micro(crawled_20, answer_paper_set)
    crawled_50_res = cal_micro(crawled_50, answer_paper_set)
    crawled_100_res = cal_micro(crawled_100, answer_paper_set)

    crawler_recalls.append(crawled_res[0] / (crawled_res[0] + crawled_res[2] if (crawled_res[0] + crawled_res[2]) > 0 else 1e-9))
    precisions.append(selected_res[0] / (selected_res[0] + selected_res[1] if (selected_res[0] + selected_res[1]) > 0 else 1e-9))
    recalls.append(selected_res[0] / (selected_res[0] + selected_res[2] if (selected_res[0] + selected_res[2]) > 0 else 1e-9))
    recalls_100.append(crawled_100_res[0] / (crawled_100_res[0] + crawled_100_res[2] if (crawled_100_res[0] + crawled_100_res[2]) > 0 else 1e-9))
    recalls_50.append(crawled_50_res[0] / (crawled_50_res[0] + crawled_50_res[2] if (crawled_50_res[0] + crawled_50_res[2]) > 0 else 1e-9))
    recalls_20.append(crawled_20_res[0] / (crawled_20_res[0] + crawled_20_res[2] if (crawled_20_res[0] + crawled_20_res[2]) > 0 else 1e-9))

print("{} & {} & {} & {} & {} & {}".format(
    round(sum(crawler_recalls) / len(crawler_recalls), 4),
    round(sum(precisions) / len(precisions), 4),
    round(sum(recalls) / len(recalls), 4),
    round(sum(recalls_100) / len(recalls_100), 4),
    round(sum(recalls_50) / len(recalls_50), 4),
    round(sum(recalls_20) / len(recalls_20), 4),
))
print("{} & {} & {} & {}  & {}".format(
    round(sum(crawler_recalls) / len(crawler_recalls), 4),
    round(sum(actions) / len(actions), 4),
    round(sum(scores) / len(scores), 4),
    round(sum(precisions) / len(precisions), 4),
    round(sum(recalls) / len(recalls), 4),
))