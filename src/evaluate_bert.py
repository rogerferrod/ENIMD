import argparse

import json
import faiss
import pickle
import os

import matplotlib.pyplot as plt
import torch
import random
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.manifold import TSNE

from tqdm import tqdm

at_k = sorted([1, 3, 5, 10, 100], reverse=True)


def run(args):
    corpus = []
    celexs = set()
    with open(args.corpus, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            if data['country'] == args.country:
                corpus.append(data)
                celexs = celexs.union(data['labels'])

    samples = []
    with open(args.queries, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            data = json.loads(line)
            if data['country'] == args.country:
                samples.append(data)

    with open(os.path.join(args.embeddings, "queries_embeddings.pickle"), 'rb') as f:
        queries_embeddings = pickle.load(f)

    with open(os.path.join(args.embeddings, "corpus_embeddings.pickle"), 'rb') as f:
        corpus_embeddings = pickle.load(f)

    indeces = []
    for i, sample in enumerate(samples):
        if sample['label'] in celexs:
            indeces.append(i)

    samples = [samples[i] for i in indeces]
    queries_embeddings = [queries_embeddings[i] for i in indeces]

    d = corpus_embeddings.shape[1]

    index = faiss.IndexFlatIP(d)

    if not args.cpu:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)

    index.add(corpus_embeddings.numpy())

    hits = {k: 0 for k in at_k}
    _map = {k: 0 for k in at_k}
    mrr = {k: 0 for k in at_k}

    for i, _ in tqdm(enumerate(samples), total=len(samples)):
        top_k = None
        for k in at_k:
            if top_k is None:
                _, top_k = index.search((torch.unsqueeze(queries_embeddings[i], 0)).numpy(), k)
                top_k = top_k[0]
            else:
                top_k = top_k[:k]

            set_hits = 0
            ap = 0
            first = True
            for c, j in enumerate(top_k):
                if samples[i]['label'] in corpus[j]['labels']:
                    hits[k] += 1
                    set_hits += 1
                    ap += set_hits / (c + 1)
                    if first:
                        mrr[k] += 1 / (c + 1)
                        first = False

            if set_hits > 0:
                _map[k] += ap / set_hits

    hits_k = dict([(x[0], x[1] / len(samples)) for x in hits.items()])
    map_k = dict([(x[0], x[1] / len(samples)) for x in _map.items()])
    mrr_k = dict([(x[0], x[1] / len(samples)) for x in mrr.items()])

    print("Hits: ")
    print(hits_k)
    print("\n\nMAP: ")
    print(map_k)
    print("\n\nMRR: ")
    print(mrr_k)

    zipped = list(zip(queries_embeddings, samples))

    random.seed(32)
    random.shuffle(zipped)
    examples = zipped[:25]
    ex_embeddings, ex_sample = list(zip(*examples))

    points = []  # [(pos, score, label)]

    for i, _ in tqdm(enumerate(ex_sample), total=len(ex_sample)):
        scores, indeces = index.search((torch.unsqueeze(ex_embeddings[i], 0)).numpy(), len(corpus_embeddings))
        top_k = indeces[0][:200]
        top_s = scores[0][:200]

        top_s = top_s - np.min(top_s)
        for c, j in enumerate(top_k):
            score = top_s[c]
            if ex_sample[i]['label'] in corpus[j]['labels']:
                points.append((c, score, 'pos'))
            else:
                points.append((c, score, 'neg'))

    df_points = pd.DataFrame(points, columns=['x', 'y', 'label'])
    df_points = df_points.sort_values(by=['label'], ascending=True)
    sns.scatterplot(data=df_points, x="x", y="y", hue="label",
                    hue_order=['pos', 'neg'], palette=['green', 'red'], legend='full',
                    size='label', sizes=(8, 2))

    plt.xlabel('Ranking position')
    plt.ylabel('Ranking score')
    plt.title('Retrieval scores for 25 random queries')
    plt.savefig(os.path.join(args.embeddings, 'ranking.png'))

    plt.figure()

    embs = []  # [(emb, label, type)]
    count = 0
    celexs = set([x['label'] for x in ex_sample])
    zipped = list(zip(corpus_embeddings, corpus))
    random.seed(32)
    random.shuffle(zipped)
    for emb, pub in zipped:
        if set(pub['labels']) & celexs:
            for label in pub['labels']:
                embs.append((emb, label, 'ART'))
        else:
            if count < 2000:
                embs.append((emb, 'NEG', 'ART'))
                count += 1

    for emb, dir in examples:
        embs.append((emb, dir['label'], 'DIR'))

    x_emb, labels, types = list(zip(*embs))
    x_emb = torch.stack(x_emb).numpy()
    tsne_emb = TSNE(n_components=2, learning_rate='auto', init='random', metric='l2',
                    perplexity=3, random_state=32).fit_transform(x_emb)

    del embs
    del zipped
    del x_emb
    x, y = list(zip(*tsne_emb))
    df_embs = pd.DataFrame(list(zip(x, y, labels, types)), columns=['x', 'y', 'label', 'type'])

    df_neg = df_embs.loc[df_embs['label'] == 'NEG']
    df_pos = df_embs.loc[df_embs['label'] != 'NEG']
    df_pos = df_pos.sort_values(by=['type'], ascending=True)
    sns.scatterplot(data=df_neg, x="x", y="y", color="grey", alpha=0.1, size=1)
    sns.scatterplot(data=df_pos, x="x", y="y", hue="label",
                    style="type", style_order=['ART', 'DIR'],
                    palette=sns.color_palette("husl", len(celexs)))

    plt.legend('', frameon=False)
    plt.axis('off')
    plt.xlabel('')
    plt.ylabel('')
    plt.title('t-SNE visualisation of {0} directives and their implementations'.format(len(celexs)))
    plt.savefig(os.path.join(args.embeddings, 'tsne.png'))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-q",
        "--queries",
        default="../../output/dataset/test-queries.jsonl",
        type=str,
        help="Queries pretokenized json file.",
    )

    parser.add_argument(
        "-c",
        "--corpus",
        default="../../output/dataset/corpus.jsonl",
        type=str,
        help="Corpus pretokenized json file.",
    )

    parser.add_argument(
        "--embeddings",
        default="../../output/embeddings_ITA_512",
        type=str,
        help="Path to the folder containing queries and corpus embeddings.",
    )

    parser.add_argument(
        "--country",
        default="ITA",
        type=str,
        help="Country",
    )

    parser.add_argument(
        '--cpu',
        action='store_true'
    )

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
