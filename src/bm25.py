import argparse
import os
import json
from multiprocessing.pool import Pool
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE

import random
from tqdm import tqdm
from rank_bm25 import *
import spacy
import pandas as pd
import seaborn as sns
from nltk.stem.snowball import SnowballStemmer

at_k = sorted([1, 3, 5, 10, 100], reverse=True)

language_map = {'ITA': 'italian', 'FRA': 'french', 'ESP': 'spanish', 'IRL': 'english', 'AUT': 'german'}

spacy_models = {'italian': 'it_core_news_lg', 'french': 'fr_core_news_lg', 'spanish': 'es_core_news_lg',
                'english': 'en_core_web_lg', 'german': 'de_core_news_lg'}

stop_words = ["o", "e", "i", "ii", "iii", "a", "b", "c"]


def tokenize(split, nlp, stemmer, is_last):
    tokenized = []
    pbar = None

    if is_last:
        pbar = tqdm(total=len(split))

    for elem in split:
        text = elem['text']
        tokens = [stemmer.stem(token.text.lower()) for token in nlp(text) if
                  token.is_alpha and token.text.lower() not in nlp.Defaults.stop_words]

        elem.update({'tokens': tokens})
        tokenized.append(elem)
        if pbar is not None:
            pbar.update(1)

    if pbar is not None:
        pbar.close()

    return tokenized


def prepare_dataset(dataset, nlp, stemmer, num_proc):
    pool = Pool(num_proc)
    processes = []
    splits = [dataset[i * len(dataset) // num_proc: (i + 1) * len(dataset) // num_proc] for i in range(num_proc)]
    for i, split in enumerate(splits):
        processes.append(pool.apply_async(tokenize, args=(split, nlp, stemmer, i == num_proc - 1)))

    pool.close()
    pool.join()

    del dataset
    new_dataset = []
    for t in processes:
        new_dataset.extend(t.get())

    return new_dataset


def compute_score(bm25, samples, corpus, is_last):
    hits = {k: 0 for k in at_k}
    _map = {k: 0 for k in at_k}
    mrr = {k: 0 for k in at_k}
    pbar = None

    if is_last:
        pbar = tqdm(total=len(samples))

    for sample in samples:
        top_k = None
        scores = bm25.get_scores(sample['tokens'])

        for k in at_k:
            if top_k is None:
                top_k = scores.argsort()[-k:][::-1]
            else:
                top_k = top_k[:k]

            set_hits = 0
            ap = 0
            first = True
            for c, j in enumerate(top_k):
                if sample['label'] in corpus[j]['labels']:
                    hits[k] += 1
                    set_hits += 1
                    ap += set_hits / (c + 1)
                    if first:
                        mrr[k] += 1 / (c + 1)
                        first = False

            if set_hits > 0:
                _map[k] += ap / set_hits

        if pbar is not None:
            pbar.update(1)

    if pbar is not None:
        pbar.close()

    return hits, _map, mrr


def update_score(table, subtable):
    for k in table.keys():
        table[k] += subtable[k]

    return table


def run(args, stemmer, nlp):
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
        for line in f:
            data = json.loads(line)
            if data['country'] == args.country and data['label'] in celexs:
                samples.append(data)

    samples = prepare_dataset(samples, nlp, stemmer, args.processes)
    corpus = prepare_dataset(corpus, nlp, stemmer, args.processes)

    bow_dataset = [c['tokens'] for c in corpus]

    bm25 = BM25Okapi(bow_dataset)

    hits = {k: 0 for k in at_k}
    _map = {k: 0 for k in at_k}
    mrr = {k: 0 for k in at_k}

    pool = Pool(args.processes)
    processes = []
    splits = [samples[i * len(samples) // args.processes: (i + 1) * len(samples) // args.processes]
              for i in range(args.processes)]

    for i, split in enumerate(splits):
        processes.append(pool.apply_async(compute_score, args=(bm25, split, corpus, i == args.processes - 1)))

    pool.close()
    pool.join()
    for t in processes:
        p_hits, p_map, p_mrr = t.get()
        hits = update_score(hits, p_hits)
        _map = update_score(_map, p_map)
        mrr = update_score(mrr, p_mrr)

    hits_k = dict([(x[0], x[1] / len(samples)) for x in hits.items()])
    map_k = dict([(x[0], x[1] / len(samples)) for x in _map.items()])
    mrr_k = dict([(x[0], x[1] / len(samples)) for x in mrr.items()])

    print("Hits: ")
    print(hits_k)
    print("\n\nMAP: ")
    print(map_k)
    print("\n\nMRR: ")
    print(mrr_k)

    random.seed(32)
    random.shuffle(samples)
    examples = samples[:25]

    points = []  # [(pos, score, label)]

    for i, ex_sample in tqdm(enumerate(examples), total=len(examples)):
        scores = [x for x in enumerate(bm25.get_scores(ex_sample['tokens']).tolist())]  # i, s
        scores = sorted(scores, key=lambda x: x[1], reverse=True)[:200]
        top_k, top_s = list(zip(*scores))

        top_s = top_s - np.min(top_s)

        for c, j in enumerate(top_k):
            score = top_s[c]
            if ex_sample['label'] in corpus[j]['labels']:
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
    plt.savefig(os.path.join(args.output, 'ranking.png'))

    plt.figure()

    celexs = set([x['label'] for x in examples])
    embs = []
    count = 0
    for pub in corpus:
        if set(pub['labels']) & celexs:
            for label in pub['labels']:
                embs.append((pub['tokens'], label, 'ART'))
        else:
            if count < 2000:
                embs.append((pub['tokens'], 'NEG', 'ART'))
                count += 1

    for dir in examples:
        embs.append((dir['tokens'], dir['label'], 'DIR'))

    bow = [x[0] for x in embs]
    tfidf = TfidfVectorizer(tokenizer=lambda x: x, lowercase=False)
    tfidf_matrix = tfidf.fit_transform(bow)

    tsne_emb = TSNE(n_components=2, learning_rate='auto', init='random', metric='l2',
                    perplexity=3, random_state=32).fit_transform(tfidf_matrix)

    x, y = list(zip(*tsne_emb))
    _, labels, types = list(zip(*embs))

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
    plt.title('t-SNE visualisation of {0} directives and their implementations (tf-idf)'.format(len(celexs)))
    plt.savefig(os.path.join(args.output, 'tsne.png'))


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
        "-o",
        "--output",
        default="../../output/",
        type=str,
        help="Output folder",
    )

    parser.add_argument(
        "--country",
        default="ITA",
        type=str,
        help="Country",
    )

    parser.add_argument(
        "-p",
        "--processes",
        default=15,
        type=int,
        help="Num processes.",
    )

    args = parser.parse_args()

    stemmer = SnowballStemmer(language_map[args.country])
    nlp = spacy.load(spacy_models[language_map[args.country]],
                     disable=["tok2vec", "morphologizer", "tagger", "parser", "attribute_ruler", "lemmatizer",
                              "ner"])

    for w in stop_words:
        nlp.Defaults.stop_words.add(w)

    run(args, stemmer, nlp)


if __name__ == "__main__":
    print('BM-25\n')
    main()
