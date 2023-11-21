import argparse
import os
import json
import pickle
import torch

from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


def run(args):
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    model = AutoModel.from_pretrained(args.model)
    model = model.to(device)
    model.eval()

    print('Loading...')
    corpus = []
    with open(args.input, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            if data['country'] == args.country:
                corpus.append(data['text'])

    # create chunks
    print('Tokenizing...')
    encoded_inputs = tokenizer(corpus, truncation=True, padding='max_length', max_length=args.length,
                               return_tensors='pt')

    del corpus
    # compute embeddings
    print('Running...')
    embeddings = torch.empty([0, 768])
    pbar = tqdm(total=len(encoded_inputs.encodings))
    for i in range((len(encoded_inputs.encodings) // args.batch) + 1):
        input_ids = encoded_inputs['input_ids'][i * args.batch:i * args.batch + args.batch].to(device)
        attention_mask = encoded_inputs['attention_mask'][i * args.batch:i * args.batch + args.batch].to(device)
        if len(input_ids) == 0:
            break
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        pool = mean_pooling(outputs, attention_mask).cpu()
        embeddings = torch.cat([embeddings, pool], 0)
        pbar.update(len(input_ids))

    pbar.close()

    e_norm = torch.norm(embeddings, p=2, dim=1, keepdim=True).clamp(min=1e-12)
    e_norm = torch.div(embeddings, e_norm)
    with open(args.output, 'wb') as f:
        pickle.dump(e_norm.cpu(), f)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-m",
        "--model",
        default="sentence-transformers/distiluse-base-multilingual-cased-v1",
        type=str,
        help="Sentence BERT model.",
    )

    parser.add_argument(
        "-t",
        "--tokenizer",
        default="sentence-transformers/distiluse-base-multilingual-cased-v1",
        type=str,
        help="Sentence BERT model.",
    )

    parser.add_argument(
        "-i",
        "--input",
        default="../../output/dataset/test-queries.jsonl",
        type=str,
        help="Input json file.",
    )

    parser.add_argument(
        "-o",
        "--output",
        default="../../output/embeddings_ITA/queries_embeddings.pickle",
        type=str,
        help="Output pickle file for embedding.",
    )

    parser.add_argument(
        "--country",
        default="ITA",
        type=str,
        help="Country",
    )

    parser.add_argument(
        "-b",
        "--batch",
        default=300,
        type=int,
        help="Batch size.",
    )

    parser.add_argument(
        "-l",
        "--length",
        default=512,
        type=int,
        help="sequence size.",
    )

    args = parser.parse_args()
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    with torch.no_grad():
        run(args)


if __name__ == "__main__":
    main()