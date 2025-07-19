#!/usr/bin/env python3
"""
enhanced_lexrank_evaluation_articles.py

Run LexRank+SpaCy-Chains summarizer on medical papers,
evaluate using ROUGE, BLEU, METEOR, BERTScore, SBERT cosine.

Usage:
    python enhanced_lexrank_evaluation_articles.py \
        --input_jsonl articles_pmc_cancer_diabetes_1000.jsonl \
        --output_csv enhanced_results.csv
"""
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from nltk.tokenize import sent_tokenize
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from bert_score import BERTScorer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rouge import Rouge
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.text_rank import TextRankSummarizer
from lexical_chains_spacy import build_similarity_chains, select_chain_sentences
from tqdm import tqdm

SMOOTH = SmoothingFunction().method1

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "figure.dpi": 300,
    "axes.grid": True,
    "grid.linestyle": "--",
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 11,
    "lines.linewidth": 2
})

def load_data(jsonl_path):
    df = pd.read_json(jsonl_path, lines=True)
    df = df.dropna(subset=['body', 'abstract']).reset_index(drop=True)
    print(f"[INFO] Loaded {len(df)} documents.")
    return df

def generate_summaries(df, max_words=350, top_k=5, per_chain=2):
    summarizer = LexRankSummarizer()
    tokenizer = Tokenizer("english")
    all_summaries = []

    for text in tqdm(df['body'], desc=f"Generating summaries (limit {max_words} words)"):
        sents = sent_tokenize(text)
        if not sents:
            all_summaries.append("")
            continue

        chains = build_similarity_chains(sents, top_k=top_k)
        picks = select_chain_sentences(chains, per_chain=per_chain)
        candidates = [sents[i] for i in range(len(sents)) if i in picks]
        if not candidates:
            candidates = [sents[0]]

        parser = PlaintextParser.from_string(" ".join(candidates), tokenizer)
        chosen = summarizer(parser.document, len(candidates))
        summary = " ".join(str(s) for s in chosen)

        words = summary.split()
        if len(words) > max_words:
            summary = " ".join(words[:max_words])

        all_summaries.append(summary)

    return all_summaries

def evaluate_all(hyps, refs, bert_scorer, sbert_model):
    paired = [(h, r) for h, r in zip(hyps, refs) if h.strip()]
    if not paired:
        return {m: 0.0 for m in (
            ['rouge-1_p','rouge-1_r','rouge-1_f',
             'rouge-2_p','rouge-2_r','rouge-2_f',
             'rouge-l_p','rouge-l_r','rouge-l_f',
             'bleu','meteor','bert_p','bert_r','bert_f','sbert_cos'])}

    hyp_list, ref_list = zip(*paired)

    rouge = Rouge()
    rouge_scores = rouge.get_scores(list(hyp_list), list(ref_list), avg=True)
    out = {}
    for n in ['1','2','l']:
        for m in ['p','r','f']:
            out[f'rouge-{n}_{m}'] = rouge_scores[f'rouge-{n}'][m]

    bleu_scores = [
        sentence_bleu([r.split()], h.split(),
                      smoothing_function=SMOOTH,
                      weights=(0.5, 0.5, 0, 0))
        for h, r in zip(hyp_list, ref_list)
    ]
    out['bleu'] = float(np.mean(bleu_scores))

    meteor_scores = [
        meteor_score([r.split()], h.split())
        for h, r in zip(hyp_list, ref_list)
    ]
    out['meteor'] = float(np.mean(meteor_scores))

    P, R, F1 = bert_scorer.score(list(hyp_list), list(ref_list))
    out['bert_p'] = float(P.mean().item())
    out['bert_r'] = float(R.mean().item())
    out['bert_f'] = float(F1.mean().item())

    hyp_emb = sbert_model.encode(list(hyp_list), convert_to_numpy=True, show_progress_bar=False)
    ref_emb = sbert_model.encode(list(ref_list), convert_to_numpy=True, show_progress_bar=False)
    cos_sims = [
        cosine_similarity(hyp_emb[i].reshape(1, -1),
                          ref_emb[i].reshape(1, -1))[0][0]
        for i in range(len(hyp_emb))
    ]
    out['sbert_cos'] = float(np.mean(cos_sims))

    return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_jsonl', default='400cleaned_medical_report.jsonl')
    parser.add_argument('--output_csv', default='finalresults.csv')
    args = parser.parse_args()

    df = load_data(args.input_jsonl)
    df = df.head(200)  # Limit to first 10 articles
    sizes = [150, 200, 250, 300, 350]  # Max words per summary
    records = []

    bert_scorer = BERTScorer(lang="en", rescale_with_baseline=True)
    sbert_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

    for word_limit in sizes:
        print(f"[RUN] Summarizing with {word_limit} words…")
        hyps = generate_summaries(df, max_words=word_limit)
        print("[EVAL] Evaluating…")
        metrics = evaluate_all(hyps, df['abstract'].tolist(), bert_scorer, sbert_model)
        print("[DONE]")

        row = {'max_words': word_limit}
        row.update(metrics)
        records.append(row)

    results_df = pd.DataFrame(records)
    results_df.to_csv(args.output_csv, index=False)
    print(f"[OUT] Saved all metrics to {args.output_csv}")

    def format_plot(ax, title, ylabel):
        ax.set_title(title)
        ax.set_xlabel("Maximum Words")
        ax.set_ylabel(ylabel)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(50))
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)

    fig, ax = plt.subplots()
    for n in ['1','2','l']:
        ax.plot(results_df['max_words'],
                results_df[f'rouge-{n}_f'],
                marker='o', label=f'ROUGE-{n} F1')
    format_plot(ax, 'ROUGE F1 vs. Summary Length', 'F1 Score')
    plt.tight_layout()
    plt.savefig('plot_rouge_articles.png')

    fig, ax = plt.subplots()
    ax.plot(results_df['max_words'], results_df['bleu'], marker='o', label='BLEU')
    ax.plot(results_df['max_words'], results_df['meteor'], marker='s', label='METEOR')
    format_plot(ax, 'BLEU & METEOR vs. Summary Length', 'Score')
    plt.tight_layout()
    plt.savefig('plot_bleu_meteor_articles.png')

    fig, ax = plt.subplots()
    ax.plot(results_df['max_words'], results_df['bert_f'], marker='^', label='BERTScore F1')
    #ax.plot(results_df['max_words'], results_df['sbert_cos'], marker='v', label='SBERT Cosine')
    format_plot(ax, 'Embedding-based Metrics vs. Summary Length', 'Score')
    plt.tight_layout()
    plt.savefig('plot_embedding_articles.png')

    print("[OUT] Publication-quality plots saved: plot_rouge_articles.png, plot_bleu_meteor_articles.png, plot_embedding_articles.png")
    plt.show()

if __name__ == '__main__':
    main()
