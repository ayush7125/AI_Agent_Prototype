# src/evaluator.py — evaluation utilities
from rouge_score import rouge_scorer
from bert_score import score as bert_score

scorer = rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL'], use_stemmer=True)

def compute_rouge(pred: str, ref: str):
    return scorer.score(ref, pred)

def compute_bertscore(preds, refs, model_type='microsoft/deberta-xlarge-mnli'):
    P, R, F = bert_score(preds, refs, model_type=model_type, lang='en', rescale_with_baseline=True)
    return {'P': P.tolist(), 'R': R.tolist(), 'F': F.tolist()}
