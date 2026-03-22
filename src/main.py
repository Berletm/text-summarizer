import json
from dataclasses import dataclass
from typing import List, Tuple
import re
import pymorphy3
from razdel import sentenize
from constants import PTH, STOP_POS, STOP_WORDS, MAX_SYMBOLS
from collections import Counter
import numpy as np
from rouge_score import rouge_scorer

morph = pymorphy3.MorphAnalyzer()

@dataclass
class Document:
    text: str
    summary: str

def read_corpus(pth: str = PTH) -> List[Document]:
    corpus = []  
      
    with open(pth, "r", encoding="utf-8") as file:
        for line in file.readlines():
            data = json.loads(line)
            corpus.append(Document(data["text"], data["summary"]))
    
    return corpus

def split_sentences(text: str) -> List[str]:
    return [s.text for s in sentenize(text)]

def process_sentence(sentence: str) -> Tuple[List[str], List[float]]:
    words = re.findall(r"\w+", sentence.lower())
    res = []
    for w in words:
        if w.isdigit() or w in STOP_WORDS:
            continue
        
        parsed_w = morph.parse(w)[0]
        if parsed_w.tag.POS not in STOP_POS:
            res.append(parsed_w.normal_form)
    
    return res

def tfidf(sentences: List[str]) -> List[Tuple[float, int]]:    
    n_sentences = len(sentences)
    processed_sents = [process_sentence(s) for s in sentences]
    
    global_word_freq = Counter()
    for words in processed_sents:
        global_word_freq.update(set(words))
    
    
    scores = []
    for i, words in enumerate(processed_sents):
        if not words:
            scores.append((0.0, i))
            continue
            
        word_counts = Counter(words)
        total_words_in_sent = len(words)
        
        tf_idf_sum = 0.0
        for word, count in word_counts.items():
            tf = count / total_words_in_sent
            
            df = global_word_freq[word]
            idf = np.log((n_sentences + 1) / (df + 1)) + 1
            
            tf_idf_sum += tf * idf
        
        avg_score = tf_idf_sum
        if i == 0:
            avg_score *= 1.5
        elif i < 3:
            avg_score *= 1.2
        scores.append((avg_score, i))
    
    return scores           

def generate_summary(sentences: List[str], weights: List[Tuple[float, int]]) -> str:
    sorted_weights = sorted(weights, key=lambda x: x[0], reverse=True)
    
    main_sentences = []
    
    for i, (w, ind) in enumerate(sorted_weights):
        if i == 0:
            main_sentences.append((w, ind))
            continue
        
        if abs(main_sentences[-1][1] - ind) == 1:
            continue
        
        main_sentences.append((w, ind))
        
    main_sentences = sorted(main_sentences, key=lambda x: x[1])
        
    s = []
    l = 0
    for w, i in main_sentences:
        if l + len(sentences[i]) >= MAX_SYMBOLS: break
        s.append(sentences[i])
        l += len(sentences[i]) 
    summary = " ".join(s)
    
    return summary
    
class Tokenizer:
    @staticmethod
    def tokenize(text: str) -> list:
        words = re.findall(r'\w+', text.lower())
        return [morph.parse(w)[0].normal_form for w in words]

def main() -> None:
    corpus = read_corpus()
    
    scorer = rouge_scorer.RougeScorer(
        ['rouge1', 'rouge2', 'rougeL', 'rougeLsum'], 
        use_stemmer=False,
        tokenizer=Tokenizer
    )
    
    avg_metrics = {"rouge1": None, "rouge2": None, "rougeL": None, "rougeLsum": None}
    
    for i, doc in enumerate(corpus[:4], 1):
        text = doc.text
        reference = doc.summary
        
        sentences = split_sentences(text)
        weights = tfidf(sentences)
        
        summary = generate_summary(sentences, weights)
        
        scores = scorer.score(reference, summary)
        print(reference)
        print(summary)

        for metric, score in scores.items():
            if avg_metrics[metric] is None:
                avg_metrics[metric] = (score.precision, score.recall, score.fmeasure)
            else:
                old_p, old_r, old_f = avg_metrics[metric]
                avg_metrics[metric] = (old_p + score.precision, old_r + score.recall, old_f + score.fmeasure)
        p, r, f = avg_metrics["rouge1"]
        print(p / i, r / i, f / i)
        print()
    n = len(corpus)
    for metric in avg_metrics.keys():
        p, r, f = avg_metrics[metric]
        avg_metrics[metric] = (p/n, r/n, f/n)
    
    print(avg_metrics)
    
if __name__ == "__main__":
    main()
    