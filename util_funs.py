import torch
import nltk
import pandas as pd
from torch.utils.data import Dataset
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from collections import Counter
from transformers import AutoTokenizer

## Evaluation Variables

# Needed for BLEU score with nltk.word_tokenize
nltk.download('punkt_tab')
rouge = Rouge()
cc = SmoothingFunction()

## Datasets

class TrainTextDataset(Dataset):
    def __init__(self, text_file, llm_path, max_length, device, num_mem=0, prompt_len=1):
        """
        Create the training or evaluation dataset.

        Args:
            text_file (str): Path for lines of texts.
            llm_path (str): Path for the base LLM.
            max_length (int): Max number of tokens to be compressed.
        """
        self.text = [line.strip() for line in open(text_file).readlines()]
        self.tokenizer = AutoTokenizer.from_pretrained(llm_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_length = max_length
        self.num_mem = num_mem       # for ICAE only
        self.prompt_len = prompt_len # number of prompt tokens; use <bos> which is a single token
        self.device = device
        self.eos_2_id = 128001 # specific to Llama 3

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        text_tokens = self.tokenizer(
            self.text[idx], 
            truncation=True, 
            padding="max_length", 
            max_length=self.max_length, 
            return_tensors="pt",
            add_special_tokens=False
        ).input_ids
        input_ids = text_tokens.squeeze()
        prefix_len = self.num_mem + self.prompt_len
        # input text tokens + EOS token
        target_tokens = torch.full((prefix_len+len(input_ids),), -100, dtype=torch.long)
        text_eos_tokens = input_ids.tolist()
        text_eos_tokens.append(self.eos_2_id) # using other eos token ID 128001 for Llama-3
        text_eos_tokens_len = len(text_eos_tokens)
        target_tokens[prefix_len-1:prefix_len-1+text_eos_tokens_len] = torch.tensor(text_eos_tokens, dtype=torch.long, device=self.device)
        return {"input_ids": input_ids, "labels": target_tokens}

class PredictionTextDataset(Dataset):
    def __init__(self, filepath, start_line=0):
        """
        Collect texts.

        Args:
            filepath (str): Path for lines of texts.
            start_line (int): The line start to be loaded.
        """
        self.lines = open(filepath).readlines()[start_line:]

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        return self.lines[idx].strip()

## Evaluation

def calculate_metrics_averages(csv_file):
        data = pd.read_csv(csv_file)
        averages = data.mean()
        return averages

def compute_exact_match(reference, candidate):
    reference = reference.strip()
    candidate = candidate.strip()
    for i, (r,c) in enumerate(zip(reference, candidate)):
        if r != c:
            return i / len(reference)
    return (i+1) / len(reference)

def compute_f1_score(reference_tokens, candidate_tokens):
    common = Counter(reference_tokens) & Counter(candidate_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(candidate_tokens)
    recall = num_same / len(reference_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def compute_metrics(reference, candidate):
    scores = {}

    # Compute Rouge scores
    rouge_scores = rouge.get_scores(candidate, reference)[0]
    for key in ['rouge-1', 'rouge-2', 'rouge-l']:
        scores[f'{key}-p'] = rouge_scores[key]['p']
        scores[f'{key}-r'] = rouge_scores[key]['r']
        scores[f'{key}-f'] = rouge_scores[key]['f']

    # Compute BLEU score
    reference_tokens = nltk.word_tokenize(reference)
    candidate_tokens = nltk.word_tokenize(candidate)
    scores['bleu'] = sentence_bleu([reference_tokens], candidate_tokens, smoothing_function=cc.method1)

    # Compute exact match
    scores['exact_match'] = compute_exact_match(reference, candidate)
    
    # Compute F1 score
    scores['f1'] = compute_f1_score(reference_tokens, candidate_tokens)

    return scores

def main_evaluate(ref_file, cand_file, output_file=None):
    with open(ref_file, 'r', encoding='utf-8') as ref_f, open(cand_file, 'r', encoding='utf-8') as cand_f:
        references = ref_f.readlines()
        candidates = cand_f.readlines()

    results = [compute_metrics(reference.strip(), candidate.strip()) for reference, candidate in zip(references, candidates)]

    if not output_file:
        print(results)
        return
    # Write results to CSV file
    with open(output_file, 'w', encoding='utf-8') as f:
        headers = [
            "line_number",
            "rouge-1-p", "rouge-1-r", "rouge-1-f",
            "rouge-2-p", "rouge-2-r", "rouge-2-f",
            "rouge-l-p", "rouge-l-r", "rouge-l-f",
            "bleu",
            "exact_match",  # Add exact match to headers
            "f1"  # Add F1 score to headers
        ]
        f.write(",".join(headers) + "\n")
        for idx, result in enumerate(results):
            line_values = [
                f"{idx + 1}",
                f"{result['rouge-1-p']:.4f}", f"{result['rouge-1-r']:.4f}", f"{result['rouge-1-f']:.4f}",
                f"{result['rouge-2-p']:.4f}", f"{result['rouge-2-r']:.4f}", f"{result['rouge-2-f']:.4f}",
                f"{result['rouge-l-p']:.4f}", f"{result['rouge-l-r']:.4f}", f"{result['rouge-l-f']:.4f}",
                f"{result['bleu']:.4f}",
                f"{result['exact_match']:.4f}",  # Include exact match in line values
                f"{result['f1']:.4f}"  # Include F1 score in line values
            ]
            f.write(",".join(line_values) + "\n")
