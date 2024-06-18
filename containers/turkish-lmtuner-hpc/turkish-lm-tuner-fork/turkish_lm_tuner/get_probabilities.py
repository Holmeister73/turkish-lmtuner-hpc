import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset
import argparse, sys
import pandas as pd
import torch.nn as nn


def compute_scores_for_target(input_text, target_text, model, tokenizer):

  
    target_text += tokenizer.eos_token
    # Alternative 1. Following the code in https://github.com/neulab/BARTScore/blob/main/WMT/bart_score.py
    # We will try to get the scores associated with specific sentences
    loss_fct = nn.NLLLoss(reduction="none", ignore_index=model.config.pad_token_id)
    lsm = nn.LogSoftmax(dim=1) # first applies the softmax to ensure all values are
    
    # Encode both the input text and also the target text (i.e., what we would be expecting to obtain)
    # The idea is that since, we know exactly what outputs we expect, we can simply forward the method and get
    # the corresponding scores.
    encoded_src = tokenizer(input_text, max_length=1024, truncation=True, padding=True, return_tensors="pt", add_special_tokens=False).to(device)
    encoded_tgt = tokenizer(target_text, max_length=1024,  truncation=True, padding=True, return_tensors="pt", add_special_tokens=False).to(device)
    print(encoded_tgt)
    output = model(
        input_ids=encoded_src["input_ids"],
        attention_mask=encoded_src["attention_mask"],
        labels=encoded_tgt["input_ids"],
    )
    logits = output.logits.view(-1, model.config.vocab_size)
    lsm_logits = lsm(logits)
    loss = loss_fct(lsm_logits, encoded_tgt["input_ids"].view(-1))
    loss = loss.view(encoded_tgt["input_ids"].shape[0], -1)
    #loss = torch.exp(-1 * loss)
    
    score_list = [x.item() for x in loss.squeeze(dim=0)]
    probability = np.exp(-sum(score_list))
    #print([tokenizer.decode(t) for t in encoded_tgt["input_ids"][0]])
    return probability, [tokenizer.decode(t) for t in encoded_tgt["input_ids"][0]]


def compute_target_scores(prompt: str, targets: list, model, tokenizer) -> tuple:
    scores = []
    word_pieces = []

    for target in targets:
        score, word_piece = compute_scores_for_target(prompt, target, model, tokenizer)
        scores.append(score)
        word_pieces.append(word_piece)
      
  return scores, word_pieces
if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    
    parser.add_argument("--misspelled_and_candidate_dataset_name", help = """Dataset containing misspelled words and their candidate suggestions""", type = str, default = "ranker_misspellings_and_candidates_product")
                        
    parser.add_argument("--model_name", help="""Either product_search or general_turkish""", type = str, default = "product_search")

    parser.add_argument("--hf_token_hub", help="""token for huggingface""",  default = None)

    parser.add_argument("--probabilities_repo_name", help = "Repo name for the calculated probabilities",  default = None)
    args=parser.parse_args()
    misspelled_and_candidate_data = args.misspelled_and_candidate_dataset_name
    model_name = args.model_name
    hf_token = args.hf_token_hub
    probabilities_repo = args.probabilities_repo_name
    
    tokenizer = AutoTokenizer.from_pretrained("boun-tabi-LMG/TURNA")
    model = AutoModelForSeq2SeqLM.from_pretrained("Holmeister/TURNA_spell_correction_"+model_name)
    
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    misspelled_and_candidate_words = load_dataset("Holmeister/"+misspelled_and_candidate_data, token = hf_token)
    misspelled_and_candidate_df = pd.DataFrame(misspelled_and_candidate_words["train"])
    with_probabilities_df = pd.Dataframe(columns = ["misspellings", "candidates", "probabilities", "token_counts"]
    probabilities = []
    token_counts = []
    candidates = []
    old_misspelling = with_probabilities_df["misspellings"][0]
    
    for row in misspelled_and_candidate_df.itertuples():
        misspelling = row[1]
        candidates.append(row[2])
        if misspelling != old_misspelling:
            old_misspelling = misspelling
            prompt = old_misspelling
            scores, word_pieces = compute_target_scores(prompt, candidates, model, tokenizer)
            probabilities.extend(scores)
            token_counts.extend([len(wordpiece) for wordpiece in word_pieces])
            candidates = []

    prompt = misspelling
    scores, word_pieces = compute_target_scores(prompt, candidates, model, tokenizer)
    probabilities.extend(scores)
    token_counts.extend([len(wordpiece) for wordpiece in word_pieces])
    
    with_probabilities_df["misspellings"] = misspelled_and_candidate_df["misspellings"]
    with_probabilities_df["candidates"] = misspelled_and_candidate_df["candidates"]
    with_probabilities_df["probabilities"] = probabilities
    with_probabilities_df["token_counts"] = token_counts
    with_probabilities_hf = Dataset.from_pandas(with_probabilities_df)
    with_probabilities_hf.push_to_hub(probabilities_repo, private = True, token = hf_token)
    
