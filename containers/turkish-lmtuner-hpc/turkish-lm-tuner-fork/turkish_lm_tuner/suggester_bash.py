import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset, Dataset
import argparse, sys
import pandas as pd

if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    
    parser.add_argument("--misspelled_dataset_name", help = """Dataset containing misspelled words""", type = str, default = "ranker_misspellings_product")
                        
    parser.add_argument("--model_name", help="""Either product_search or general_turkish""", type = str, default = "product_search")

    parser.add_argument("--hf_token_hub", help="""token for huggingface""",  default = None)

    parser.add_argument("--suggestions_repo_name", help = "Repo name for the suggestions",  default = None)

    parser.add_argument("--suggestion_amount", help = "k in the top-k suggestions", type = int, default = 10)
    args=parser.parse_args()
    misspelled_data = args.misspelled_dataset_name
    model_name = args.model_name
    hf_token = args.hf_token_hub
    suggestions_repo = args.suggestions_repo_name
    suggestion_amount = args.suggestion_amount
    tokenizer = AutoTokenizer.from_pretrained("boun-tabi-LMG/TURNA")
    model = AutoModelForSeq2SeqLM.from_pretrained("Holmeister/TURNA_spell_correction_"+model_name)
    
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    misspelled_words = load_dataset("Holmeister/"+misspelled_data, token = hf_token)
    misspelled_df = pd.DataFrame(misspelled_words["train"])
    misspellings = list(misspelled_df["misspellings"])
    corrects = list(misspelled_df["correct_versions"])
    suggestions_df = pd.DataFrame(columns = ["misspellings", "suggestions", "correct_versions", "probabilities", "token_counts" ])
    suggestions = []
    probabilities_list = []
    token_counts = [] 
    misspellings_list = []   
    corrects_list = []
    for source_text, correct in zip(misspellings, corrects):
        source_ids = tokenizer(source_text, return_tensors="pt").input_ids.to(device)
        misspellings_list.extend([source_text for i in range(suggestion_amount)])
        corrects_list.extend([correct for i in range(suggestion_amount)])
        # generate the output using beam search
        beam_outputs = model.generate(
            inputs=source_ids,
            num_beams=suggestion_amount*2,
            num_return_sequences = suggestion_amount,
            min_length=0,
            max_new_tokens = 20,
            length_penalty=0,
            output_scores=True,
            return_dict_in_generate=True,
            early_stopping = True
        )
        transition_scores = model.compute_transition_scores(
            beam_outputs.sequences, beam_outputs.scores, beam_outputs.beam_indices, normalize_logits=True
        )
        probabilities = torch.exp(transition_scores.sum(axis=1))
        #print("Output:\n" + 100 * '-')
        for sequence, probability in zip(beam_outputs.sequences, probabilities):
          decoded_prediction = tokenizer.decode(sequence, skip_special_tokens=True)
          tokenized_predictions = tokenizer.convert_ids_to_tokens(sequence, skip_special_tokens = True)
          #print(tokenizer.convert_ids_to_tokens(sequence, skip_special_tokens = True))
          #print(f"Sequence: {decoded_prediction}, Score: {probability}")
          suggestions.append(decoded_prediction)
          probabilities_list.append(probability.item())
          token_counts.append(len(tokenized_predictions)+1) # +1 for eos token

    suggestions_df["misspellings"] = misspellings_list
    suggestions_df["probabilities"] = probabilities_list
    suggestions_df["token_counts"] = token_counts
    suggestions_df["suggestions"] = suggestions
    suggestions_df["correct_versions"] = corrects_list
    suggestions_hf = Dataset.from_pandas(suggestions_df)
    suggestions_hf.push_to_hub(suggestions_repo, private = True, token = hf_token)
    
        
