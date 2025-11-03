# ============================== CB-LLM (Classification): ACS Label Builder ==============================
# This script implements the "Automatic Concept Scoring (ACS)" step for CB-LLMs (classification).
# It converts each input text into a vector and each human-readable concept into a vector using a chosen
# sentence encoder (MPNet / SimCSE / ANGLE), then computes cosine-similarity scores between them.
# These similarity matrices become pseudo "concept labels" used to train the Concept Bottleneck Layer (CBL).
#
# Paper alignment:
#   • ACS (Eq. 1): build concept-score labels S(x, c) via text–concept similarity
#   • ACC (Eq. 2): not in this file; applied later during CBL training to correct noisy ACS scores
#   • CBL training (Eq. 3): consumes the saved .npy similarity matrices from this script
#   • Final Linear (Eq. 4): trained after the CBL, to predict class labels from concept activations
# =======================================================================================================

import argparse
import os
import torch
import torch.nn.functional as F
import numpy as np
from datasets import load_dataset
import config as CFG
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
from utils import mean_pooling, decorate_dataset, decorate_concepts
import sys
import time

parser = argparse.ArgumentParser()

# Use GPU if available; CPU otherwise. ANGLE can be heavy on CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset key used by Hugging Face Datasets and by CFG.* mappings (e.g., example field names, concept set).
parser.add_argument("--dataset", type=str, default="SetFit/sst2")

# Decide which sentence encoder to use for text–concept similarity:
#   mpnet  : sentence-transformers/all-mpnet-base-v2 (encoder-only)
#   simcse : princeton-nlp/sup-simcse-bert-base-uncased (encoder-only)
#   angle  : LLaMA-7B + LoRA NLI head (decoder-only, heavier; needs PEFT weights)
parser.add_argument("--concept_text_sim_model", type=str, default="mpnet", help="mpnet, simcse or angle")

# Tokenization / dataloader knobs
parser.add_argument("--max_length", type=int, default=512)
parser.add_argument("--num_workers", type=int, default=0)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
args = parser.parse_args()

# ------------------------------ Lightweight wrapper for encoded batches ------------------------------
class SimDataset(torch.utils.data.Dataset):
    """
    Holds already-tokenized inputs as simple tensors for fast DataLoader batching.
    encode_sim is a dict with keys like 'input_ids', 'attention_mask', each a list/array per example.
    """
    def __init__(self, encode_sim):
        self.encode_sim = encode_sim

    def __getitem__(self, idx):
        t = {key: torch.tensor(values[idx]) for key, values in self.encode_sim.items()}

        return t

    def __len__(self):
        return len(self.encode_sim['input_ids'])

def build_sim_loaders(encode_sim):
    """
    Build DataLoader for concept-text similarity encoding.
    Returns a DataLoader that yields batches of tokenized inputs.
    """
    dataset = SimDataset(encode_sim)
    if args.concept_text_sim_model == 'angle': # ANGLE is large model; smaller batches
        batch_size = 8
    else:
        batch_size = 256
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=args.num_workers, shuffle=False)
    return dataloader


# ======================================= LOAD DATA & ENCODERS =======================================
print("loading data...")
train_dataset = load_dataset(args.dataset, split='train')
if args.dataset == 'SetFit/sst2':
    val_dataset = load_dataset(args.dataset, split='validation')
print("training data len: ", len(train_dataset))
if args.dataset == 'SetFit/sst2':
    print("val data len: ", len(val_dataset))

# Human-readable concept strings for the selected dataset (defined in config.py).
# These are the "interpretable" concepts the CBL will learn to represent.
concept_set = CFG.concept_set[args.dataset]
print("concept len: ", len(concept_set))

# -------------------- Choose and load a sentence encoder (embedding function) --------------------
# We only need vector representations (no gradients), so we put models in eval() and use no_grad().
if args.concept_text_sim_model == 'mpnet': # Encoder-only model (good all-purpose STS)
    # Used for: General purpose semantic textual similarity (e.g. paraphrase identification)
    print("tokenizing and preparing mpnet")
    tokenizer_sim = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
    sim_model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2').to(device)
    sim_model.eval()
elif args.concept_text_sim_model == 'simcse': # Encoder-only BERT-based model fine-tuned with contrastive learning
    # Used for: Produce high-quality sentence embeddings that capture semantic similarity
    print("tokenizing and preparing simcse")
    tokenizer_sim = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")
    sim_model = AutoModel.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased").to(device)
    sim_model.eval()
elif args.concept_text_sim_model == 'angle': # ANGLE uses a decoder LM (LLaMA-7B) adapted for NLI-style similarity via LoRA (PEFT).
    # We load the base model and then apply the LoRA delta weights (PeftModel)
    print("tokenizing and preparing angle")
    config = PeftConfig.from_pretrained('SeanLee97/angle-llama-7b-nli-v2')
    tokenizer_sim = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
    sim_model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path).bfloat16()
    sim_model = PeftModel.from_pretrained(sim_model, 'SeanLee97/angle-llama-7b-nli-v2')
    sim_model = sim_model.to(device)
    sim_model.eval()
    # For ANGLE we optionally reformat examples and concepts for prompt-style inputs (utils.* helpers).
    train_dataset = train_dataset.map(decorate_dataset, fn_kwargs={"d": args.dataset})
    if args.dataset == 'SetFit/sst2':
        val_dataset = val_dataset.map(decorate_dataset, fn_kwargs={"d": args.dataset})
    concept_set = decorate_concepts(concept_set)
else:
    raise Exception("concept-text sim model should be mpnet, simcse or angle")

# -------------------- Tokenize the dataset texts for the chosen encoder --------------------
# We map each example’s raw text field (CFG.example_name[dataset]) through the tokenizer into input_ids & attention_mask.
encoded_sim_train_dataset = train_dataset.map(
    lambda e: tokenizer_sim(e[CFG.example_name[args.dataset]], padding=True, truncation=True,
                            max_length=args.max_length), batched=True,
    batch_size=len(train_dataset))
# Remove original text columns to keep only model inputs.
encoded_sim_train_dataset = encoded_sim_train_dataset.remove_columns([CFG.example_name[args.dataset]])
# Same for validation (SST-2 only, used later for validation concept-labels).
if args.dataset == 'SetFit/sst2':
    encoded_sim_train_dataset = encoded_sim_train_dataset.remove_columns(['label_text'])
if args.dataset == 'dbpedia_14':
    encoded_sim_train_dataset = encoded_sim_train_dataset.remove_columns(['title'])
encoded_sim_train_dataset = encoded_sim_train_dataset[:len(encoded_sim_train_dataset)]

if args.dataset == 'SetFit/sst2':
    encoded_sim_val_dataset = val_dataset.map(
        lambda e: tokenizer_sim(e[CFG.example_name[args.dataset]], padding=True, truncation=True,
                                max_length=args.max_length), batched=True,
        batch_size=len(val_dataset))
    encoded_sim_val_dataset = encoded_sim_val_dataset.remove_columns([CFG.example_name[args.dataset]])
    if args.dataset == 'SetFit/sst2':
        encoded_sim_val_dataset = encoded_sim_val_dataset.remove_columns(['label_text'])
    if args.dataset == 'dbpedia_14':
        encoded_sim_val_dataset = encoded_sim_val_dataset.remove_columns(['title'])
    encoded_sim_val_dataset = encoded_sim_val_dataset[:len(encoded_sim_val_dataset)]

# Tokenize concept strings; we’ll embed these once and reuse for all examples.
encoded_c = tokenizer_sim(concept_set, padding=True, truncation=True, max_length=args.max_length)

# DataLoaders for efficient batching of (already tokenized) examples.
train_sim_loader = build_sim_loaders(encoded_sim_train_dataset)
if args.dataset == 'SetFit/sst2':
    val_sim_loader = build_sim_loaders(encoded_sim_val_dataset)

# ======================================= GET CONCEPT LABELS (ACS) =======================================
# Goal: compute S(x, c) = cosine( f(text=x), f(concept=c) ) for all examples x and concepts c.
# The resulting matrices are saved to .npy and later used to train the Concept Bottleneck Layer (CBL).
print("getting concept labels...")
encoded_c = {k: torch.tensor(v).to(device) for k, v in encoded_c.items()}

with torch.no_grad():
     # -------------------- Embed the concepts once --------------------
    if args.concept_text_sim_model == 'mpnet':
        # Sentence-level embedding: Mean_pooling averages token embeddings (weighted by attention mask) --> single vector for concept
        concept_features = sim_model(input_ids=encoded_c["input_ids"], attention_mask=encoded_c["attention_mask"])
        concept_features = mean_pooling(concept_features, encoded_c["attention_mask"])
    elif args.concept_text_sim_model == 'simcse':
        # For SimCSE, we use the pooler output directly as the sentence embedding
        concept_features = sim_model(input_ids=encoded_c["input_ids"], attention_mask=encoded_c["attention_mask"], output_hidden_states=True, return_dict=True).pooler_output
    elif args.concept_text_sim_model == 'angle':
        # For Angle model, we take the last token's hidden state from the final layer as the sentence embedding
        # ANGLE is decoder based generative model, so use last hidden state of final token ([:, -1]) as the sentence embedding
        concept_features = sim_model(output_hidden_states=True, input_ids=encoded_c["input_ids"], attention_mask=encoded_c["attention_mask"]).hidden_states[-1][:, -1].float()
    else:
        raise Exception("concept-text sim model should be mpnet, simcse or angle")
    
    # L2-normalize so dot-product equals cosine similarity (since ||u||=||v||=1 → u·v = cosθ).
    concept_features = F.normalize(concept_features, p=2, dim=1) # Apply L2 normalization to concept embeddings, so each vector has unit length. Later used for cosine similarity (text_features @ concept_features.T)

# -------------------- Embed all texts and compute similarities to all concepts --------------------
print("calculating concept-text similarities...")
start = time.time() # how long embedding computation takes
train_sim = []
for i, batch_sim in enumerate(train_sim_loader): #Iterates through the training DataLoader in batches.
    print("batch ", str(i), end="\r")
    batch_sim = {k: v.to(device) for k, v in batch_sim.items()}
    with torch.no_grad():
        # Embed batch of texts into the same vector space as concepts.
        if args.concept_text_sim_model == 'mpnet':
            text_features = sim_model(input_ids=batch_sim["input_ids"], attention_mask=batch_sim["attention_mask"])
            text_features = mean_pooling(text_features, batch_sim["attention_mask"])
        elif args.concept_text_sim_model == 'simcse':
            text_features = sim_model(input_ids=batch_sim["input_ids"], attention_mask=batch_sim["attention_mask"], output_hidden_states=True, return_dict=True).pooler_output
        elif args.concept_text_sim_model == 'angle':
            text_features = sim_model(output_hidden_states=True, input_ids=batch_sim["input_ids"], attention_mask=batch_sim["attention_mask"]).hidden_states[-1][:, -1].float()
        else:
            raise Exception("concept-text sim model should be mpnet, simcse or angle")
        text_features = F.normalize(text_features, p=2, dim=1)
    
    # ACS scores: cosine similarities between each text in the batch and all concepts
    # Shape: [batch_size, num_concepts]
    train_sim.append(text_features @ concept_features.T)


# Concatenate all batch results into a single matrix:
# Final shape: [num_train_examples, num_concepts]
train_similarity = torch.cat(train_sim, dim=0).cpu().detach().numpy()
end = time.time()
print("time of concept scoring:", (end-start)/3600, "hours")

# Optional: also compute ACS for validation split (useful for monitoring during CBL training).
if args.dataset == 'SetFit/sst2':
    val_sim = []
    for batch_sim in val_sim_loader:
        batch_sim = {k: v.to(device) for k, v in batch_sim.items()}
        with torch.no_grad():
            if args.concept_text_sim_model == 'mpnet':
                text_features = sim_model(input_ids=batch_sim["input_ids"], attention_mask=batch_sim["attention_mask"])
                text_features = mean_pooling(text_features, batch_sim["attention_mask"])
            elif args.concept_text_sim_model == 'simcse':
                text_features = sim_model(input_ids=batch_sim["input_ids"], attention_mask=batch_sim["attention_mask"], output_hidden_states=True, return_dict=True).pooler_output
            elif args.concept_text_sim_model == 'angle':
                text_features = sim_model(output_hidden_states=True, input_ids=batch_sim["input_ids"], attention_mask=batch_sim["attention_mask"]).hidden_states[-1][:, -1].float()
            else:
                raise Exception("concept-text sim model should be mpnet or angle")
            text_features = F.normalize(text_features, p=2, dim=1)
        val_sim.append(text_features @ concept_features.T)
    val_similarity = torch.cat(val_sim, dim=0).cpu().detach().numpy()

# ======================================= SAVE ACS LABELS =======================================
# Files are structured so the downstream training scripts can pick them up by convention:
#   ./<encoder>_acs/<dataset_key>/concept_labels_train.npy
#   ./<encoder>_acs/<dataset_key>/concept_labels_val.npy  (SST-2 only)
d_name = args.dataset.replace('/', '_')
prefix = "./"
if args.concept_text_sim_model == 'mpnet':
    prefix += "mpnet_acs"
elif args.concept_text_sim_model == 'simcse':
    prefix += "simcse_acs"
elif args.concept_text_sim_model == 'angle':
    prefix += "angle_acs"
prefix += "/"
prefix += d_name
prefix += "/"
if not os.path.exists(prefix):
    os.makedirs(prefix)

# These .npy matrices are the ACS labels consumed by train_CBL.py (Eq. 3) and then train_FL.py (Eq. 4).
np.save(prefix + "concept_labels_train.npy", train_similarity)
if args.dataset == 'SetFit/sst2':
    np.save(prefix + "concept_labels_val.npy", val_similarity)


# ======================================= What happens next? =======================================
# 1) train_CBL.py will:
#    - Load these ACS labels
#    - (Optionally) apply ACC to denoise them
#    - Train the Concept Bottleneck Layer so individual neurons align with interpretable concepts
#
# 2) train_FL.py will:
#    - Freeze/Load the trained CBL
#    - Train a sparse linear head to predict class labels from concept activations
#
# This is how CB-LLM achieves interpretability (neurons ↔ concepts) and preserves accuracy near black-box baselines.