# decode_sampling.py
import torch
from torch.utils.data import DataLoader, Subset
from transformers import PreTrainedTokenizer, PreTrainedModel, LogitsProcessor, LogitsProcessorList
import numpy as np
import random
import matplotlib.pyplot as plt
import os
from copy import copy
from functools import partial
from collections import defaultdict
from typing import List, Tuple

from data.dataset import collate_fn
from model.utils import _plot_attention_grid

def generate_input_context(bos_token, prefix, seed_tokens, seed_len, seen_seeds, conditional_prompt=""):
    """
    Generates a single input prompt with a unique seed, optionally followed by a conditional prompt.

    Args:
        bos_token (str): The beginning-of-sentence token.
        prefix (str): The task-specific prefix (e.g., "path").
        seed_tokens (list): The vocabulary of seed symbols.
        seed_len (int): The length of the seed sequence.
        seen_seeds (set): A set of seeds that have already been used.
        conditional_prompt (str, optional): The prompt to condition on. Defaults to "".

    Returns:
        str: The fully formatted input context for the model.
    """
    # Generate a novel seed that hasn't been seen in the training set
    novel_seed = ""
    if seed_len > 0:
        while True:
            h = "".join([random.choice(seed_tokens) for _ in range(seed_len)])
            if h not in seen_seeds:
                novel_seed = h
                seen_seeds.add(h)
                break
    
    # Combine the components: BOS, prefix, seed, and the conditional prompt
    parts = [bos_token + novel_seed + prefix, conditional_prompt]
    
    # Filter out any empty parts and join with spaces
    return " ".join(part for part in parts if part)

class VocabMaskLogitsProcessor(LogitsProcessor):
    def __init__(self, vocab_mask_ids):
        self.vocab_mask_ids = vocab_mask_ids

    def __call__(self, input_ids, scores):
        scores[:, self.vocab_mask_ids] = -float("inf")
        return scores
    
def decode_batch(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    input_contexts: List[str],
    max_length: int,
    vocab_mask_ids: List[int] = None,
    greedy: bool = True,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> Tuple[List[str], List[torch.Tensor]]:
    """
    input_contexts: List[str] — input context strings
    returns: Tuple[List[str], List[torch.Tensor]] — decoded completions, clipped at EOS, and their full token IDs (context + completion)
    """
    model.eval()

    encodings = tokenizer(
        input_contexts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    input_ids = encodings["input_ids"].to(model.device)
    attention_mask = encodings["attention_mask"].to(model.device)
    
    context_lengths = attention_mask.sum(dim=1)
    
    logits_processors = None
    if vocab_mask_ids:
        logits_processors = LogitsProcessorList()
        #logits_processors.append(VocabMaskLogitsProcessor(vocab_mask_ids))

    generation_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "do_sample": not greedy,
        "temperature": temperature,
        "top_k": 0,
        "top_p": top_p,
        "logits_processor": logits_processors,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "max_length": max_length,
    }

    with torch.no_grad():
        outputs = model.generate(**generation_kwargs)

    # "Normalize" outputs by removing any left padding
    # After this step, every sequence in `unpadded_outputs` starts with a real token.
    unpadded_outputs = []
    if tokenizer.padding_side == "left":
        # Find the number of padding tokens at the beginning of each sequence
        # The first '1' in the attention mask marks the start of the real tokens
        num_left_pads = torch.argmax(attention_mask.to(torch.int), dim=1)
        for i, seq in enumerate(outputs):
            unpadded_outputs.append(seq[num_left_pads[i]:])
    else:
        # If padding is on the right, outputs are already "normalized"
        unpadded_outputs = list(outputs)

    # Now `unpadded_outputs` is a list of tensors, each starting with the real tokens.
    # and `context_lengths` is the correct length of the context for each sequence.
    generated_only = [
        seq[context_len:] for seq, context_len in zip(unpadded_outputs, context_lengths)
    ]

    eos_id = tokenizer.eos_token_id
    clipped_generated = [
        seq[:(seq == eos_id).nonzero(as_tuple=True)[0][0]] if eos_id in seq else seq
        for seq in generated_only
    ]

    # The logic for `clipped_outputs` is now also simpler and more robust,
    # as it operates on the unpadded sequences.
    clipped_outputs = [
        unpadded_seq[:context_len + len(clipped_seq) + 1] # include EOS
        for unpadded_seq, context_len, clipped_seq in zip(unpadded_outputs, context_lengths, clipped_generated)
    ]

    return tokenizer.batch_decode(clipped_generated, skip_special_tokens=True), clipped_outputs, context_lengths

def generate_samples(
    model, 
    dataset, 
    tokenizer, 
    decode_fn, 
    greedy, 
    seed_tokens, 
    seed_len, 
    max_length, 
    prefix="", 
    temperature=1.0, 
    top_p=1.0, 
    batch_size=8,
    conditional_prompts=None,
    num_conditional_samples=None,
    num_samples=100
):
    """
    Generates samples from a model, supporting both conditional and unconditional modes.

    Args:
        model: The transformer model.
        dataset: The dataset object containing training seeds.
        tokenizer: The tokenizer.
        decode_fn: The function to call for model generation and decoding.
        greedy (bool): Whether to use greedy decoding.
        seed_tokens (list): The vocabulary of seed symbols.
        seed_len (int): The length of the seed sequence.
        max_length (int): The maximum number of tokens to generate.
        prefix (str, optional): The task-specific prefix. Defaults to "".
        temperature (float, optional): Sampling temperature. Defaults to 1.0.
        top_p (float, optional): Nucleus sampling p-value. Defaults to 1.0.
        batch_size (int, optional): The batch size for generation. Defaults to 8.
        conditional_prompts (list, optional): A list of strings to condition on. If None,
                                              operates in unconditional mode. Defaults to None.
        num_conditional_samples (list, optional): A list of integers specifying how many samples
                                                  to generate for each conditional_prompt.
                                                  Required if conditional_prompts is not None.
        num_samples (int, optional): The total number of samples to generate in unconditional
                                     mode. Defaults to 100.

    Returns:
        tuple: A tuple containing a list of all generated samples and a list of raw model outputs.
    """
    model.eval()
    seen_seeds = copy(dataset.train_seeds)
    BOS = tokenizer.bos_token
    
    all_generated_samples = []
    prompts_to_process = []
    
    # --- Determine the list of prompts to process ---
    if conditional_prompts is not None:
        # Conditional Mode: Unroll the prompts into a single list
        if num_conditional_samples is None or len(conditional_prompts) != len(num_conditional_samples):
            raise ValueError("In conditional mode, 'num_conditional_samples' must be a list of the same length as 'conditional_prompts'.")
        
        print(f"\nCollating conditional prompts for efficient batching.")
        for prompt, count in zip(conditional_prompts, num_conditional_samples):
            prompts_to_process.extend([prompt] * count)
    else:
        # Unconditional Mode: Create a list of empty prompts
        print(f"\nGenerating {num_samples} unconditional samples.")
        prompts_to_process = [""] * num_samples
    # --- Process the prompts in batches ---
    num_processed = 0
    all_outputs = []
    all_input_lengths = []
    while num_processed < len(prompts_to_process):
        current_batch_size = min(batch_size, len(prompts_to_process) - num_processed)
        
        # Get the current batch of base prompts (e.g., ["v1 v10", "v1 v10", "v5 v25"])
        prompt_batch = prompts_to_process[num_processed : num_processed + current_batch_size]
        
        # Create the full input prompts with seeds for the model
        batch_input_list = [
            generate_input_context(BOS, prefix, seed_tokens, seed_len, seen_seeds, conditional_prompt=p)
            for p in prompt_batch
        ]

        # print(batch_input_list)
        
        # Generate samples for the batch
        vocab_mask_ids = tokenizer.convert_tokens_to_ids([BOS] + seed_tokens + [prefix])
        decoded_samples, outputs, input_lengths = decode_fn(model, tokenizer, batch_input_list, max_length, vocab_mask_ids, greedy, temperature, top_p)
        
        all_generated_samples.extend(decoded_samples)
        num_processed += len(decoded_samples)
        all_outputs.extend(outputs)
        all_input_lengths.extend(input_lengths.tolist())

    all_outputs = simple_collate_function(all_outputs, tokenizer)

    return all_generated_samples, all_outputs, all_input_lengths

def compute_perplexity(model, dataset, tokenizer, batch_size=512):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    collate_with_tokenizer = partial(collate_fn, tokenizer=tokenizer)
    
    # Create the DataLoader
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        collate_fn=collate_with_tokenizer,
        shuffle=False  # No need to shuffle for evaluation
    )
    
    with torch.no_grad():
        # Iterate over the correctly padded batches from the DataLoader
        for batch in dataloader:
            
            # Move the entire batch of padded tensors to the device
            input_ids = batch["input_ids"].to(model.device)
            labels = batch["labels"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)
            
            # Pass attention_mask to the model, which is good practice
            outputs = model(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
            logits = outputs.logits
            
            # Your loss calculation logic is correct
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='sum')
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
            num_tokens = (labels != -100).sum().item()
            
            total_loss += loss.item()
            total_tokens += num_tokens
    
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss))
    return perplexity

def save_train_nll(
    model, 
    dataset, 
    tokenizer, 
    batch_size=32, 
    temperature=1.0,
    step=None,
    num_samples=None,
    nll_hist_path=None
):
    """
    Computes the per-sequence NLL for a given number of samples.

    Args:
        model: The model to evaluate.
        dataset: The full dataset.
        tokenizer: The tokenizer for the collate function.
        batch_size: The batch size for the DataLoader.
        num_samples: The number of samples to evaluate on. If None, uses the full dataset.
    """
    model.eval()
    
    # 1. Create a subset of the dataset if num_samples is specified
    if num_samples is not None:
        # Ensure we don't request more samples than are available
        num_samples = min(num_samples, len(dataset))
        eval_dataset = Subset(dataset, range(num_samples))
    else:
        eval_dataset = dataset
    
    neg_log_likelihoods = []

    collate_with_tokenizer = partial(collate_fn, tokenizer=tokenizer)
    dataloader = DataLoader(
        eval_dataset, 
        batch_size=batch_size, 
        collate_fn=collate_with_tokenizer,
        shuffle=False
    )
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(model.device)
            labels = batch["labels"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)
            
            outputs = model(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
            logits = outputs.logits / temperature
            
            # 1. Get per-token NLL by setting reduction to 'none'
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
            
            # The loss will have shape (batch_size * sequence_length)
            per_token_loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
            
            # 2. Reshape the loss to (batch_size, sequence_length)
            per_token_loss = per_token_loss.view(labels.shape)
            
            # 3. Sum the NLL for each sequence and divide by the true length
            # The label is -100 for padding and seed tokens, so we can use that to get the real loss
            per_sequence_loss = torch.where(labels != -100, per_token_loss, 0.0).sum(dim=1)
            num_tokens_per_sequence = (labels != -100).sum(dim=1)
            
            # Avoid division by zero for sequences that might be all padding
            # Although unlikely with truncation, this is a safe practice.
            num_tokens_per_sequence = torch.clamp(num_tokens_per_sequence, min=1)
            
            normalized_nll = per_sequence_loss / num_tokens_per_sequence
            
            # 4. Extend the list with the NLL for each sequence in the batch
            neg_log_likelihoods.extend(normalized_nll.tolist())
        
    if nll_hist_path is not None and step is not None:
        plt.figure(figsize=(10, 5))
        plt.hist(neg_log_likelihoods, bins=50, alpha=0.7, color='blue')
        plt.title('Distribution of Neg. Log Likelihoods @ Step {} and Temp {}'.format(step, temperature))
        plt.xlabel('Negative Log Likelihood')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.savefig(os.path.join(nll_hist_path, f"step_{step}_temp_{temperature}.png"))
        print(f"✅ NLL distribution saved at step {step}")
        plt.close()

    return 

def visualize_attention_weights(model, tokenizer, train_dataset, temperature, step, save_path,
                                num_samples=100, max_length=20, batch_size=16, for_all_layers=True,
                                creativity=0.0, memorization=0.0):
    """
    Generates sequences, gets attention weights, and plots dynamically-sized heatmaps
    for the last layer or all layers, organizing outputs by step.
    """
    model.eval()
    DEVICE = model.device
    print(f"\n--- Generating {num_samples} samples for attention analysis at step {step} ---")

    all_attentions_per_layer = []
    all_seq_lengths = []
    all_samples_text = []

    # --- Batch Processing Loop ---
    is_first_batch = True
    num_layers = 0
    for i in range(0, num_samples, batch_size):
        current_num_samples = min(batch_size, num_samples - i)
        print(f"Processing {current_num_samples} samples (Batch {i//batch_size + 1})")
        with torch.no_grad():
            samples, full_outputs, input_lengths = generate_samples(
                model=model, dataset=train_dataset, tokenizer=tokenizer, decode_fn=decode_batch,
                greedy=(temperature == 0.0), seed_tokens=train_dataset.seed_tokens,
                seed_len=train_dataset.seed_len, max_length=max_length, temperature=temperature,
                top_p=1.0, num_samples=current_num_samples, batch_size=current_num_samples,
                prefix=train_dataset.prefix
            )
        with torch.no_grad():
            rich_outputs = model(
                input_ids=full_outputs["input_ids"].to(DEVICE),
                attention_mask=full_outputs["attention_mask"].to(DEVICE),
                output_attentions=True
            )

        if is_first_batch:
            num_layers = len(rich_outputs.attentions)
            all_attentions_per_layer = [[] for _ in range(num_layers)]
            is_first_batch = False

        for layer_idx in range(num_layers):
            all_attentions_per_layer[layer_idx].append(rich_outputs.attentions[layer_idx].cpu())

        all_samples_text.extend(samples)
        all_seq_lengths.append(full_outputs["attention_mask"].sum(dim=1).cpu())

    if not all_seq_lengths:
        print("No samples were generated. Cannot visualize attention.")
        return

    seq_lengths = torch.cat(all_seq_lengths, dim=0).numpy()
    unique_lengths, counts = np.unique(seq_lengths, return_counts=True)
    most_common_len = int(unique_lengths[np.argmax(counts)])

    # --- Create a directory for the current step ---
    step_save_path = os.path.join(save_path, f"step_{step}")
    os.makedirs(step_save_path, exist_ok=True)
    print(f"\nSaving plots to directory: {step_save_path}")

    # --- Main Loop to Process and Plot Layers ---
    layers_to_process = range(num_layers) if for_all_layers else [num_layers - 1]

    for layer_idx in layers_to_process:
        print(f"\n--- Processing Attention for Layer {layer_idx} ---")
        all_attentions = all_attentions_per_layer[layer_idx]
        global_indices_to_average = np.where(seq_lengths == most_common_len)[0]
        tensors_to_average = []
        global_idx = 0
        for batch_attentions in all_attentions:
            batch_size_current = batch_attentions.shape[0]
            batch_lengths = seq_lengths[global_idx : global_idx + batch_size_current]
            local_indices_to_keep = np.where(batch_lengths == most_common_len)[0]
            if len(local_indices_to_keep) > 0:
                filtered_batch_tensors = batch_attentions[local_indices_to_keep, :, :most_common_len, :most_common_len]
                tensors_to_average.append(filtered_batch_tensors)
            global_idx += batch_size_current

        if not tensors_to_average:
            print(f"Could not find any samples for layer {layer_idx} with the most common length ({most_common_len}).")
            continue

        filtered_attentions = torch.cat(tensors_to_average, dim=0)
        num_indices_averaged = filtered_attentions.shape[0]
        num_heads = filtered_attentions.shape[1]

        print(f"Averaging attention maps for {num_indices_averaged} samples of length {most_common_len}.")
        if layer_idx == layers_to_process[0]:
            print("--- Example Samples Being Averaged ---")
            num_to_print = min(5, len(global_indices_to_average))
            for i in range(num_to_print):
                sample_idx = global_indices_to_average[i]
                print(f"[{i+1}] {repr(all_samples_text[sample_idx])}")
            print("------------------------------------")

        avg_per_head_map = filtered_attentions.mean(dim=0).float().numpy()
        overall_avg_map = avg_per_head_map.mean(axis=0)

        HL = train_dataset.seed_len
        input_tokens = ['<BOS>'] + [f"H{i+1}" for i in range(HL)]
        if train_dataset.prefix:
            input_tokens.append(train_dataset.prefix)
        num_input_tokens = len(input_tokens)
        num_generated_tokens = most_common_len - num_input_tokens
        readable_sequence = input_tokens + [f"T{i+1}" for i in range(num_generated_tokens)]

        title = (
            f"Layer {layer_idx} - Average Attention Heatmaps\n"
            f"(Step {step}, Temp {temperature:.2f}, Creativity {creativity:.2f}, Memorization {memorization:.2f})\n"
            f"({train_dataset.dataset_name}, {num_indices_averaged} samples)"
        )
        output_filename = os.path.join(step_save_path, f"layer{layer_idx}_attentions.png")

        _plot_attention_grid(
            avg_per_head_map=avg_per_head_map,
            overall_avg_map=overall_avg_map,
            readable_sequence=readable_sequence,
            num_heads=num_heads,
            num_input_tokens=num_input_tokens,
            title=title,
            save_path=output_filename
        )

def simple_collate_function(input_ids, tokenizer):
    """
    A simple, device-aware collate function that pads a list of input_ids 
    to the maximum length in the batch.
    
    Args:
        input_ids (list[torch.Tensor]): A list of 1D tensors representing tokenized sequences.
        tokenizer: The tokenizer instance, used to get the padding token ID.
    
    Returns:
        dict: A dictionary with padded 'input_ids' and 'attention_mask'.
    """
    # Handle the case of an empty list to avoid errors
    if not input_ids:
        return {
            "input_ids": torch.empty(0, 0, dtype=torch.long), 
            "attention_mask": torch.empty(0, 0, dtype=torch.long)
        }
        
    # Determine the maximum sequence length and the target device from the input
    max_length = max(ids.size(0) for ids in input_ids)
    device = input_ids[0].device  # Get the device from the first tensor
    
    pad_token_id = tokenizer.pad_token_id
    
    padded_input_ids = []
    attention_masks = []
    for ids in input_ids:
        padding_needed = max_length - ids.size(0)
        
        # Pad the input_ids
        padded_ids = torch.cat([
            ids,
            torch.tensor([pad_token_id] * padding_needed, dtype=torch.long, device=device)
        ])
        padded_input_ids.append(padded_ids)
        
        # Create the attention mask
        attention_mask = torch.cat([
            # --- FIX: Ensure these tensors are also on the correct device ---
            torch.ones(ids.size(0), dtype=torch.long, device=device),
            torch.zeros(padding_needed, dtype=torch.long, device=device)
        ])
        attention_masks.append(attention_mask)
        
    # Stack the lists of padded tensors into a final batch tensor
    batch = {
        "input_ids": torch.stack(padded_input_ids),
        "attention_mask": torch.stack(attention_masks),
    }
    return batch 