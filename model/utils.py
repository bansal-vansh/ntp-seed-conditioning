import json
import numpy as np
import wandb
import torch
import random
from transformers import GPT2Config, AutoTokenizer, GPT2LMHeadModel, AutoModelForCausalLM
import math
import matplotlib.pyplot as plt
import sys
sys.path.append("..")  # Adjust the path to import from the parent directory
from data.dataset import CustomTokenizer
from model.networks import AttentionOnlyLMHeadModel

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_gpt2_config(model_name, tokenizer=None, n_embed=None, n_layer=None, n_head=None):
    model_configs = {
        'gpt2': dict(n_embd=768,  n_layer=12, n_head=12),
        'gpt2-medium': dict(n_embd=1024, n_layer=24, n_head=16),
        'gpt2-large': dict(n_embd=1280, n_layer=36, n_head=20),
        'gpt2-xl': dict(n_embd=1600, n_layer=48, n_head=25),
    }

    if model_name not in model_configs:
        if n_embed is not None and n_layer is not None and n_head is not None:
            model_configs[model_name] = dict(n_embd=n_embed, n_layer=n_layer, n_head=n_head)
        else:
            raise ValueError(f"Please provide n_embed, n_layer, and n_head for custom model {model_name}")

    cfg = model_configs[model_name]

    return GPT2Config(
        vocab_size=len(tokenizer),  # Default GPT-2 vocab size
        n_positions=1024,
        n_ctx=1024,
        n_embd=cfg['n_embd'],
        n_layer=cfg['n_layer'],
        n_head=cfg['n_head'],
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

def get_attention_only_config(model_name, tokenizer=None, n_embed=None, n_layer=None, n_head=None):
    model_name = model_name.removesuffix("-pretrained")

    if model_name == 'attention-only-2':
        config = GPT2Config(
            vocab_size=tokenizer.vocab_size, n_positions=128, n_layer=2,
            n_head=4, n_embd=128, bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
        )
    elif model_name == 'attention-only-4':
        config = GPT2Config(
            vocab_size=tokenizer.vocab_size, n_positions=256, n_layer=4,
            n_head=8, n_embd=256, bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
        )
    elif model_name == 'attention-only-12':
        config = GPT2Config(
            vocab_size=tokenizer.vocab_size, n_positions=1024, n_layer=12,
            n_head=12, n_embd=768, bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
        )
    else:
        if n_embed is not None and n_layer is not None and n_head is not None:
            config = GPT2Config(
                vocab_size=tokenizer.vocab_size, n_positions=1024, n_layer=n_layer,
                n_head=n_head, n_embd=n_embed, bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
            )
        else:
            raise ValueError(f"Please provide n_embed, n_layer, and n_head for custom model {model_name}")
    return config


def get_tokenizer(args, DEVICE, custom=False):
    model_name = args.model_type.removesuffix("-pretrained")
    if model_name.startswith("attention-only") or custom:
        tokenizer = CustomTokenizer()
    elif model_name.startswith("gemma"):
        tokenizer = AutoTokenizer.from_pretrained(
            f"google/gemma-2b",
            cache_dir="/scratch/cluster/vansh/hf_cache",
            )
    elif model_name.startswith("gpt2"):
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
    else:
        raise ValueError(f"No tokenizer defined for model type {args.model_type}")
    
    return tokenizer

def get_model(args, tokenizer, DEVICE, n_embed=None, n_layer=None, n_head=None):
    model_name = args.model_type.removesuffix("-pretrained")
    if "gemma" in model_name:
        if "pretrained" in args.model_type:
            print(f"Loading pretrained {model_name} model")
            model = AutoModelForCausalLM.from_pretrained(
                f"google/{model_name}",
                torch_dtype=torch.bfloat16,
                # attn_implementation="eager",
                cache_dir="/scratch/cluster/vansh/hf_cache",
            ).to(DEVICE)
            model.resize_token_embeddings(len(tokenizer))
        else: 
            raise ValueError("Scratch models are not supported for Gemma models yet")
    
    elif "gpt2" in model_name:
        config = get_gpt2_config(model_name=model_name, 
                                 tokenizer=tokenizer, 
                                 n_embed=n_embed, 
                                 n_layer=n_layer, 
                                 n_head=n_head)
        if "pretrained" in args.model_type:
            print(f"Loading pretrained {args.model_type} model")
            model = GPT2LMHeadModel.from_pretrained(model_name)
            model.resize_token_embeddings(len(tokenizer))
        else:
            print(f"Initializing new {args.model_type} model")
            model = GPT2LMHeadModel(config=config)
            model.resize_token_embeddings(len(tokenizer))
    
    elif "attention-only" in model_name:
        if "pretrained" in args.model_type:
            raise ValueError("Pretrained models are not supported for attention-only models yet")
        print(f"Initializing new {args.model_type} model")
        config = get_attention_only_config(model_name=model_name, tokenizer=tokenizer, 
                                 n_embed=n_embed, 
                                 n_layer=n_layer, 
                                 n_head=n_head)
        model = AttentionOnlyLMHeadModel(config=config)
        model.resize_token_embeddings(len(tokenizer))

    else:
        raise ValueError(f"No model defined for model type {args.model_type}")
        
    return model
    
def setup_wandb(project, name, config):
    if wandb.run is not None:
        wandb.finish()
    wandb.init(project=project, name=name, config=config)

def save_results_json(results, path):
    with open(path, "w") as f:
        json.dump(results, f, indent=2)

def evaluate_model_checkpoint(model, step, temperatures, num_runs, decode_fn, train_dataset, eval_fn, base_seed=42):
    results = {}

    # 1. Define all evaluation settings to run
    # Format: (key_for_results, greedy_flag, temperature, label_suffix)
    eval_settings = [
        ("argmax", True, 0.0, "argmax")
    ]
    for temp in temperatures:
        eval_settings.append(
            (f"temp={temp}", False, temp, f"softmax @ temp={temp}")
        )
    best_creativity_temp = 0.0
    best_creativity_score = float('-inf')
    corresponding_memorization = 0.0

    # 2. Unified evaluation loop
    for key, greedy, temp, label_suffix in eval_settings:
        # Run the evaluation num_runs times for the current setting
        all_runs = np.array([
            eval_fn(model,
                    greedy=greedy,
                    temperature=temp,
                    label=f"Step {step} {label_suffix}",
                    decode_fn=decode_fn,
                    train_dataset=train_dataset,
                    seed=base_seed + i)
            for i in range(num_runs)
        ])

        # Calculate stats and convert to lists for JSON compatibility
        mean_run = np.mean(all_runs, axis=0).tolist()
        std_run = np.std(all_runs, axis=0).tolist()

        # If this is the best creativity score, update the best key
        if mean_run[1] > best_creativity_score:
            best_creativity_score = mean_run[1]
            best_creativity_temp = temp
            corresponding_memorization = mean_run[0]

        # Store results using the setting's key
        results[key] = {
            "rep_power_mean": mean_run[0],
            "rep_power_std": std_run[0],
            "creativity_mean": mean_run[1],
            "creativity_std": std_run[1],
            "uniqueness_mean": mean_run[2],
            "uniqueness_std": std_run[2],
            "coherence_mean": mean_run[3],
            "coherence_std": std_run[3],
        }

    return results, best_creativity_temp, best_creativity_score, corresponding_memorization

def log_results_to_wandb(step, results):
    wandb.log({
        "rep_power/argmax_mean": results["argmax"]["rep_power_mean"],
        "rep_power/argmax_std": results["argmax"]["rep_power_std"],
        "creativity/argmax_mean": results["argmax"]["creativity_mean"],
        "creativity/argmax_std": results["argmax"]["creativity_std"],
        "step": step,
    }, step=step)

    for k, v in results.items():
        if k.startswith("temp="):
            t = float(k.split("=")[-1])
            wandb.log({
                f"rep_power/softmax_mean/temp={t}": v["rep_power_mean"],
                f"rep_power/softmax_std/temp={t}": v["rep_power_std"],
                f"creativity/softmax_mean/temp={t}": v["creativity_mean"],
                f"creativity/softmax_std/temp={t}": v["creativity_std"],
                "temperature": t,
                "step": step,
            }, step=step)

def save_best_model_creativity(model, tokenizer, step_best_score, best_score, save_path):
    """
    Checks if the current model has a better score for a given metric and saves it.

    Args:
        model: The model object to potentially save.
        tokenizer: The tokenizer to save with the model.
        step_best_score (float): The score of the current model at this step.
        best_score (float): The best score seen so far for this metric.
        save_path (str): The directory path to save the best model.

    Returns:
        float: The updated best score for the metric.
    """

    if step_best_score > best_score:
        print(f"New best creativity: {step_best_score:.4f} (previously {best_score:.4f})")
        
        if save_path:
            print(f"Saving new best model to {save_path}")
            model.save_pretrained(save_path, safe_serialization=False)
            if tokenizer:
                tokenizer.save_pretrained(save_path)
        
        return step_best_score  # Return the new best score
    
    return best_score  # Return the old score if not improved


def _plot_attention_grid(avg_per_head_map, overall_avg_map, readable_sequence, num_heads, num_input_tokens, title, save_path):
    """Helper function to handle the matplotlib plotting logic."""
    seq_len = len(readable_sequence)
    
    # --- 1. Dynamic Tick & Font Size Adjustment ---
    # Determine how many tick labels to skip to avoid overlap
    tick_step = max(1, seq_len // 20)  # Try to show a max of ~20 labels
    dynamic_font_size = max(6, 14 - seq_len // 6) # Decrease font size for longer sequences
    print(f"Using tick step {tick_step} and font size {dynamic_font_size} for sequence length {seq_len}.")

    tick_locs = np.arange(seq_len)[::tick_step]
    tick_labels = [readable_sequence[i] for i in tick_locs]

    # --- 2. Enhanced Aesthetics ---
    plt.style.use('seaborn-v0_8-deep') # Use a professional and pleasant style
    
    n_plots = num_heads + 1
    ncols = int(math.ceil(math.sqrt(n_plots)))
    nrows = int(math.ceil(n_plots / ncols))
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 5), constrained_layout=True)
    axes = axes.flatten()

    vmax = np.max(avg_per_head_map) # Use a consistent color scale across all plots
    
    for i in range(num_heads):
        ax = axes[i]
        # Use a prettier, more vibrant colormap like 'magma' or 'plasma'
        im = ax.imshow(avg_per_head_map[i], cmap='magma', vmin=0, vmax=vmax)

        # Add a subtle grid to delineate cells
        ax.set_xticks(np.arange(seq_len+1)-.5, minor=True)
        ax.set_yticks(np.arange(seq_len+1)-.5, minor=True)
        ax.grid(which="minor", color="black", linestyle='-', linewidth=0.5)
        ax.tick_params(which="minor", size=0)

        ax.set_title(f"Head {i}", fontsize=dynamic_font_size + 2)
        ax.set_xticks(tick_locs)
        ax.set_yticks(tick_locs)
        ax.set_xticklabels(tick_labels, rotation=90, fontsize=dynamic_font_size)
        ax.set_yticklabels(tick_labels, fontsize=dynamic_font_size)
        
        # --- 3. Add Context Separator Line ---
        # Draw a line to separate prompt/prefix from generated tokens
        separator_pos = num_input_tokens - 0.5
        ax.axvline(separator_pos, color='white', linestyle='--', linewidth=1.2, alpha=0.8)
        ax.axhline(separator_pos, color='white', linestyle='--', linewidth=1.2, alpha=0.8)

    # Plot the overall average
    ax = axes[num_heads]
    ax.imshow(overall_avg_map, cmap='magma', vmin=0, vmax=vmax)
    ax.set_title("Average of All Heads", fontsize=dynamic_font_size + 2)
    ax.set_xticks(tick_locs)
    ax.set_yticks(tick_locs)
    ax.set_xticklabels(tick_labels, rotation=90, fontsize=dynamic_font_size)
    ax.set_yticklabels(tick_labels, fontsize=dynamic_font_size)
    ax.axvline(num_input_tokens - 0.5, color='white', linestyle='--', linewidth=1.2, alpha=0.8)
    ax.axhline(num_input_tokens - 0.5, color='white', linestyle='--', linewidth=1.2, alpha=0.8)
    ax.set_xticks(np.arange(seq_len+1)-.5, minor=True)
    ax.set_yticks(np.arange(seq_len+1)-.5, minor=True)
    ax.grid(which="minor", color="black", linestyle='-', linewidth=0.5)
    ax.tick_params(which="minor", size=0)

    # Turn off unused subplots
    for i in range(n_plots, len(axes)):
        axes[i].axis('off')

    fig.suptitle(title, fontsize=20, weight='bold')
    
    # Add a single, shared colorbar
    fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.05, pad=0.05, label="Attention Weight")
    
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"\nSaved grid of attention heatmaps to {save_path}")

