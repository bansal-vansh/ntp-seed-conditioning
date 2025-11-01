# %%
import torch
import numpy as np
import os, sys
device = 1
os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

from data.dataset import CustomTokenizer
from transformers import GPT2Config
from data.circle import make_dataset, compute_canonical_permutation
from model.networks import AttentionOnlyLMHeadModel
from model.eval import decode_batch, generate_samples
from model.train import train_main
from model.utils import set_seed
from data.dataset import collate_fn

# %%
DEVICE = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")

# define dataset parameters
dataset_name = "circle"
model_type = "attention-only-12"
model_path = None

M = 15
N = 6
H = 26
HL = 5
seed_per_pi = False
num_train_samples =10000
batch_size = 32
num_eval_samples = 1000
eval_batch_size = 256
num_ckpts = 20
epochs = 30
eval_runs = 1
top_p=1.0
save_name = f"M{M}-N{N}-H{H}-NT{num_train_samples}-E{epochs}-top_p{top_p}-{model_type}"
data_root = f"/datastor1/vansh/lang_sampling/data"
regenerate_data = False

# %%
tokenizer = CustomTokenizer(padding_side='left')
# tokenizer.padding_side = 'left'  # Ensure left-padding for decoder-only architecture
train_dataset, test_dataset, tokenizer, train_strs, train_perms, VOCAB, SEED_TOKENS, EVAL_TOKENIZER = make_dataset(
                                                                                                                    M=M,
                                                                                                                    N=N,
                                                                                                                    H=H,
                                                                                                                    seed_len=HL,
                                                                                                                    seed_per_pi=seed_per_pi,
                                                                                                                    num_train_samples=num_train_samples,
                                                                                                                    num_test_samples=num_eval_samples,
                                                                                                                    tokenizer=tokenizer,
                                                                                                                    data_root=data_root,
                                                                                                                    regenerate=regenerate_data
                                                                                                                )
VOCAB_IDs = {EVAL_TOKENIZER._convert_token_to_id(tok) for tok in VOCAB}
data_collator = lambda features: collate_fn(features, tokenizer=tokenizer)

# %%
tokenizer.get_vocab()

# %%
train_dataset[0] 

# %%
collate_fn([train_dataset[0], train_dataset[3]], tokenizer=tokenizer)

# %%
model_name = model_type.removesuffix("-pretrained")

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
    raise ValueError(f"No configuration defined for model type {model_type}")

# 3. Instantiate the model from scratch or load from a path
if "pretrained" in model_type:
    if not model_path:
        raise ValueError("Must provide 'model_path' in args when using a pretrained model.")
    print(f"Loading pretrained {model_name} model from {model_path}")
    model = AttentionOnlyLMHeadModel.from_pretrained(model_path)
else:
    print(f"Initializing new {model_name} model from scratch")
    model = AttentionOnlyLMHeadModel(config=config)

model.to(DEVICE)
model.resize_token_embeddings(len(tokenizer))
print(f"Model {model_name} initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters.")

# %%
train_perms_set = set(train_perms)
print(f"Training permutations set size: {len(train_perms_set)}")

def is_coherent_after_walk(seq, N, vocab_ids=VOCAB_IDs) -> bool:
    """
    Given that π exists (walk succeeded), check coherence:
    - correct number of edges
    - correct tokens
    """
    if len(seq) != 2 * N:
        return False

    vocab_set = set(vocab_ids)
    if any(token not in vocab_set for token in seq):
        return False

    return True

def evaluate_model(model, 
                greedy, 
                temperature, 
                label, 
                decode_fn, 
                train_dataset, 
                seed=42, 
                batch_size=eval_batch_size, 
                num_eval_samples=num_eval_samples,
                top_p=top_p):
    set_seed(seed)
    print(f"\nEvaluating {label} for seed {seed}...")

    samples, outputs = generate_samples(model, 
                                train_dataset, 
                                tokenizer,
                                decode_fn=decode_fn, 
                                greedy=greedy, 
                                seed_tokens=SEED_TOKENS, 
                                seed_len=HL, 
                                max_length=train_dataset[1]["labels"].shape[0],
                                temperature=temperature, 
                                top_p=top_p, 
                                num_samples=num_eval_samples, 
                                batch_size=batch_size
                                )
    print(f"Generated {len(samples)} samples.")
    tokenized_samples = [EVAL_TOKENIZER.encode(s) for s in samples]

    unique_perms = set()
    unique_coherent_perms = set()
    num_coherent = 0
    incoherent_samples = []
    # Final metrics loop
    for s in tokenized_samples:
        pi = compute_canonical_permutation(s, N=N)
        if pi is not None:
            unique_perms.add(pi)
            if is_coherent_after_walk(s, N=N, vocab_ids=VOCAB_IDs):
                num_coherent += 1
                unique_coherent_perms.add(pi)
            else:
                incoherent_samples.append(s)
        else:
            incoherent_samples.append(s)

    num_unique_samples = len(set([tuple(s) for s in tokenized_samples])) 

    num_memorized = len([s for s in unique_coherent_perms if s in train_perms_set])
    num_creative = len([s for s in unique_coherent_perms if s not in train_perms_set])
    num_unique_perms = len(unique_perms)

    representation_power = (num_memorized / len(samples))
    creativity = (num_creative / len(samples)) 
    uniqueness = (num_unique_perms / len(samples)) 
    coherence = (num_coherent / len(samples)) 

    uniqueness_strings = (num_unique_samples / len(samples)) 
    # perplexity = compute_perplexity(model, test_dataset, batch_size=batch_size)

    # print(f"Perplexity: {perplexity:.4f}")
    print(f"Coherence: {coherence:.4f} ({num_coherent}/{len(samples)})")
    print(f"Representation power: {representation_power:.4f} ({num_memorized}/{len(samples)})")
    print(f"Creativity: {creativity:.4f} ({num_creative}/{len(samples)})")
    print(f"Uniqueness (permutations): {uniqueness:.4f} ({num_unique_perms}/{len(samples)})")
    print(f"Uniqueness (strings): {uniqueness_strings:.4f} ({num_unique_samples}/{len(samples)})")

    return np.array([
        representation_power,
        creativity,
        uniqueness,
        coherence,
        len(samples)
    ])

# %%
train_main(
    model=model,
    dataset_name=dataset_name,
    save_name=save_name,
    hl=HL,
    batch_size=batch_size,
    num_epochs=epochs,
    temperatures=[0.3, 0.5, 0.7, 1.0, 2.0],
    num_eval_runs=eval_runs,
    train_dataset=train_dataset,
    data_collator=data_collator,
    device=DEVICE,
    decode_fn=decode_batch,
    eval_fn=evaluate_model,
    num_checkpoints=num_ckpts,
    log_to_wandb=False,
    save_results=True, 
    lr=1e-4
)
# %%
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm
model_path = os.path.join("/datastor1/vansh/lang_sampling/results/", dataset_name, save_name, "best_models", f"HL{HL}.pt")
# Load the model. It will automatically find and use the .safetensors file.
model = AttentionOnlyLMHeadModel.from_pretrained(model_path).to(DEVICE)
# You can also load the tokenizer from the same directory
tokenizer = CustomTokenizer.from_pretrained(model_path)

# %%
def visualize_average_attention(model, tokenizer, num_samples=1000, max_length=20):
    """
    Generates multiple sequences, averages their last-layer attention maps,
    and plots the result.
    """
    model.eval()
    attention_maps_by_len = defaultdict(list)
    
    print(f"\n--- Generating {num_samples} samples for attention analysis ---")
    
    # 1. Generate one batch of samples efficiently.
    with torch.no_grad():
        # Assuming generate_samples can take num_samples and handle batching internally
        samples, output_ids = generate_samples(
            model, 
            train_dataset, 
            tokenizer,
            decode_fn=decode_batch, 
            greedy=False, 
            seed_tokens=SEED_TOKENS, 
            seed_len=HL, 
            max_length=max_length,
            temperature=1.0, 
            top_p=top_p, 
            num_samples=num_samples, 
            batch_size=batch_size
        )
        len(samples)  # This should be equal to num_samples

        print(f"Generated {len(samples)} samples.")
        print(f"Output IDs shape: {output_ids.shape}")
        # 2. Perform a single forward pass on the generated IDs to get attentions.
        rich_outputs = model(input_ids=output_ids, output_attentions=True)
        attentions = rich_outputs.attentions # This is a tuple of tensors, one per layer

    # 3. Get the attention tensor from the LAST layer. Its shape is (batch, seq, seq).
    last_layer_attentions = attentions[-1]

    print(f"Processing attentions for a batch of {last_layer_attentions.shape[0]} samples.")

    # 4. Loop through the batch and collect each attention map.
    for i in range(last_layer_attentions.shape[0]):
        attention_map = last_layer_attentions[i].cpu().numpy()
        seq_len = attention_map.shape[0]
        attention_maps_by_len[seq_len].append(attention_map)

    # --- END OF CORRECTED LOGIC ---

    # Find the most common sequence length to average over
    if not attention_maps_by_len:
        print("No samples generated. Cannot visualize attention.")
        return
        
    most_common_len = max(attention_maps_by_len, key=lambda k: len(attention_maps_by_len[k]))
    maps_to_average = attention_maps_by_len[most_common_len]
    
    print(f"Averaging attention maps for {len(maps_to_average)} samples of length {most_common_len}.")
    
    # Average the attention maps
    average_attention_map = np.mean(maps_to_average, axis=0)
    
    # --- Visualization ---
    readable_sequence = ['<BOS>'] + [f'T{i+1}' for i in range(most_common_len - 1)]
    
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(average_attention_map, cmap='viridis')

    ax.set_xticks(np.arange(len(readable_sequence)))
    ax.set_yticks(np.arange(len(readable_sequence)))
    
    ax.set_xticklabels(readable_sequence)
    ax.set_yticklabels(readable_sequence)
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    ax.set_xlabel("Key Position (Attending To)")
    ax.set_ylabel("Query Position (Generating From)")
    ax.set_title(f"Average Attention Heatmap (Last Layer) over {len(maps_to_average)} samples")
    fig.colorbar(im)
    fig.tight_layout()
    
    plt.savefig(f"average_attention_heatmap_HL{HL}.png")
    print("\nSaved average attention heatmap to average_attention_heatmap.png")


# %%
visualize_average_attention(model, tokenizer, num_samples=1000, max_length=train_dataset[1]["labels"].shape[0])