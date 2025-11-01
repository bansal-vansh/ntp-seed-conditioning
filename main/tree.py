# train.py
import argparse
import torch
from transformers import GPT2Config
import numpy as np
from copy import copy

from data_tree import make_dataset
from sampler import decode_batch, generate_prefix
from trainer import train_main
from utils import set_seed

def parse_args():
    parser = argparse.ArgumentParser(description="Train GPT model on sibling-parent task")
    parser.add_argument('--K', type=int, default=5, help='Number of children per parent')
    parser.add_argument('--L', type=int, default=4, help='Depth of the tree')
    parser.add_argument('--H', type=int, default=26, help='Seed vocabulary size')
    parser.add_argument('--HL', type=int, default=5, help='Seed length')
    parser.add_argument('--loops', action='store_true', help='Allow repeated vertex tokens (loops)')
    parser.add_argument('--num_train_samples', type=int, default=None, help='Number of training samples')
    parser.add_argument('--num_eval_samples', type=int, default=1000, help='Number of evaluation samples')
    parser.add_argument('--epochs', type=int, default=15, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Training batch size')
    parser.add_argument('--device', type=int, default=0, help='CUDA device ID')
    parser.add_argument('--save_name', type=str, default='experiment', help='Experiment name')
    parser.add_argument('--eval_runs', type=int, default=10, help='Number of evaluation runs per checkpoint')
    return parser.parse_args()

def main():
    args = parse_args()
    dataset, train_strs, graph, BOS_ID, SEED_TOKENS, VOCAB_SIZE = make_dataset(
        K=args.K,
        L=args.L,
        H=args.H,
        seed_len=args.HL,
        unique=not args.loops,
        num_samples=args.num_train_samples,
        seed=42,
    )

    train_set = set(tuple(s) for s in train_strs)
    
    def is_memorized(sample):
        return tuple(sample) in train_set

    def is_coherent(seq, tree):
        if len(seq) != args.L:
            return False

        prefix = []
        for token in seq:
            children = tree.get(tuple(prefix), None)
            if children is None or token not in children:
                return False
            prefix.append(token)
        return True

    def evaluate_model(model, greedy, temperature, label, decode_fn, dataset, seed=42, batch_size=64, num_eval_samples=args.num_eval_samples):
        set_seed(seed)
        print(f"\nEvaluating {label} for seed {seed}...")
        model.eval()

        samples = []
        seen_seeds = copy(dataset.train_seeds)
        device = model.device

        num_batches = (num_eval_samples + batch_size - 1) // batch_size

        for _ in range(num_batches):
            # Generate prefixes of consistent length
            prefix_list = [
                generate_prefix(BOS_ID, SEED_TOKENS, seed_len=args.HL, seen_seeds=seen_seeds)
                for _ in range(batch_size)
            ]

            # Stack into (batch_size, prefix_len)
            batch_input = torch.cat(prefix_list, dim=0).to(device)

            # Call the passed decode_fn (must support batched input)
            decoded = decode_fn(
                model=model,
                prefixes=batch_input,
                L=args.L,
                vocab_mask_ids=[BOS_ID] + SEED_TOKENS,
                greedy=greedy,
                temperature=temperature
            )

            samples.extend(decoded[:num_eval_samples - len(samples)])  # Crop overflow

        unique_samples = set(samples)
        representation_power = len([s for s in unique_samples if is_memorized(s)]) / len(samples)
        creativity = len([s for s in unique_samples if is_coherent(s, graph) and not is_memorized(s)]) / len(samples)
        uniqueness = len(unique_samples) / len(samples)

        return np.array([representation_power, creativity, uniqueness, len(samples)])

    config = GPT2Config(
        vocab_size=VOCAB_SIZE,
        n_positions=1024,
        n_ctx=1024,
        n_embd=768,
        n_layer=12,
        n_head=12,
        bos_token_id=BOS_ID,
        eos_token_id=None,
    )

    DEVICE = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    train_main(
        save_name=args.save_name,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        temperatures=[0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0],
        num_eval_runs=args.eval_runs,
        train_dataset=dataset,
        config=config,
        device=DEVICE,
        decode_fn=decode_batch,
        eval_fn=evaluate_model,
        every_n_epochs=args.evaluate_after_every,
    )

if __name__ == "__main__":
    main()
