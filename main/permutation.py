import argparse
import torch
import numpy as np

from data.permutation import make_dataset, compute_permutation
from model.eval import decode_batch, generate_samples, compute_perplexity
from model.train import train_main
from model.utils import set_seed, get_tokenizer, get_model
from data.dataset import collate_fn

def parse_args():
    parser = argparse.ArgumentParser(description="Train GPT model on sibling-parent task")
    parser.add_argument('--N', type=int, default=9, help='Number of nodes in the permutation')
    parser.add_argument('--H', type=int, default=26, help='Seed vocabulary size')
    parser.add_argument('--HL', type=int, default=5, help='Seed length')
    parser.add_argument('--num_train_samples', type=int, default=50000, help='Number of training samples')
    parser.add_argument('--num_eval_samples', type=int, default=1000, help='Number of evaluation samples')
    parser.add_argument('--data_root', type=str, default='/datastor1/vansh/lang_sampling/data', help='Root directory for storing the data')
    parser.add_argument('--regenerate_data', action='store_true', help='Data regeneration')
    parser.add_argument('--model_type', type=str, default='gpt2', help='Model type')
    parser.add_argument('--top_p', type=float, default=0.9, help='Top p for nucleus sampling')
    parser.add_argument('--epochs', type=int, default=15, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate for training')
    parser.add_argument('--batch_size', type=int, default=16, help='Training batch size')
    parser.add_argument('--eval_batch_size', type=int, default=64, help='Evaluation batch size')
    parser.add_argument('--device', type=int, default=0, help='CUDA device ID')
    parser.add_argument('--save_name', type=str, default='experiment', help='Experiment name')
    parser.add_argument('--dataset_name', type=str, default='circle', help='Dataset name')
    parser.add_argument('--eval_runs', type=int, default=5, help='Number of evaluation runs per checkpoint')
    parser.add_argument('--num_ckpts', type=int, default=100, help='Number of checkpoints')
    return parser.parse_args()

def main():
    args = parse_args()

    DEVICE = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    tokenizer = get_tokenizer(args, DEVICE)

    train_dataset, test_dataset, tokenizer, train_strs, train_perms, VOCAB, SEED_TOKENS, EVAL_TOKENIZER = make_dataset(
                                                                                                                        N=args.N,
                                                                                                                        H=args.H,
                                                                                                                        seed_len=args.HL,
                                                                                                                        num_train_samples=args.num_train_samples,
                                                                                                                        num_test_samples=args.num_eval_samples,
                                                                                                                        tokenizer=tokenizer,
                                                                                                                        regenerate=args.regenerate_data,
                                                                                                                        data_root=args.data_root
                                                                                                                    )
    VOCAB_IDs = {EVAL_TOKENIZER._convert_token_to_id(tok) for tok in VOCAB}
    data_collator = lambda features: collate_fn(features, tokenizer=tokenizer)

    model = get_model(args, tokenizer, DEVICE)
    model.resize_token_embeddings(len(tokenizer))

    train_perms_set = set(train_perms)
    print(f"Training permutations set size: {len(train_perms_set)}")

    def is_coherent(seq, N, vocab_ids=VOCAB_IDs) -> bool:
        """
        Given that π exists (walk succeeded), check coherence:
        - correct number of edges
        - correct tokens
        """
        if len(seq) != N:
            return False

        vocab_set = set(vocab_ids)
        if any(token not in vocab_set for token in seq):
            return False
        
        for token in seq:
            if seq.count(token) != 1:
                return False

        return True

    def evaluate_model(model, 
                   greedy, 
                   temperature, 
                   label, 
                   decode_fn, 
                   train_dataset, 
                   seed=42, 
                   batch_size=args.eval_batch_size, 
                   num_eval_samples=args.num_eval_samples,
                   top_p=args.top_p):
        set_seed(seed)
        print(f"\nEvaluating {label} for seed {seed}...")

        samples = generate_samples(model, 
                                   train_dataset, 
                                   tokenizer,
                                   decode_fn=decode_fn, 
                                   greedy=greedy, 
                                   seed_tokens=SEED_TOKENS, 
                                   seed_len=args.HL, 
                                   max_length=train_dataset[1]["labels"].shape[0]*2,
                                   temperature=temperature, 
                                   top_p=top_p, 
                                   num_samples=num_eval_samples, 
                                   batch_size=batch_size
                                   )
        tokenized_samples = [EVAL_TOKENIZER.encode(s) for s in samples]

        unique_perms = set()
        unique_coherent_perms = set()
        num_coherent = 0
        incoherent_samples = []
        # Final metrics loop
        for s in tokenized_samples:
            pi = compute_permutation(s, vocab_ids=VOCAB_IDs)
            if pi is None:
                incoherent_samples.append(s)
                continue
            unique_perms.add(pi)
            if is_coherent(s, N=args.N, vocab_ids=VOCAB_IDs):
                num_coherent += 1
                unique_coherent_perms.add(pi)
            else:
                incoherent_samples.append(s)

        num_memorized = len([s for s in unique_coherent_perms if s in train_perms_set])
        num_creative = len([s for s in unique_coherent_perms if s not in train_perms_set])
        num_unique = len(unique_perms)

        representation_power = (num_memorized / len(samples))
        creativity = (num_creative / len(samples)) 
        uniqueness = (num_unique / len(samples)) 
        coherence = (num_coherent / len(samples)) 
        # perplexity = compute_perplexity(model, test_dataset, batch_size=batch_size)

        # print(f"Perplexity: {perplexity:.4f}")
        print(f"Coherence: {coherence:.4f} ({num_coherent}/{len(samples)})")
        print(f"Representation power: {representation_power:.4f} ({num_memorized}/{len(samples)})")
        print(f"Creativity: {creativity:.4f} ({num_creative}/{len(samples)})")
        print(f"Uniqueness: {uniqueness:.4f} ({num_unique}/{len(samples)})")

        return np.array([
            representation_power,
            creativity,
            uniqueness,
            coherence,
            len(samples)
        ])

    train_main(
        model=model,
        dataset_name=args.dataset_name,
        save_name=args.save_name,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        temperatures=[0.3, 0.5, 0.7, 1.0, 2.0],
        num_eval_runs=args.eval_runs,
        train_dataset=train_dataset,
        data_collator=data_collator,
        device=DEVICE,
        decode_fn=decode_batch,
        eval_fn=evaluate_model,
        num_checkpoints=args.num_ckpts,
        lr=args.learning_rate
    )


if __name__ == "__main__":
    main()
