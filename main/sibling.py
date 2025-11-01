import argparse
import torch
from transformers import AutoTokenizer, GPT2LMHeadModel
import numpy as np
from copy import copy

from data.dataset import collate_fn
from data.sibling import make_dataset
from model.eval import decode_batch, generate_samples
from model.train import train_main
from model.utils import set_seed, get_model, get_tokenizer

def parse_args():
    parser = argparse.ArgumentParser(description="Train GPT model on sibling-parent task")
    parser.add_argument('--P', type=int, default=5, help='Number of parent tokens')
    parser.add_argument('--C', type=int, default=2500, help='Number of child tokens')
    parser.add_argument('--L', type=int, default=3, help='Sequence length')
    parser.add_argument('--H', type=int, default=26, help='Seed vocabulary size')
    parser.add_argument('--HL', type=int, default=5, help='Seed length')
    parser.add_argument('--prob', type=float, default=0.2, help='Edge probability')
    parser.add_argument('--top_p', type=float, default=0.9, help='Top p for nucleus sampling')
    parser.add_argument('--random', action='store_true', help='Use random ER graph instead of fixed')
    parser.add_argument('--no_planning', action='store_true', help='Disable planning')
    parser.add_argument('--regenerate_data', action='store_true', help='Data regeneration')
    parser.add_argument('--add_new_tokens', action='store_true', help='Add nodes as new tokens')
    parser.add_argument('--num_train_samples', type=int, default=50000, help='Number of training samples')
    parser.add_argument('--num_eval_samples', type=int, default=1000, help='Number of evaluation samples')
    parser.add_argument('--epochs', type=int, default=15, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--eval_batch_size', type=int, default=32, help='Evaluation batch size')
    parser.add_argument('--device', type=int, default=0, help='CUDA device ID')
    parser.add_argument('--use_pretrained', action='store_true', help='Use pretrained model')
    parser.add_argument('--pretrain_adjacency', action='store_true', help='Pretrain on graph adjacency bigrams')
    parser.add_argument('--model_type', type=str, default='gpt2', help='Model size')
    parser.add_argument('--save_name', type=str, default='experiment', help='Experiment name')
    parser.add_argument('--data_root', type=str, default='/datastor1/vansh/lang_sampling/data', help='Root directory for storing the data')
    parser.add_argument('--dataset_name', type=str, default='sibling', help='Dataset name')
    parser.add_argument('--eval_runs', type=int, default=5, help='Number of evaluation runs per checkpoint')
    parser.add_argument('--num_ckpts', type=int, default=100, help='Number of checkpoints')
    parser.add_argument('--seed', type=int, default=100, help='Seed for reproducibility')
    
    parser.add_argument('--custom_tokenizer', action='store_true', help='Use custom tokenizer')
    parser.add_argument('--n_embed', type=int, default=768, help='Number of embedding dimensions (for custom models)')
    parser.add_argument('--n_layer', type=int, default=4, help='Number of layers (for custom models)')
    parser.add_argument('--n_head', type=int, default=3, help='Number of attention heads (for custom models)')

    return parser.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)

    DEVICE = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    tokenizer = get_tokenizer(args, DEVICE, custom=args.custom_tokenizer)
    
    def canonicalize(triple):
        if len(triple) != 3:
            return tuple([-1, -1, -1])
        
        if not args.no_planning:
            p, a, b = triple
            return tuple([p] + sorted([a, b]))
        else:
            a, b, p = triple
            return tuple(sorted([a, b]) + [p])
        
    train_dataset, test_dataset, pretrain_dataset, train_strs, tokenized_graph, SEED_TOKENS, EVAL_TOKENIZER, prefixes = make_dataset(P=args.P,
                                                                                                                                    C=args.C,
                                                                                                                                    H=args.H,
                                                                                                                                    seed_len=args.HL,
                                                                                                                                    edge_prob=args.prob,
                                                                                                                                    pretrain_adjacency=args.pretrain_adjacency,
                                                                                                                                    planning=not args.no_planning,
                                                                                                                                    fixed=not args.random,
                                                                                                                                    num_train_samples=args.num_train_samples,
                                                                                                                                    num_test_samples=args.num_eval_samples,
                                                                                                                                    tokenizer=tokenizer,
                                                                                                                                    data_root=args.data_root,
                                                                                                                                    regenerate=args.regenerate_data,
                                                                                                                                    add_new_tokens=args.add_new_tokens)
    data_collator = lambda features: collate_fn(features, tokenizer=tokenizer)
    model = get_model(args, tokenizer, DEVICE, n_embed=args.n_embed, n_layer=args.n_layer, n_head=args.n_head)
    model.resize_token_embeddings(len(tokenizer))
    
    if args.pretrain_adjacency:
        print("Pretraining on graph adjacency edges...")
        model = train_main(
            model=model,
            dataset_name=args.dataset_name+"-pretrain",
            save_name=args.save_name,
            hl=0,
            batch_size=128,
            num_epochs=5,  # Shorter pretraining
            temperatures=[],
            num_eval_runs=0,
            train_dataset=pretrain_dataset,
            data_collator=data_collator,
            device=DEVICE,
            decode_fn=None,
            eval_fn=None,  # No evaluation during pretraining
            num_checkpoints=1,
            save_results=False,
            num_workers=16
        )

    train_set = set(canonicalize(EVAL_TOKENIZER.encode(s)) for s in train_strs)
    print(f"Training set size: {len(train_set)}")
    
    def is_memorized(seq):
        return canonicalize(seq) in train_set

    def is_coherent(seq, graph):
        if len(seq) != args.L:
            return False
        if args.no_planning:
            p, a, b = seq
        else:
            a, b, p = seq
        children = graph.get(p, [])
        return (a in children) and (b in children) and (a != b)

    def evaluate_model(model, 
                       greedy, 
                       temperature, 
                       label, 
                       decode_fn, 
                       train_dataset, 
                       prefix=prefixes["train"],
                       seed=args.seed, 
                       batch_size=args.eval_batch_size, 
                       num_eval_samples=args.num_eval_samples,
                       top_p=args.top_p,):
        set_seed(seed)
        print(f"\nEvaluating {label} for seed {seed}...")
        
        samples, outputs, input_lengths = generate_samples(model, 
                                                        train_dataset, 
                                                        tokenizer,
                                                        decode_fn=decode_fn, 
                                                        greedy=greedy, 
                                                        seed_tokens=SEED_TOKENS, 
                                                        seed_len=args.HL, 
                                                        prefix=prefix,
                                                        max_length=train_dataset[0]["labels"].shape[0]*2,
                                                        temperature=temperature, 
                                                        top_p=top_p, 
                                                        num_samples=num_eval_samples, 
                                                        batch_size=batch_size
                                                        )
         
        print("Generated samples:")
        for s in samples[:5]:
            print(s)
            
        tokenized_samples = [EVAL_TOKENIZER.encode(s) for s in samples]
        unique_samples = set([tuple(s) for s in tokenized_samples])

        num_memorized = len([s for s in unique_samples if is_memorized(s)])
        num_creative = len([s for s in unique_samples if is_coherent(s, tokenized_graph) and not is_memorized(s)])
        num_unique = len(unique_samples)
        num_coherent = len([s for s in tokenized_samples if is_coherent(s, tokenized_graph)])

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
        num_workers=16,
        eval_batch_size=args.eval_batch_size,
        lr=args.learning_rate,
    )

if __name__ == "__main__":
    main()
