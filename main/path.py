import argparse
import numpy as np
import torch
from data.dataset import collate_fn
from data.path import make_dataset
from model.eval import decode_batch, generate_samples
from model.train import train_main
from model.utils import set_seed, get_model_tokenizer

def parse_args():
    parser = argparse.ArgumentParser(description="Train GPT model on the path finding task")
    parser.add_argument('--N', type=int, default=200, help='Number of nodes in the graph')
    parser.add_argument('--H', type=int, default=26, help='Seed vocabulary size')
    parser.add_argument('--HL', type=int, default=5, help='Seed length')
    parser.add_argument('--prob', type=float, default=0.05, help='Edge probability')
    parser.add_argument('--num_layers', type=float, default=5, help='Number of layers in the hierarchical graph structure')
    parser.add_argument('--out_degree', type=float, default=5, help='Out-degree in the fixed-degree graph structure')
    parser.add_argument('--top_p', type=float, default=0.9, help='Top p for nucleus sampling')
    parser.add_argument('--graph_type', type=str, default='bernoulli', choices=['bernoulli', 'fixed-degree', 'hierarchical'], help='Graph type')
    parser.add_argument('--no_planning', action='store_true', help='Disable planning')
    parser.add_argument('--num_train_samples', type=int, default=10000, help='Number of training samples')
    parser.add_argument('--num_eval_samples', type=int, default=1000, help='Number of evaluation samples')
    parser.add_argument('--epochs', type=int, default=15, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Training batch size')
    parser.add_argument('--eval_batch_size', type=int, default=256, help='Evaluation batch size')
    parser.add_argument('--device', type=int, default=0, help='CUDA device ID')
    parser.add_argument('--pretrain_adjacency', action='store_true', help='Pretrain on graph adjacency bigrams')
    parser.add_argument('--model_type', type=str, default='gpt2', choices=['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'], help='Model size')
    parser.add_argument('--save_name', type=str, default='experiment', help='Experiment name')
    parser.add_argument('--dataset_name', type=str, default='sibling', help='Dataset name')
    parser.add_argument('--eval_runs', type=int, default=5, help='Number of evaluation runs per checkpoint')
    parser.add_argument('--num_ckpts', type=int, default=100, help='Number of checkpoints')
    return parser.parse_args()

def main():
    args = parse_args()

    DEVICE = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    model, tokenizer = get_model_tokenizer(args, DEVICE)
        
    train_dataset, test_dataset, pretrain_dataset, train_prompts, test_prompts, tokenized_graph, SEED_TOKENS, EVAL_TOKENIZER, prefixes, prompts_to_num_samples = make_dataset(N=args.N,
                                                                                                                                                                            H=args.H,
                                                                                                                                                                            seed_len=args.HL,
                                                                                                                                                                            graph_type=args.graph_type,
                                                                                                                                                                            edge_prob=args.prob,
                                                                                                                                                                            num_layers=args.num_layers,
                                                                                                                                                                            out_degree=args.out_degree,
                                                                                                                                                                            pretrain_adjacency=args.pretrain_adjacency,
                                                                                                                                                                            planning=not args.no_planning,
                                                                                                                                                                            fixed=not args.random,
                                                                                                                                                                            num_train_samples=args.num_train_samples,
                                                                                                                                                                            num_test_samples=args.num_eval_samples,
                                                                                                                                                                            tokenizer=tokenizer)
    data_collator = lambda features: collate_fn(features, tokenizer=tokenizer)
    model.resize_token_embeddings(len(tokenizer))

    train_set = set(tuple(EVAL_TOKENIZER.encode(s)) for s in train_dataset.strings)
    print(f"Training set size: {len(train_set)}")
    
    def is_memorized(seq):
        return tuple(seq) in train_set

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
                       seed=42, 
                       batch_size=args.eval_batch_size, 
                       num_eval_samples=args.num_eval_samples,
                       top_p=args.top_p):
        
        set_seed(seed)
        print(f"\nEvaluating {label} for seed {seed}...")
        num_conditional_samples = [prompts_to_num_samples[prompt] for prompt in test_prompts] if test_prompts and prompts_to_num_samples else None
        samples = generate_samples(model, 
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
                                   batch_size=batch_size, 
                                   conditional_prompts=test_prompts,
                                   num_conditional_samples=num_conditional_samples,
                                   )
        tokenized_samples = [EVAL_TOKENIZER.encode(s) for s in samples]
        start = 0
        representation_power = []
        creativity = []
        uniqueness = []
        coherence = []
        if num_conditional_samples is None:
            num_conditional_samples = [len(tokenized_samples)]
        for i in num_conditional_samples:
            end = start + i
            print(f"Evaluating {i} samples for prompt {test_prompts[start]}")
            conditional_samples = tokenized_samples[start:end]
            unique_samples = set([tuple(s) for s in conditional_samples])

            num_memorized = len([s for s in unique_samples if is_memorized(s)])
            num_creative = len([s for s in unique_samples if is_coherent(s, tokenized_graph) and not is_memorized(s)])
            num_unique = len(unique_samples)
            num_coherent = len([s for s in conditional_samples if is_coherent(s, tokenized_graph)])

            representation_power.append(num_memorized / len(conditional_samples))
            creativity.append(num_creative / len(conditional_samples)) 
            uniqueness.append(num_unique / len(conditional_samples)) 
            coherence.append(num_coherent / len(conditional_samples)) 
            # perplexity = compute_perplexity(model, test_dataset, batch_size=batch_size)

            # print(f"Perplexity: {perplexity:.4f}")
            print(f"Coherence: {coherence[-1]:.4f} ({num_coherent}/{len(samples)})")
            print(f"Representation power: {representation_power[-1]:.4f} ({num_memorized}/{len(samples)})")
            print(f"Creativity: {creativity[-1]:.4f} ({num_creative}/{len(samples)})")
            print(f"Uniqueness: {uniqueness[-1]:.4f} ({num_unique}/{len(samples)})")

            start = end
        if len(num_conditional_samples) == 1:
            representation_power = representation_power[0]
            creativity = creativity[0]
            uniqueness = uniqueness[0]
            coherence = coherence[0]
        
        return np.array([
            representation_power,
            creativity,
            uniqueness,
            coherence,
        ])

    if args.pretrain_adjacency:
        print("Pretraining on graph adjacency bigrams...")
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
    
    train_main(
        model=model,
        dataset_name=args.dataset_name,
        save_name=args.save_name,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        temperatures=[0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0],
        num_eval_runs=args.eval_runs,
        train_dataset=train_dataset,
        data_collator=data_collator,
        device=DEVICE,
        decode_fn=decode_batch,
        eval_fn=evaluate_model,
        num_checkpoints=args.num_ckpts,
        num_workers=16
    )

if __name__ == "__main__":
    main()
