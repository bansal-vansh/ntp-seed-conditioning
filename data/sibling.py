import random
from collections import defaultdict
import os, json

from data.dataset import GPTDataset, CustomTokenizer
from data.graph_pretrain import make_pretraining_dataset

def build_vocab(P, C, H):
    PARENTS = [f"p{i}" for i in range(P)]
    CHILDREN = [f"c{i}" for i in range(C)]
    SEED_TOKENS = [f"<H{i}>" for i in range(H)]
    BOS = "<BOS>"
    EOS = "<EOS>"
    prefixes = {
        "pretrain": "<edge>",
        "train": "<triplet>" 
    }
    return PARENTS, CHILDREN, SEED_TOKENS, BOS, EOS, prefixes

def generate_fixed_bipartite_graph(parents, children):
    num_children_per_parent = len(children) // len(parents)
    E = defaultdict(list)
    for i, p in enumerate(parents):
        for j in range(num_children_per_parent):
            E[p].append(children[i * num_children_per_parent + j])
    return {p: ch for p, ch in E.items() if len(ch) >= 1}

def generate_random_bipartite_graph(parents, children, edge_prob=0.2):
    E = defaultdict(list)
    for c in children:
        for p in parents:
            if random.random() < edge_prob:
                E[p].append(c)
    return {p: ch for p, ch in E.items() if len(ch) >= 1}

def tokenize_graph(graph, tokenizer):
    tokenized_graph = {}
    for parent, children in graph.items():
        tokenized_children = [tokenizer._convert_token_to_id(c) for c in children]
        tokenized_graph[tokenizer._convert_token_to_id(parent)] = tokenized_children
    return tokenized_graph

def generate_sibling_parent_strings(graph, num_samples, planning=True, separate=False):
    triples = []
    for _ in range(num_samples):
        parent = random.choice(list(graph.keys()))
        children = graph[parent]
        if len(children) < 2:
            continue
        if not separate:
            a, b = random.sample(children, 2)
            if not planning:
                triples.append(" ".join([parent, a, b]))
            else:
                triples.append(" ".join([a, b, parent]))
        else:
            half = len(children) // 2
            first_half = children[:half]
            second_half = children[half:]
            a, b = random.choice(first_half), random.choice(second_half)
            if not planning:
                triples.append(" ".join([parent, a, b]))
            else:
                triples.append(" ".join([a, b, parent]))

    return triples

def generate_seeds(strings, seed_tokens, seed_len):
    str_to_seed = {}

    if len(seed_tokens) ** seed_len > len(strings):
        used_seeds = set()
        for string in strings:
            if string not in str_to_seed:
                while True:
                    h = "".join([random.choice(seed_tokens) for _ in range(seed_len)])
                    if h not in used_seeds:
                        str_to_seed[string] = h
                        used_seeds.add(h)
                        break
    elif seed_len > 0:
        raise ValueError("The number of unique seeds is too small for the number of samples. "
                         "Consider increasing seed_len or reducing num_samples.")

    return str_to_seed

def make_dataset(P, C, H, seed_len, edge_prob, pretrain_adjacency=False, planning=True, fixed=True, num_train_samples=5000, num_test_samples=1000, tokenizer=None, data_root=None, regenerate=False, add_new_tokens=False):
    PARENTS, CHILDREN, SEED_TOKENS, BOS, EOS, prefixes = build_vocab(P, C, H)
    if tokenizer is None:
        raise ValueError("Tokenizer must be provided for dataset creation.")

    if add_new_tokens: tokenizer.add_tokens(PARENTS + CHILDREN)
    tokenizer.add_special_tokens({'bos_token': BOS, 'eos_token': EOS, "pad_token": EOS})
    tokenizer.add_special_tokens({'additional_special_tokens': SEED_TOKENS + list(prefixes.values())})

    print(f"BOS token: {tokenizer.bos_token}, ID: {tokenizer.bos_token_id}")
    print(f"EOS token: {tokenizer.eos_token}, ID: {tokenizer.eos_token_id}")
    print(f"PAD token: {tokenizer.pad_token}, ID: {tokenizer.pad_token_id}")

    EVAL_TOKENIZER = CustomTokenizer()

    EVAL_TOKENIZER.add_tokens(PARENTS + CHILDREN)
    EVAL_TOKENIZER.add_special_tokens({'bos_token': BOS, 'eos_token': EOS, "pad_token": EOS})
    EVAL_TOKENIZER.add_special_tokens({'additional_special_tokens': SEED_TOKENS + list(prefixes.values())})
    
    if not data_root:
        raise ValueError("data_root must be specified.")
    
    if fixed:
        graph_label = f"fixed"
    else:
        if edge_prob is None:
            raise ValueError("Edge probability must be specified for Bernoulli graph generation.")
        graph_label = f"prob{edge_prob}"
    planning_label = "planning" if planning else "no_planning"
    data_path = os.path.join(data_root, "sibling", f"P{P}-C{C}-{graph_label}-{planning_label}") if data_root else None
    seed_dict_path = os.path.join(data_path, f"str_to_seed_H{H}_HL{seed_len}.json")

    if not os.path.exists(data_path):
        regenerate = True
    
    if not regenerate:
        print(f"Loading existing dataset from {data_path}")
        graph = json.load(open(os.path.join(data_path, "graph.json"), "r"))
        train_strs = json.load(open(os.path.join(data_path, "train.json"), "r"))
        test_strs = json.load(open(os.path.join(data_path, "test.json"), "r"))
        try:
            str_to_seed = json.load(open(seed_dict_path, "r"))
        except FileNotFoundError:
            print("seeds file not found. Regenerating seeds.")
            str_to_seed = {}

        if len(train_strs) < num_train_samples or len(test_strs) < num_test_samples:
            print(f"Warning: Existing dataset has fewer samples than requested. Regenerating...")
            regenerate = True
            train_strs, test_strs = [], []
            graph = {}
            str_to_seed = {}
        else:
            train_strs = train_strs[:num_train_samples]
            test_strs = test_strs[:num_test_samples]

    if regenerate:
        print(f"Generating new dataset with {num_train_samples} training samples and {num_test_samples} test samples...")
        random.shuffle(CHILDREN)
        # 3. Generate Graph
        if fixed:
            graph = generate_fixed_bipartite_graph(PARENTS, CHILDREN)
        else:
            graph = generate_random_bipartite_graph(PARENTS, CHILDREN, edge_prob=edge_prob)
        # Save the generated graph
        os.makedirs(data_path, exist_ok=True)
        with open(os.path.join(data_path, "graph.json"), "w") as f:
            json.dump(graph, f)
        
        strs = generate_sibling_parent_strings(graph, num_train_samples + num_test_samples, planning=planning, separate=False)

        train_strs = strs[:num_train_samples]
        test_strs = strs[num_train_samples:]
        str_to_seed = {}

        with open(os.path.join(data_path, "train.json"), "w") as f:
            json.dump(train_strs, f)
        with open(os.path.join(data_path, "test.json"), "w") as f:
            json.dump(test_strs, f)
    
    strs = train_strs + test_strs
    if not str_to_seed or regenerate:
        str_to_seed = generate_seeds(strs, SEED_TOKENS, seed_len)
        
        with open(seed_dict_path, "w") as f:
            json.dump(str_to_seed, f)
        print(f"Generated and saved {len(str_to_seed)} unique seeds for the strings.")

    if not pretrain_adjacency: prefixes["train"] = ""
    train_dataset = GPTDataset(train_strs, str_to_seed=str_to_seed, tokenizer=tokenizer, prefix=prefixes["train"], seed_len=seed_len, seed_tokens=SEED_TOKENS, dataset_name="sibling")
    test_dataset = GPTDataset(test_strs, str_to_seed=str_to_seed, tokenizer=tokenizer, prefix=prefixes["train"], seed_len=seed_len, seed_tokens=SEED_TOKENS, dataset_name="sibling")

    pretrain_dataset = make_pretraining_dataset(graph, prefix=prefixes["pretrain"], tokenizer=tokenizer) if pretrain_adjacency else None

    tokenized_graph = tokenize_graph(graph, EVAL_TOKENIZER)

    return train_dataset, test_dataset, pretrain_dataset, train_strs, tokenized_graph, SEED_TOKENS, EVAL_TOKENIZER, prefixes