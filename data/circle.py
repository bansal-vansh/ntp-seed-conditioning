import random
from collections import defaultdict
from data.dataset import GPTDataset, CustomTokenizer
import os, json

def build_vocab(M, H):
    VOCAB = [f"v{i}" for i in range(M)]
    SEED_TOKENS = [f"<H{i}>" for i in range(H)]
    BOS = "<BOS>"
    EOS = "<EOS>"

    return VOCAB, SEED_TOKENS, BOS, EOS

def generate_circles(vocab, N, num_samples):
    assert len(vocab) >= N
    circles = []
    resolving_permutations = []
    for _ in range(num_samples):
        nodes = random.sample(vocab, N)
        edges = [(nodes[i], nodes[(i + 1) % N]) for i in range(N)]
        pi_inv = list(range(N))
        random.shuffle(pi_inv)  # generate random π
        edges_shuffled = [edges[j] for j in pi_inv]

        pi = [0] * N
        for i in range(N):
            pi[pi_inv[i]] = i
        
        circles.append(" ".join([v for edge in edges_shuffled for v in edge]))  # store circle as string
        resolving_permutations.append(canonicalize(pi))  # store π

    return circles, resolving_permutations

# revisit this code later
def compute_canonical_permutation(seq, N):
    """
    Try to compute resolving permutation π.
    If walk exists, return π (list of indices).
    If walk invalid (bad degrees, cannot close), return None.
    """
    if len(seq) != 2 * N:
        return None
    
    edges = [(seq[i], seq[i+1]) for i in range(0, len(seq), 2)]
    adj = {}
    edge_to_index = {}

    for i in range(N):
        u, v = edges[i]
        adj[u] = v
        edge_to_index[(u, v)] = i

    start = seq[0]
    visited = set()
    pi = []

    current = start
    for _ in range(N):
        visited.add(current)
        neighbor = adj.get(current)
        if neighbor is None:
            return None
        index = edge_to_index.get((current, neighbor))
        if index is None:
            return None
        pi.append(index)
        current = neighbor

    if current != start or len(visited) != N:
        return None

    return tuple(pi)

def canonicalize(pi):
    """
    Canonicalize π by rotation to start from 0.
    """
    start = pi.index(0)
    return tuple(pi[start:] + pi[:start])

def generate_seeds(strings, resolving_permutations, seed_tokens, seed_len, seed_per_pi=True):
    key_to_seed = {}
    str_to_seed = {}

    if len(seed_tokens) ** seed_len > len(strings):
        used_seeds = set()
        for string, pi in zip(strings, resolving_permutations):

            if seed_per_pi:
                key = pi # seed per pi
            else:
                key = string  # seed per string

            if key not in key_to_seed:
                while True:
                    h = "".join([random.choice(seed_tokens) for _ in range(seed_len)])
                    if h not in used_seeds:
                        key_to_seed[key] = h
                        used_seeds.add(h)
                        break

            str_to_seed[string] = key_to_seed[key]

    elif seed_len > 0:
        raise ValueError("The number of unique seeds is too small for the number of samples. "
                         "Consider increasing seed_len or reducing num_samples.")

    return str_to_seed

def make_dataset(M, N, H, seed_len, seed_per_pi=True, num_train_samples=5000, num_test_samples=5000, tokenizer=None, data_root=None, regenerate=False, add_new_tokens=False):
    VOCAB, SEED_TOKENS, BOS, EOS = build_vocab(M, H)
    EVAL_TOKENIZER = CustomTokenizer()

    if tokenizer is None:
        raise ValueError("Tokenizer must be provided for dataset creation.")
    
    if add_new_tokens: tokenizer.add_tokens(VOCAB)
    tokenizer.add_special_tokens({'bos_token': BOS, 'eos_token': EOS, "pad_token": EOS})
    tokenizer.add_special_tokens({'additional_special_tokens': SEED_TOKENS})

    print(f"BOS token: {tokenizer.bos_token}, ID: {tokenizer.bos_token_id}")
    print(f"EOS token: {tokenizer.eos_token}, ID: {tokenizer.eos_token_id}")
    print(f"PAD token: {tokenizer.pad_token}, ID: {tokenizer.pad_token_id}")

    EVAL_TOKENIZER.add_tokens(VOCAB)
    EVAL_TOKENIZER.add_special_tokens({'bos_token': BOS, 'eos_token': EOS, "pad_token": EOS})
    EVAL_TOKENIZER.add_special_tokens({'additional_special_tokens': SEED_TOKENS})

    if not data_root:
        raise ValueError("data_root must be specified.")

    data_path = os.path.join(data_root, "circle", f"M{M}-N{N}") if data_root else None
    if not os.path.exists(data_path):
        regenerate = True
    seed_dict_path = os.path.join(data_path, f"str_to_seed_H{H}_HL{seed_len}.json")
    
    if not regenerate:
        print(f"Loading existing dataset from {data_path}")
        train_strs = json.load(open(os.path.join(data_path, "train.json"), "r"))
        test_strs = json.load(open(os.path.join(data_path, "test.json"), "r"))
        train_can_perms = json.load(open(os.path.join(data_path, "train_perms.json"), "r"))
        test_can_perms = json.load(open(os.path.join(data_path, "test_perms.json"), "r"))
        try:
            str_to_seed = json.load(open(seed_dict_path, "r"))
        except FileNotFoundError:
            print("Seeds file not found. Regenerating seeds.")
            str_to_seed = {}

        if len(train_strs) < num_train_samples or len(test_strs) < num_test_samples:
            print(f"Warning: Existing dataset has fewer samples than requested. Regenerating...")
            regenerate = True
            train_strs, train_can_perms = [], []
            test_strs, test_can_perms = [], []
            str_to_seed = {}
        else:
            train_strs = train_strs[:num_train_samples]
            train_can_perms = [tuple(perm) for perm in train_can_perms[:num_train_samples]]
            test_strs = test_strs[:num_test_samples]
            test_can_perms = [tuple(perm) for perm in test_can_perms[:num_test_samples]]
    
    if regenerate:
        print(f"Generating new dataset with {num_train_samples} training samples and {num_test_samples} test samples...")
        # Save the generated graph
        os.makedirs(data_path, exist_ok=True)

        # 4. Generate Path Strings
        total_samples = num_train_samples + num_test_samples
        # Generate circles + known π for each
        strs, can_perms = generate_circles(VOCAB, N, total_samples)
        # Split into train and test sets
        train_strs = strs[:num_train_samples]
        test_strs = strs[num_train_samples:]
        train_can_perms = [tuple(perm) for perm in can_perms[:num_train_samples]]
        test_can_perms = [tuple(perm) for perm in can_perms[num_train_samples:]]
        str_to_seed = {}

        with open(os.path.join(data_path, "train.json"), "w") as f:
            json.dump(train_strs, f)
        with open(os.path.join(data_path, "test.json"), "w") as f:
            json.dump(test_strs, f)
        with open(os.path.join(data_path, "train_perms.json"), "w") as f:
            json.dump(train_can_perms, f)
        with open(os.path.join(data_path, "test_perms.json"), "w") as f:
            json.dump(test_can_perms, f)
        print(f"Successfully generated and saved {len(train_strs)} training samples and {len(test_strs)} test samples.")
    
    # # 4. Generate Path Strings
    strs = train_strs + test_strs
    can_perms = train_can_perms + test_can_perms
    if regenerate or not str_to_seed:
        # Generate seeds according to desired mode
        str_to_seed = generate_seeds(
            strs, can_perms, SEED_TOKENS, seed_len, seed_per_pi=seed_per_pi
        )
        with open(seed_dict_path, "w") as f:
            json.dump(str_to_seed, f)
        print(f"Generated and saved {len(str_to_seed)} unique seeds for the strings.")
    
    assert len(train_strs) == len(train_can_perms) == num_train_samples, "Mismatch in number of training strings and permutations."
    assert len(test_strs) == len(test_can_perms) == num_test_samples, "Mismatch in number of test strings and permutations."

    # Build dataset
    train_dataset = GPTDataset(train_strs, str_to_seed=str_to_seed, tokenizer=tokenizer, seed_len=seed_len, seed_tokens=SEED_TOKENS, dataset_name="circle")
    test_dataset = GPTDataset(test_strs, str_to_seed=str_to_seed, tokenizer=tokenizer, seed_len=seed_len, seed_tokens=SEED_TOKENS, dataset_name="circle")

    return train_dataset, test_dataset, tokenizer, train_strs, train_can_perms, VOCAB, SEED_TOKENS, EVAL_TOKENIZER