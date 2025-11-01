import random
from collections import defaultdict
from data.dataset import GPTDataset, CustomTokenizer
import os, json

def build_vocab(N, H):
    VOCAB = [f"v{i}" for i in range(N)]
    SEED_TOKENS = [f"<H{i}>" for i in range(H)]
    BOS = "<BOS>"
    EOS = "<EOS>"

    return VOCAB, SEED_TOKENS, BOS, EOS

def generate_perms(vocab, N, num_samples):
    assert len(vocab) == N
    strings = []
    perms = []
    for _ in range(num_samples):
        perm = random.sample(range(N), N)  # generate a random permutation of indices
        strings.append(" ".join([vocab[perm[i]] for i in range(N)]))  # store permutation as string
        perms.append(tuple(perm))  # store permutation as tuple
    return strings, perms

def canonicalize(pi):
    """
    Canonicalize π by rotation to start from 0.
    """
    start = pi.index(0)
    return tuple(pi[start:] + pi[:start])

def compute_permutation(seq, vocab_ids):
    perm = []
    for token_id in vocab_ids:
        if token_id in seq:
            idx = seq.index(token_id)
        else:
            return None
        perm.append(idx)
    return tuple(perm)

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

def make_dataset(N, H, seed_len, num_train_samples=5000, num_test_samples=5000, tokenizer=None, data_root=None, regenerate=False):
    VOCAB, SEED_TOKENS, BOS, EOS = build_vocab(N, H)
    EVAL_TOKENIZER = CustomTokenizer()

    if tokenizer is None:
        raise ValueError("Tokenizer must be provided for dataset creation.")
    
    tokenizer.add_tokens(VOCAB)
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

    data_path = os.path.join(data_root, "permutation", f"N{N}") if data_root else None
    if not os.path.exists(data_path):
        regenerate = True
    
    if not regenerate:
        print(f"Loading existing dataset from {data_path}")
        train_strs = json.load(open(os.path.join(data_path, "train.json"), "r"))
        train_perms = json.load(open(os.path.join(data_path, "train_perms.json"), "r"))
        test_strs = json.load(open(os.path.join(data_path, "test.json"), "r"))
        try:
            str_to_seed = json.load(open(os.path.join(data_path, f"str_to_seed_{seed_len}.json"), "r"))
        except FileNotFoundError:
            print("seeds file not found. Regenerating seeds.")
            str_to_seed = {}
        

        if len(train_strs) < num_train_samples or len(test_strs) < num_test_samples:
            print(f"Warning: Existing dataset has fewer samples than requested. Regenerating...")
            regenerate = True
            train_strs = []
            train_perms = []
            test_strs = []
            str_to_seed = {}
        else:
            train_strs = train_strs[:num_train_samples]
            train_perms = train_perms[:num_train_samples]
            test_strs = test_strs[:num_test_samples]
    
    if regenerate:
        print(f"Generating new dataset with {num_train_samples} training samples and {num_test_samples} test samples...")
        # Save the generated graph
        os.makedirs(data_path, exist_ok=True)

        # 4. Generate Path Strings
        total_samples = num_train_samples + num_test_samples
        # Generate circles + known π for each
        strs, perms = generate_perms(VOCAB, N, total_samples)
        # Split into train and test sets
        train_strs = strs[:num_train_samples]
        train_perms = perms[:num_train_samples]
        test_strs = strs[num_train_samples:]
        str_to_seed = {}

        with open(os.path.join(data_path, "train.json"), "w") as f:
            json.dump(train_strs, f)
        with open(os.path.join(data_path, "test.json"), "w") as f:
            json.dump(test_strs, f)
        with open(os.path.join(data_path, "train_perms.json"), "w") as f:
            json.dump(train_perms, f)
        print(f"Successfully generated and saved {len(train_strs)} training samples and {len(test_strs)} test samples.")
    
    # # 4. Generate Path Strings
    strs = train_strs + test_strs
    # Generate seeds according to desired mode
    if not str_to_seed:
        # Generate seeds according to desired mode
        str_to_seed = generate_seeds(
            strs, SEED_TOKENS, seed_len
        )
        with open(os.path.join(data_path, f"str_to_seed_{seed_len}.json"), "w") as f:
            json.dump(str_to_seed, f)
        print(f"Generated and saved {len(str_to_seed)} unique seeds for the strings.")
    
    # Build dataset
    train_dataset = GPTDataset(train_strs, str_to_seed=str_to_seed, tokenizer=tokenizer, seed_len=seed_len, seed_tokens=SEED_TOKENS, dataset_name="permutation")
    test_dataset = GPTDataset(test_strs, str_to_seed=str_to_seed, tokenizer=tokenizer, seed_len=seed_len, seed_tokens=SEED_TOKENS, dataset_name="permutation")

    return train_dataset, test_dataset, tokenizer, train_strs, train_perms, VOCAB, SEED_TOKENS, EVAL_TOKENIZER