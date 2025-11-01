import random
from collections import defaultdict
from itertools import combinations
from torch.utils.data import Dataset
import torch
import numpy as np

from utils import set_seed
from dataset import GPTDataset, generate_training_strings

def build_vocab(K, L, H):
    BOS_ID = 0
    V = (K**(L+1) - 1)//(K - 1) - 1 # Number of unique vertex tokens in a complete k-ary tree excluding the root
    VOCAB = list(range(1, V + 1))
    SEED_TOKENS = list(range(V + 1, V + H + 1))
    VOCAB_SIZE = V + H + 1
    return BOS_ID, VOCAB, SEED_TOKENS, VOCAB_SIZE

def build_k_ary_tree(vocab, K, L):
    tree = defaultdict(list)
    def add_children(prefix):
        if len(prefix) == L:
            return
        children = random.sample(vocab, K)
        tree[tuple(prefix)] = children
        for c in children:
            add_children(prefix + [c])
    add_children([])
    return tree

def build_unique_k_ary_tree(vocab, K, L):
    tree = defaultdict(list)
    used_tokens = set()
    token_pool = vocab.copy()
    random.shuffle(token_pool)  # randomize global node allocation
    token_iter = iter(token_pool)

    def add_children(prefix):
        if len(prefix) == L:
            return
        children = []
        for _ in range(K):
            try:
                token = next(token_iter)
                children.append(token)
                used_tokens.add(token)
            except StopIteration:
                break  # stop if we run out of unique tokens
        if children:
            tree[tuple(prefix)] = children
            for c in children:
                add_children(prefix + [c])
    add_children([])
    return tree

def generate_all_path_strings(tree):
    all_paths = []
    def dfs(node, path):
        if node not in tree:
            all_paths.append(path)
            return
        for child in tree[node]:
            dfs(tuple(list(node) + [child]), path + [child])
    dfs(tuple(), [])
    return all_paths

def make_dataset(K, L, H, seed_len, unique=True, num_samples=5000, seed=42):
    set_seed(seed)
    BOS_ID, VOCAB, SEED_TOKENS, VOCAB_SIZE = build_vocab(K, L, H)
    random.shuffle(VOCAB)
    if unique:
        tree = build_unique_k_ary_tree(VOCAB, K, L)
    else:
        tree = build_k_ary_tree(VOCAB, K, L)
    all_paths = generate_all_path_strings(tree)
    if num_samples==None:
        num_samples = len(all_paths)
    train_strs, triple_to_seed = generate_training_strings(all_paths, num_samples, SEED_TOKENS, seed_len, seed=seed)
    dataset = GPTDataset(train_strs, triple_to_seed, BOS_ID, use_seed=(seed_len > 0))
    return dataset, train_strs, tree, BOS_ID, SEED_TOKENS, VOCAB_SIZE
