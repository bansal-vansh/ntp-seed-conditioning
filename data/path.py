import random
from collections import defaultdict

from data.dataset import GPTDataset, CustomTokenizer
from data.graph_pretrain import make_pretraining_dataset
import os, json
import numpy as np
import networkx as nx

def build_path_vocab(N, H):
    """
    Builds the vocabulary for the path-finding dataset.

    Args:
        N (int): The number of nodes in the graph.
        H (int): The number of unique seed symbols.

    Returns:
        tuple: A tuple containing lists of nodes, seed tokens, special tokens,
               and the prefixes dictionary.
    """
    NODES = [f"v{i}" for i in range(N)]
    SEED_TOKENS = [f"<H{i}>" for i in range(H)]
    BOS = "<BOS>"
    EOS = "<EOS>"
    prefixes = {
        "train": "path",
        "pretrain": "edge"
    }
    return NODES, SEED_TOKENS, BOS, EOS, prefixes

def generate_fixed_degree_dag(nodes, out_degree=3):
    """
    Generates a Directed Acyclic Graph (DAG) where each node attempts to have
    a fixed number of outgoing edges ('out_degree'). This model is much less
    sensitive to its parameters than a probabilistic one.

    Args:
        nodes (list): A list of node names (e.g., ['v0', 'v1', ...]).
        out_degree (int): The desired number of outgoing edges for each node.

    Returns:
        defaultdict: An adjacency list representation of the graph.
    """
    graph = defaultdict(list)
    num_nodes = len(nodes)
    for i in range(num_nodes):
        # The pool of nodes that ui can connect to (uj where j > i) (where [u1, u2, ..., uN] is the list of nodes after shuffling)
        potential_targets = nodes[i+1:]
        
        # If the number of available targets is less than the desired degree,
        # connect to all of them.
        if len(potential_targets) <= out_degree:
            actual_targets = potential_targets
        else:
            # Randomly sample 'out_degree' nodes to connect to.
            actual_targets = random.sample(potential_targets, out_degree)
        
        graph[nodes[i]] = actual_targets
        
    return graph

def generate_hierarchical_dag(nodes, num_layers, nodes_per_layer, edge_prob=0.1, shuffle_nodes=True):
    """
    Generates a Hierarchical Directed Acyclic Graph (DAG) that is guaranteed
    to be connected.

    Args:
        nodes (list): A list of all node names.
        num_layers (int): The total number of layers.
        nodes_per_layer (int): The number of nodes within each layer.
        edge_prob (float): The probability of an edge existing between a node
                           in one layer and a node in the next.
        shuffle_nodes (bool): Parameter included to match the original signature.
                              The algorithm does not use this parameter.

    Returns:
        defaultdict: A defaultdict-based adjacency list representation of the graph.
    """
    if len(nodes) != num_layers * nodes_per_layer:
        raise ValueError("Total number of nodes must equal num_layers * nodes_per_layer")

    # This is the main loop from the algorithm (lines 18-20)
    while True:
        # Initialize an empty graph for this attempt
        graph = defaultdict(list)
        
        # Create the layers from the provided node list
        layers = [nodes[i:i + nodes_per_layer] for i in range(0, len(nodes), nodes_per_layer)]

        # This section implements the CreateLayeredDAG function (lines 7-15)
        for i in range(num_layers - 1):
            current_layer_nodes = layers[i]
            next_layer_nodes = layers[i+1]

            for node in current_layer_nodes:
                for target_node in next_layer_nodes:
                    if random.random() < edge_prob:
                        graph[node].append(target_node)
        
        # This is the connectivity check from the algorithm (line 20)
        # We convert our adjacency list to a NetworkX object temporarily to test it
        temp_nx_graph = nx.DiGraph(graph)
        # Ensure all nodes are in the graph for the connectivity check, even if they have no edges
        temp_nx_graph.add_nodes_from(nodes) 
        
        if nx.is_weakly_connected(temp_nx_graph):
            # If the graph is connected, exit the loop and return the graph
            return graph

def generate_bernoulli_dag(nodes, edge_prob=0.05):
    """
    Generates a random, connected Directed Acyclic Graph (DAG) and returns it
    as an adjacency list.

    The algorithm ensures the graph is acyclic by only allowing edges from lower-indexed
    nodes to higher-indexed nodes. It repeatedly generates graphs until a weakly
    connected one is found.

    Args:
        nodes (list): A list of node names for the graph.
        p (float): The probability of an edge existing between any two valid nodes.

    Returns:
        dict: An adjacency list representation of the connected DAG, where keys
              are node names and values are lists of their neighbors.
    """
    num_nodes = len(nodes)
    if num_nodes <= 0:
        raise ValueError("The list of nodes cannot be empty.")

    # Create a mapping from integer indices to the provided node names
    mapping = {i: name for i, name in enumerate(nodes)}

    # Loop until the generated DAG is connected
    while True:
        # Create a random upper triangular adjacency matrix
        random_matrix = np.random.choice([0, 1], size=(num_nodes, num_nodes), p=[1-edge_prob, edge_prob])
        adj_matrix = np.triu(random_matrix, k=1)

        # Create a temporary graph object
        dag = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)

        # Check if the graph is weakly connected
        if nx.is_weakly_connected(dag):
            # Relabel the integer nodes to the desired names
            nx.relabel_nodes(dag, mapping, copy=False)
            # Convert to an adjacency list and return
            return nx.to_dict_of_lists(dag)

def tokenize_graph(graph, tokenizer):
    tokenized_graph = {}
    for parent, children in graph.items():
        tokenized_children = [tokenizer.token_to_id[c] for c in children]
        tokenized_graph[tokenizer.token_to_id[parent]] = tokenized_children
    return tokenized_graph

def find_all_paths(graph, start_node, end_node, current_path=None):
    """
    Finds all paths between a start and end node in a DAG using DFS.

    Args:
        graph (dict): The graph's adjacency list.
        start_node (str): The starting node for the path.
        end_node (str): The target end node for the path.
        current_path (list, optional): The path traversed so far. Used for recursion.

    Returns:
        list: A list of lists, where each inner list represents a complete path
              from the start to the end node.
    """
    if current_path is None:
        current_path = []
    
    # Add current node to the path
    path = current_path + [start_node]

    # If we've reached the end, we have found a valid path
    if start_node == end_node:
        return [path]

    # If the start node is not in the graph, no paths exist
    if start_node not in graph:
        return []

    # Recurse for all neighbors
    all_found_paths = []
    for neighbor in graph[start_node]:
        # Continue the search from the neighbor
        new_paths = find_all_paths(graph, neighbor, end_node, path)
        for p in new_paths:
            all_found_paths.append(p)
            
    return all_found_paths

def find_one_random_path(graph, start_node, end_node):
    """
    Finds a single random path from start_node to end_node using a randomized DFS.
    This version corrects the usage of the 'visited' set.
    """
    def dfs(current_node, current_path):
        # Add the current node to the path for this specific recursive exploration
        path = current_path + [current_node]
        
        if current_node == end_node:
            return path

        if current_node not in graph:
            return None

        neighbors = list(graph.get(current_node, []))
        random.shuffle(neighbors)
        
        for neighbor in neighbors:
            # The 'path' list passed here correctly tracks visited nodes for the current branch
            if neighbor not in path:
                new_path = dfs(neighbor, path)
                if new_path:
                    return new_path
        
        return None

    return dfs(start_node, [])

def generate_path_strings(graph, nodes, num_samples, EVAL_TOKENIZER, uniform_sampling=True):
    """
    Generates training strings representing paths through the graph.
    Each string is formatted as "start_node end_node start_node ... end_node".

    Args:
        graph (dict): The graph's adjacency list.
        nodes (list): A list of all node names in the graph.
        num_samples (int): The number of path strings to generate.

    Returns:
        list: A list of formatted path strings.
    """
    path_strings = []
    attempts = 0
    max_attempts = num_samples * 5 # Avoid infinite loops in very sparse graphs
    prompts = []
    prompt_to_number_of_paths = defaultdict(int)
    
    while len(path_strings) < num_samples and attempts < max_attempts:
        attempts += 1
        
        # Sample two distinct nodes
        start_node, end_node = random.sample(nodes, 2)

        if EVAL_TOKENIZER.token_to_id[start_node] > EVAL_TOKENIZER.token_to_id[end_node]:
            start_node, end_node = end_node, start_node
        
        if uniform_sampling:
            all_paths = find_all_paths(graph, start_node, end_node)
            if all_paths:
                prompt = f"{start_node} {end_node}"
                prompt_to_number_of_paths[prompt] = len(all_paths)
                # If paths exist, pick one at random
                chosen_path = random.choice(all_paths)
            else:
                chosen_path = None
        # If uniform sampling is not used, find one random path
        else:
            # Find a single random path. This is much more efficient than finding all paths.
            chosen_path = find_one_random_path(graph, start_node, end_node)

        if chosen_path:
            # If a path was found, format the string as "start end path..."
            # which creates the duplicated format, e.g., "v1 v5 v1 v3 v5"
            path_str = " ".join(chosen_path)
            path_strings.append(path_str)
            prompts.append(f"{start_node} {end_node}")
            
    if len(path_strings) < num_samples:
        print(f"Warning: Could only generate {len(path_strings)}/{num_samples} paths. "
              "Consider increasing edge_prob or graph size.")

    return path_strings, prompts, prompt_to_number_of_paths


def generate_seeds(strings, seed_tokens, seed_len):
    """
    Generates a unique seed for each unique string provided.

    Args:
        strings (list): A list of strings to be seeded.
        seed_tokens (list): The vocabulary of seed symbols to use.
        seed_len (int): The number of seed symbols in each generated seed.

    Returns:
        dict: A dictionary mapping each unique string to its generated seed.
    """
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

def make_dataset(N, H, seed_len, graph_type="bernoulli", edge_prob=0.05, out_degree=4, num_layers=5, pretrain_adjacency=False, num_train_samples=5000, num_test_samples=1000, tokenizer=None, data_root=None, regenerate=False):
    """
    Main function to orchestrate the creation of the path-finding dataset.

    Args:
        N (int): Number of nodes in the graph.
        H (int): Number of seed symbols.
        seed_len (int): Length of the generated seeds.
        edge_prob (float): Probability of an edge in the DAG.
        num_train_samples (int): Number of samples for the training set.
        num_test_samples (int): Number of samples for the test set.
        tokenizer: A tokenizer object to be configured. Must be provided.

    Returns:
        tuple: Contains the train_dataset, test_dataset, raw test strings, and other
               useful artifacts for training and evaluation.
    """
    if tokenizer is None:
        raise ValueError("A tokenizer instance must be provided.")

    # 1. Build Vocabulary
    NODES, SEED_TOKENS, BOS, EOS, prefixes = build_path_vocab(N, H)

    # 2. Configure Tokenizer
    if tokenizer is None:
        raise ValueError("Tokenizer must be provided for dataset creation.")
    
    tokenizer.add_special_tokens({'bos_token': BOS})
    print(f"BOS token: {tokenizer.bos_token}, ID: {tokenizer.bos_token_id}")

    tokenizer.add_special_tokens({'eos_token': EOS})
    print(f"EOS token: {tokenizer.eos_token}, ID: {tokenizer.eos_token_id}")

    tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})
    print(f"PAD token: {tokenizer.pad_token}, ID: {tokenizer.pad_token_id}")

    tokenizer.add_tokens(NODES)
    tokenizer.add_special_tokens({'additional_special_tokens': SEED_TOKENS + list(prefixes.values())})

    random.shuffle(NODES)  # Shuffle nodes to ensure randomness in the graph structure
    EVAL_TOKENIZER = CustomTokenizer(NODES + SEED_TOKENS + [BOS, EOS] + list(prefixes.values()))
    if not data_root:
        raise ValueError("data_root must be specified.")
    
    if graph_type == "bernoulli":
        if edge_prob is None:
            raise ValueError("Edge probability must be specified for Bernoulli graph generation.")
        graph_label = f"graph_{graph_type}-prob{edge_prob}"

    elif graph_type == "fixed-degree":
        if out_degree is None:
            raise ValueError("Out-degree must be specified for fixed-degree graph generation.")
        if out_degree < 1 or out_degree >= N:
            raise ValueError(f"Invalid out-degree {out_degree}. Must be in range [1, {N-1}].")
        if out_degree > N // 2:
            print(f"Warning: Out-degree {out_degree} is high relative to the number of nodes {N}. This may lead to dense connections.")
        graph_label = f"graph_{graph_type}-out{out_degree}"

    elif graph_type == "hierarchical":
        if edge_prob is None or num_layers is None:
            raise ValueError("Edge probability and number of layers must be specified for hierarchical graph generation.")
        graph_label = f"graph_{graph_type}-prob{edge_prob}-L{num_layers}"

    else:
        raise ValueError(f"Unknown graph type: {graph_type}. Supported types are 'bernoulli', 'fixed-degree', and 'hierarchical'.")
    
    data_path = os.path.join(data_root, "path", f"N{N}-{graph_label}") if data_root else None
    if not os.path.exists(data_path):
        regenerate = True
    
    if not regenerate:
        print(f"Loading existing dataset from {data_path}")
        graph = json.load(open(os.path.join(data_path, "graph.json"), "r"))
        train_strs = json.load(open(os.path.join(data_path, "train.json"), "r"))
        test_strs = json.load(open(os.path.join(data_path, "test.json"), "r"))
        train_prompts = json.load(open(os.path.join(data_path, "train_prompts.json"), "r"))
        test_prompts = json.load(open(os.path.join(data_path, "test_prompts.json"), "r"))
        prompt_to_number_of_paths = json.load(open(os.path.join(data_path, "prompt_to_number_of_paths.json"), "r"))

        if len(train_strs) < num_train_samples or len(test_strs) < num_test_samples:
            print(f"Warning: Existing dataset has fewer samples than requested. Regenerating...")
            regenerate = True
            train_strs, train_prompts = [], []
            test_strs, test_prompts = [], []
            prompt_to_number_of_paths = defaultdict(int)
        else:
            train_strs = train_strs[:num_train_samples]
            train_prompts = train_prompts[:num_train_samples]
            test_strs = test_strs[:num_test_samples]
            test_prompts = test_prompts[:num_test_samples]
    
    if regenerate:
        print(f"Generating new dataset with {num_train_samples} training samples and {num_test_samples} test samples...")
        
        # 3. Generate Graph
        if graph_type == "bernoulli":
            graph = generate_bernoulli_dag(NODES, edge_prob=edge_prob)

        elif graph_type == "fixed-degree":
            graph = generate_fixed_degree_dag(NODES, out_degree=out_degree)

        elif graph_type == "hierarchical":
            nodes_per_layer = N // num_layers
            graph = generate_hierarchical_dag(NODES, num_layers=num_layers, nodes_per_layer=nodes_per_layer, edge_prob=edge_prob)
            
        else:
            raise ValueError(f"Unknown graph type: {graph_type}. Supported types are 'bernoulli', 'fixed-degree', and 'hierarchical'.")

        # Save the generated graph
        os.makedirs(data_path, exist_ok=True)
        with open(os.path.join(data_path, "graph.json"), "w") as f:
            json.dump(graph, f)

        # 4. Generate Path Strings
        total_samples = num_train_samples + num_test_samples
        strs, prompts, prompt_to_number_of_paths = generate_path_strings(graph, NODES, total_samples, EVAL_TOKENIZER, uniform_sampling=True)
        
        # Split into train and test sets
        train_strs = strs[:num_train_samples]
        test_strs = strs[num_train_samples:]
        train_prompts = prompts[:num_train_samples]
        test_prompts = prompts[num_train_samples:]

        with open(os.path.join(data_path, "train.json"), "w") as f:
            json.dump(train_strs, f)
        with open(os.path.join(data_path, "test.json"), "w") as f:
            json.dump(test_strs, f)
        with open(os.path.join(data_path, "train_prompts.json"), "w") as f:
            json.dump(train_prompts, f)
        with open(os.path.join(data_path, "test_prompts.json"), "w") as f:
            json.dump(test_prompts, f)
        with open(os.path.join(data_path, "prompt_to_number_of_paths.json"), "w") as f:
            json.dump(prompt_to_number_of_paths, f)
        print(f"Successfully generated and saved {len(train_strs)} training samples and {len(test_strs)} test samples.")
    
    # # 4. Generate Path Strings
    strs = train_strs + test_strs
    str_to_seed = generate_seeds(strs, SEED_TOKENS, seed_len)
    print(f"Generated {len(str_to_seed)} unique seeds for the strings.")
    
    assert len(train_strs) == len(train_prompts) == num_train_samples, "Mismatch in number of training strings and prompts."
    assert len(test_strs) == len(test_prompts) == num_test_samples, "Mismatch in number of test strings and prompts."

    train_dataset = GPTDataset(train_strs, str_to_seed=str_to_seed, tokenizer=tokenizer, seed_len=seed_len, seed_tokens=SEED_TOKENS, dataset_name="path")
    test_dataset = GPTDataset(test_strs, str_to_seed=str_to_seed, tokenizer=tokenizer, seed_len=seed_len, seed_tokens=SEED_TOKENS, dataset_name="path")

    pretrain_dataset = make_pretraining_dataset(graph, prefix=prefixes["pretrain"], tokenizer=tokenizer) if pretrain_adjacency else None

    tokenized_graph = tokenize_graph(graph, EVAL_TOKENIZER)

    print(f"Successfully generated {len(train_strs)} training samples and {len(test_strs)} test samples.")
    
    return train_dataset, test_dataset, pretrain_dataset, train_prompts, test_prompts, tokenized_graph, SEED_TOKENS, EVAL_TOKENIZER, prefixes, prompt_to_number_of_paths