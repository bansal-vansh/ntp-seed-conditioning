from data.dataset import GPTDataset

def make_pretraining_dataset(graph, prefix, tokenizer=None):
    
    strs =[f"{parent} {child}" for parent, siblings in graph.items() for child in siblings]
    str_to_seed = {}
    pretrain_dataset = GPTDataset(strs, prefix=prefix, str_to_seed=str_to_seed, tokenizer=tokenizer)

    return pretrain_dataset