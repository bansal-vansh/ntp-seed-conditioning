import torch
from torch.utils.data import Dataset
import random
import json, os, re
from transformers import PreTrainedTokenizer
from typing import Dict, List, Optional, Tuple

def sample_training_strings_from_all(all_strings, num_samples, seed_tokens, seed_len):
    train_strings = random.sample(all_strings, min(num_samples, len(all_strings)))

    str_to_seed = {}
    if len(seed_tokens)**seed_len > num_samples:
        used_seeds = set()
        for string in train_strings:
            while True:
                h = " ".join([random.choice(seed_tokens) for _ in range(seed_len)])
                if h not in used_seeds:
                    str_to_seed[string] = h
                    used_seeds.add(h)
                    break
    elif seed_len > 0:
        raise ValueError("The number of unique seeds is too small for the number of samples. "
                         "Consider increasing seed_len or reducing num_samples.")

    return train_strings, str_to_seed

class GPTDataset(Dataset):
    """
    A generalized, memory-efficient, and fast "lazy-loading" version of GPTDataset.

    It accepts a list of strings and processes them one by one on-the-fly,
    avoiding high memory consumption and slow initialization for large datasets.
    """
    def __init__(self, strings, prompts=None, tokenizer=None, prefix="", seed_len=0, seed_tokens=[], str_to_seed=None, max_length=512, dataset_name=None):
        """
        Initialization is now instantaneous. We just store references to the data
        and configuration. No processing happens here.
        """
        self.prefix = prefix
        self.strings = strings
        self.prompts = prompts
        self.tokenizer = tokenizer
        self.seed_len = seed_len
        self.use_seed = seed_len > 0
        self.seed_tokens = seed_tokens
        self.str_to_seed = str_to_seed or {}
        self.max_length = max_length
        self.dataset_name = dataset_name
        
        if self.tokenizer is None:
            raise ValueError("Tokenizer must be provided for GPTDataset initialization.")
        
        self.train_seeds = set()
        if self.use_seed:
            if not self.str_to_seed:
                raise ValueError("Warning: use_seed is True, but the str_to_seed map is empty or None.")
            else:
                print("Pre-calculating all training seeds...")
                # This one-time loop is necessary to have the full set available.
                for text in self.strings:
                    seed_text = self.str_to_seed.get(text)
                    if seed_text:
                        self.train_seeds.add(seed_text)

        if self.use_seed and not self.str_to_seed:
            print("Warning: use_seed is True, but the str_to_seed map is empty or None.")

    def __len__(self):
        """Returns the total number of strings in the dataset."""
        return len(self.strings)

    def __getitem__(self, idx):
        """
        Processes a single sample, tokenizes context and completion separately,
        and correctly masks the labels for the context part.
        """
        # 1. Separate the context from the completion
        completion_text = self.strings[idx]
        prompt_text = self.prompts[idx] if self.prompts else ""

        # Build the full context string (seed, prefix, prompt)
        seed_text = self.str_to_seed.get(completion_text, "") if self.use_seed else ""
        # Add spaces for clean separation
        context_parts = [self.tokenizer.bos_token + seed_text + self.prefix, prompt_text]
        context_text = " ".join(part for part in context_parts if part) 

        # 2. Tokenize context and completion separately to get their lengths
        context_encodings = self.tokenizer(context_text, add_special_tokens=False)
        completion_encodings = self.tokenizer(f" {completion_text}{self.tokenizer.eos_token}", add_special_tokens=False)

        context_ids = context_encodings["input_ids"]
        completion_ids = completion_encodings["input_ids"]
        
        # Store the length of the context part
        context_len = len(context_ids)

        # 3. Combine them to form the final input_ids
        # This is what the model sees
        input_ids = torch.tensor(context_ids + completion_ids)
        
        # 4. Create labels and mask the context part
        # This is what the model learns from
        labels = input_ids.clone()
        labels[:context_len] = -100  # Mask the context

        # Truncate if the combined length is too long
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            labels = labels[:self.max_length]

        # Create the attention mask for the final input_ids
        attention_mask = torch.ones_like(input_ids)
        
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }

class CustomTokenizer(PreTrainedTokenizer):
    """
    A custom tokenizer that is populated programmatically and is fully compatible
    with the Hugging Face ecosystem for saving and loading.
    """
    # This attribute tells the HF library the name of the vocabulary file.
    # It's used by `save_pretrained` and `from_pretrained`.
    vocab_files_names = {"vocab_file": "vocab.json"}

    def __init__(
        self,
        vocab_file=None, # This argument is provided by `from_pretrained`
        padding_side="left",
        **kwargs
    ):
        self.vocab_map = {}
        self.ids_to_tokens_map = {}

        default_special_tokens = {
            'pad_token': '[PAD]',
            'bos_token': '[SOS]',
            'eos_token': '[EOS]',
            'unk_token': '[UNK]'
        }
        for key, value in default_special_tokens.items():
            if key not in kwargs:
                kwargs[key] = value

        kwargs['padding_side'] = padding_side

        # If a vocab_file is provided, load the vocabulary from it.
        if vocab_file is not None:
            with open(vocab_file, "r", encoding="utf-8") as f:
                self.vocab_map = json.load(f)
            # Rebuild the inverse mapping from the loaded vocabulary
            self.ids_to_tokens_map = {v: k for k, v in self.vocab_map.items()}
        else:
            # If no vocab file, this is a new tokenizer; add default special tokens.
            self._add_tokens(list(kwargs.get(key) for key in default_special_tokens.keys()))

        # Call the parent's __init__ after the vocabulary is populated.
        super().__init__(**kwargs)

        self.padding_side = padding_side

    def add_special_tokens(self, special_tokens_dict, **kwargs) -> int:
        """
        Adds a dictionary of special tokens and ensures they are added to the vocabulary.
        This method is overridden to handle the custom vocabulary management.
        """
        # 1. First, add all new token strings to our custom vocabulary.
        tokens_to_add = []
        for key, value in special_tokens_dict.items():
            if isinstance(value, list):
                tokens_to_add.extend(v for v in value if isinstance(v, str))
            elif isinstance(value, str):
                tokens_to_add.append(value)

        # Use our internal method to add these tokens to the vocab_map.
        # This must be done BEFORE calling the parent's method.
        self._add_tokens(tokens_to_add)

        # 2. Now, call the parent's method. It will handle setting the attributes
        # like self.bos_token, self.eos_token, and will correctly find the token
        # IDs because they are now present in our vocab_map.
        return super().add_special_tokens(special_tokens_dict, **kwargs)
    
    def __len__(self):
        """
        Override the parent's __len__ method to return the size of our custom vocabulary.
        """
        return self.vocab_size
    
    @property
    def vocab_size(self) -> int:
        """Returns the size of the vocabulary."""
        return len(self.vocab_map)

    def get_vocab(self) -> Dict[str, int]:
        """Returns a copy of the vocabulary mapping."""
        return self.vocab_map.copy()

    def _add_tokens(self, new_tokens: List[str], **kwargs) -> int:
        """Adds a list of new tokens to the vocabulary."""
        added_count = 0
        for token in new_tokens:
            if isinstance(token, str) and token not in self.vocab_map:
                new_id = len(self.vocab_map)
                self.vocab_map[token] = new_id
                self.ids_to_tokens_map[new_id] = token
                added_count += 1
        return added_count

    def _tokenize(self, text: str, **kwargs) -> List[str]:
        """
        Tokenizes a string, correctly handling special tokens.
        """
        # Create a regex pattern that finds any of the special tokens.
        special_tokens_pattern = '|'.join(re.escape(str(token)) for token in self.all_special_tokens_extended)
        pattern = f"({special_tokens_pattern})"

        # Split the text by the special tokens, keeping them in the output.
        chunks = re.split(pattern, text)

        # Process the chunks: keep special tokens, and split regular text by spaces.
        final_tokens = []
        for chunk in chunks:
            if chunk in self.all_special_tokens_extended:
                final_tokens.append(chunk)
            else:
                final_tokens.extend(chunk.strip().split())

        # Filter out any empty strings that may result from the splits.
        return [token for token in final_tokens if token]

    def _convert_token_to_id(self, token: str) -> int:
        """Converts a token (str) into an ID using the vocabulary."""
        return self.vocab_map.get(token, self.vocab_map.get(self.unk_token))

    def _convert_id_to_token(self, index: int) -> str:
        """Converts an index (integer) into a token (str) using the vocabulary."""
        return self.ids_to_tokens_map.get(index)

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        """Saves the custom vocabulary to a file."""
        if not os.path.isdir(save_directory):
            os.makedirs(save_directory)

        # Use the filename defined in the `vocab_files_names` class attribute.
        vocab_file = os.path.join(save_directory, self.vocab_files_names["vocab_file"])

        with open(vocab_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.get_vocab(), ensure_ascii=False, indent=2))

        return (vocab_file,)

def collate_fn(features, tokenizer):
    """
    A simple, manual data collator that pads batches to the max length.
    This replaces DataCollatorWithPadding to bypass potential library bugs.
    
    Args:
        features (list[dict]): A list of dictionary samples from the dataset.
        tokenizer: The tokenizer instance.
    """
    # First, determine the maximum sequence length in the batch
    max_length = max(len(feature["input_ids"]) for feature in features)
    
    # Get the padding values from the tokenizer
    pad_token_id = tokenizer.pad_token_id
    
    # Initialize lists to hold the padded tensors
    padded_input_ids = []
    padded_attention_masks = []
    padded_labels = []
    
    # Manually pad each sample in the batch
    for feature in features:
        # How much padding is needed for this specific sample
        padding_needed = max_length - len(feature["input_ids"])
        
        # Pad 'input_ids'
        padded_ids = torch.cat([
            feature["input_ids"],
            torch.tensor([pad_token_id] * padding_needed, dtype=torch.long)
        ])
        padded_input_ids.append(padded_ids)
        
        # Pad 'attention_mask'
        padded_mask = torch.cat([
            feature["attention_mask"],
            torch.tensor([0] * padding_needed, dtype=torch.long)
        ])
        padded_attention_masks.append(padded_mask)
        
        # Pad 'labels'
        padded_label = torch.cat([
            feature["labels"],
            torch.tensor([-100] * padding_needed, dtype=torch.long)
        ])
        padded_labels.append(padded_label)
        
    # Stack the lists of padded tensors into a final batch tensor
    batch = {
        "input_ids": torch.stack(padded_input_ids),
        "labels": torch.stack(padded_labels),
        "attention_mask": torch.stack(padded_attention_masks),
    }
    
    return batch

# class CustomTokenizer:
#     def __init__(self, vocab_list, bos_token="<BOS>", eos_token="<EOS>"):
#         # Build vocab
#         self.token_to_id = {token: idx for idx, token in enumerate(vocab_list)}
#         self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}
        
#         # Set special tokens
#         self.bos_token = bos_token
#         self.eos_token = eos_token
#         self.pad_token = eos_token  # Default pad token

#         self.bos_token_id = self.token_to_id[bos_token]
#         self.eos_token_id = self.token_to_id[eos_token]
#         self.pad_token_id = self.eos_token_id       # Use eos_id if pad not in vocab
        
#         # User must explicitly manage special tokens
#         self.all_special_tokens = [bos_token, eos_token]
#         self.all_special_ids = [self.bos_token_id, self.eos_token_id]
    
#     def add_special_tokens(self, special_tokens_dict):
#         if 'bos_token' in special_tokens_dict:
#             bos_token = special_tokens_dict['bos_token']
#             if bos_token not in self.token_to_id:
#                 idx = len(self.token_to_id)
#                 self.token_to_id[bos_token] = idx
#                 self.id_to_token[idx] = bos_token
#             self.bos_token = bos_token
#             self.bos_token_id = self.token_to_id[bos_token]

#         if 'eos_token' in special_tokens_dict:
#             eos_token = special_tokens_dict['eos_token']
#             if eos_token not in self.token_to_id:
#                 idx = len(self.token_to_id)
#                 self.token_to_id[eos_token] = idx
#                 self.id_to_token[idx] = eos_token
#             self.eos_token = eos_token
#             self.eos_token_id = self.token_to_id[eos_token]

#         if 'pad_token' in special_tokens_dict:
#             pad_token = special_tokens_dict['pad_token']
#             if pad_token not in self.token_to_id:
#                 idx = len(self.token_to_id)
#                 self.token_to_id[pad_token] = idx
#                 self.id_to_token[idx] = pad_token
#             self.pad_token = pad_token
#             self.pad_token_id = self.token_to_id[pad_token]

#         if 'additional_special_tokens' in special_tokens_dict:
#             for tok in special_tokens_dict['additional_special_tokens']:
#                 if tok not in self.token_to_id:
#                     idx = len(self.token_to_id)
#                     self.token_to_id[tok] = idx
#                     self.id_to_token[idx] = tok

#         # Recompute special tokens list (explicit — user-specified only)
#         self.all_special_tokens = []
#         if hasattr(self, 'bos_token'):
#             self.all_special_tokens.append(self.bos_token)
#         if hasattr(self, 'eos_token'):
#             self.all_special_tokens.append(self.eos_token)
#         if 'additional_special_tokens' in special_tokens_dict:
#             self.all_special_tokens += special_tokens_dict['additional_special_tokens']

#         self.all_special_ids = [self.token_to_id[tok] for tok in self.all_special_tokens]
    
#     def encode(self, text):
#         tokens = text.strip().split(" ")
#         return [self.token_to_id[token] for token in tokens]
    
#     def decode(self, token_ids):
#         return " ".join([self.id_to_token[token_id] for token_id in token_ids])
