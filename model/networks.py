# Import from Hugging Face transformers library
import torch
import torch.nn as nn

from transformers import PreTrainedModel, GPT2Config, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
from typing import Optional, Tuple, Union
# --- 1. Custom Model Architecture using Hugging Face ---

class AttentionOnlyBlock(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = torch.nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = torch.nn.MultiheadAttention(
            embed_dim=config.n_embd,
            num_heads=config.n_head,
            dropout=config.attn_pdrop,
            batch_first=True
        )
        self.dropout = torch.nn.Dropout(config.resid_pdrop)

    def forward(self, x, attention_mask=None, output_attentions=False, **kwargs):
        residual = x
        x = self.ln_1(x)
        # The final, combined mask is passed here.
        attn_output, attn_weights = self.attn(
            x, x, x, 
            attn_mask=attention_mask, 
            need_weights=output_attentions,
            average_attn_weights=False  # gives attn_weights shape (batch_size, num_heads, seq_len, seq_len)
        )
        x = residual + self.dropout(attn_output)
        
        outputs = (x,)
        if output_attentions:
            outputs += (attn_weights,)
        return outputs

# class AttentionOnlyGPT2Model(PreTrainedModel):
#     config_class = GPT2Config
#     def __init__(self, config):
#         super().__init__(config)
#         self.wte = torch.nn.Embedding(config.vocab_size, config.n_embd)
#         self.wpe = torch.nn.Embedding(config.n_positions, config.n_embd)
#         self.drop = torch.nn.Dropout(config.embd_pdrop)
#         self.h = torch.nn.ModuleList([AttentionOnlyBlock(config) for _ in range(config.n_layer)])
#         self.ln_f = torch.nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
#         # Ensure the config knows to output attentions if requested during generation
#         self.config.output_attentions = True

#     def forward(self, input_ids=None, attention_mask=None, output_attentions=None, past_key_values=None, **kwargs):
#         # Determine if attentions should be returned
#         output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        
#         # --- 1. EMBEDDING LAYER ---
#         input_embeds = self.wte(input_ids)
#         position_ids = torch.arange(0, input_ids.size(-1), dtype=torch.long, device=input_ids.device).unsqueeze(0)
#         position_embeds = self.wpe(position_ids)
#         hidden_states = input_embeds + position_embeds
#         hidden_states = self.drop(hidden_states)
        
#         # --- 2. CORRECTED MASK CREATION FOR MHA ---
#         batch_size, seq_length = input_ids.shape
#         # Get the number of heads from one of the attention blocks
#         num_heads = self.h[0].attn.num_heads

#         # Create the causal (look-ahead) mask of shape (seq_length, seq_length)
#         # True values will be masked out.
#         causal_mask = ~torch.tril(torch.ones((seq_length, seq_length), device=input_ids.device, dtype=torch.bool))

#         # Combine with the padding mask if it exists
#         if attention_mask is not None:
#             # attention_mask is (batch, seq_len). 0 for padding.
#             padding_mask_bool = (attention_mask == 0).unsqueeze(1)  # Shape: (batch_size, 1, seq_len)
#             # Combine with causal mask via broadcasting
#             final_3d_mask = causal_mask.unsqueeze(0) | padding_mask_bool # Shape: (batch_size, seq_len, seq_len)
#         else:
#             # If no padding, just add a batch dimension to the causal mask
#             # final_3d_mask = causal_mask.expand(batch_size, seq_length, seq_length)
#             final_3d_mask = causal_mask.unsqueeze(0)

#         # Expand the 3D mask to repeat for each head, then reshape to the format MHA expects.
#         # The error message confirms the target shape is (batch_size * num_heads, seq_len, seq_len).
#         expanded_mask = final_3d_mask.unsqueeze(1).repeat(1, num_heads, 1, 1)
#         final_mask_for_attn = expanded_mask.view(batch_size * num_heads, seq_length, seq_length)

#         # --- 3. TRANSFORMER BLOCKS ---
#         all_attentions = [] if output_attentions else None
        
#         for block in self.h:
#             # Pass the correctly shaped mask to the attention block
#             outputs = block(hidden_states, attention_mask=final_mask_for_attn, output_attentions=output_attentions)
#             hidden_states = outputs[0]
#             if output_attentions:
#                 all_attentions.append(outputs[1])
                
#         hidden_states = self.ln_f(hidden_states)
        
#         # --- 4. RETURN OUTPUT ---
#         # Return a tuple of tensors for attentions to match Hugging Face convention
#         attentions_tuple = tuple(all_attentions) if output_attentions and all_attentions else None
        
#         return {
#             "last_hidden_state": hidden_states, 
#             "attentions": attentions_tuple,
#             "past_key_values": None # Not implemented in this simplified model
#         }


class AttentionOnlyGPT2Model(PreTrainedModel):
    config_class = GPT2Config
    def __init__(self, config):
        super().__init__(config)
        self.wte = torch.nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = torch.nn.Embedding(config.n_positions, config.n_embd)
        self.drop = torch.nn.Dropout(config.embd_pdrop)
        self.h = torch.nn.ModuleList([AttentionOnlyBlock(config) for _ in range(config.n_layer)])
        self.ln_f = torch.nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

    def forward(self, input_ids=None, attention_mask=None, output_attentions=None, past_key_values=None, **kwargs):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        
        input_embeds = self.wte(input_ids)
        position_ids = torch.arange(0, input_ids.size(-1), dtype=torch.long, device=input_ids.device).unsqueeze(0)
        position_embeds = self.wpe(position_ids)
        hidden_states = input_embeds + position_embeds
        hidden_states = self.drop(hidden_states)
        
        all_attentions = []
        
        # --- MASKING FIX ---
        batch_size, seq_length = input_ids.shape
        num_heads = self.h[0].attn.num_heads

        # 1. Create the causal (look-ahead) mask where True means "ignore".
        causal_mask = torch.triu(torch.ones((seq_length, seq_length), device=input_ids.device, dtype=torch.bool), diagonal=1)
        
        # 2. Combine with the padding mask if provided.
        if attention_mask is not None:
            padding_mask = (attention_mask == 0).unsqueeze(1) # Shape: (batch_size, 1, seq_len)
            final_mask = causal_mask.unsqueeze(0) | padding_mask # Shape: (batch_size, seq_len, seq_len)
        else:
            final_mask = causal_mask.unsqueeze(0).expand(batch_size, seq_length, seq_length)
        
        # 3. Repeat the mask for each attention head to match the expected shape.
        # Shape: (batch_size, num_heads, seq_len, seq_len)
        expanded_mask = final_mask.unsqueeze(1).repeat(1, num_heads, 1, 1)
        # Reshape to what the internal MHA function expects.
        # Shape: (batch_size * num_heads, seq_len, seq_len)
        final_mask_for_attn = expanded_mask.view(batch_size * num_heads, seq_length, seq_length)


        for block in self.h:
            # 4. Pass the correctly shaped mask to the attention block.
            outputs = block(hidden_states, attention_mask=final_mask_for_attn, output_attentions=output_attentions)
            hidden_states = outputs[0]
            if output_attentions:
                all_attentions.append(outputs[1])
        hidden_states = self.ln_f(hidden_states)
        
        return {
            "last_hidden_state": hidden_states, 
            "attentions": all_attentions if output_attentions else None,
            "past_key_values": None
        }

class AttentionOnlyLMHeadModel(PreTrainedModel, GenerationMixin):
    config_class = GPT2Config
    def __init__(self, config):
        super().__init__(config)
        self.transformer = AttentionOnlyGPT2Model(config)
        self.lm_head = torch.nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.main_input_name = "input_ids"
        self.tie_weights()
    
    def tie_weights(self):
        """
        Tie the weights between the input embeddings and the output embeddings.
        """
        output_embeddings = self.get_output_embeddings()
        input_embeddings = self.get_input_embeddings()

        if output_embeddings is not None and input_embeddings is not None:
            output_embeddings.weight = input_embeddings.weight

        # Also, ensure the config reflects this decision
        if hasattr(self.config, "tie_word_embeddings"):
            self.config.tie_word_embeddings = True

    def get_input_embeddings(self):
        return self.transformer.wte

    def set_input_embeddings(self, new_embeddings):
        self.transformer.wte = new_embeddings

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        """
        Required by .generate() to prepare inputs for the next step.
        """
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "attention_mask": kwargs.get("attention_mask", None)
        }
    def get_output_embeddings(self):
        """
        This method is required by the GenerationMixin to know about the output layer.
        """
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        """
        This method is required by the GenerationMixin to set a new output layer.
        """
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        
        # Pass the attention_mask down to the core model.
        transformer_outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            past_key_values=past_key_values,
        )

        # # ========== ADD THIS DEBUGGING BLOCK ==========
        # print("--- DEBUGGING TRANSFORMER OUTPUT ---")
        # print(f"Type of transformer_outputs: {type(transformer_outputs)}")
        # if hasattr(transformer_outputs, 'keys'):
        #     print(f"Keys: {transformer_outputs.keys()}")
        # print(f"Value of attentions: {transformer_outputs.get('attentions')}")
        # print("--------------------------------------")
        # # ============================================

        hidden_states = transformer_outputs["last_hidden_state"]
        logits = self.lm_head(hidden_states)
        loss = None
        if labels is not None:
            # The loss function will ignore padded tokens if the label is -100.
            # The Trainer handles this automatically when labels are padded with the tokenizer's pad_token_id.
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=transformer_outputs.get("past_key_values"),
            attentions=transformer_outputs.get("attentions"),
        )