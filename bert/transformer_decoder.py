import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Seq2SeqDecoder(nn.Module):
    def __init__(self, dictionary):
        super().__init__()
        self.dictionary = dictionary

    def forward(self, tgt_inputs, encoder_out):
        raise NotImplementedError

    def reorder_incremental_state(self, incremental_state, new_order):
        """Reorder incremental state. This should be called when the order of the input has changed from the previous
        time step. A typical use case is beam search, where the input order changes between time steps based on the
        selection of beams.
        """
        def apply_reorder_incremental_state(module):
            if module != self and hasattr(module, 'reorder_incremental_state'):
                module.reorder_incremental_state(incremental_state, new_order)
        self.apply(apply_reorder_incremental_state)

class TransformerDecoder(Seq2SeqDecoder):
    def __init__(self, args, embeddings):
        super().__init__(dictionary)
        
        self.embeddings = embeddings
        
        self.layers = nn.ModuleList([TransformerDecoderLayer(args) for i in range(args.decoder_layers)])
        self.layer_norm = nn.LayerNorm(embed_dim)

        self.embed_out = nn.Parameter(torch.Tensor(len(dictionary), embed_dim))
        nn.init.normal_(self.embed_out, mean=0, std=embed_dim ** -0.5)

    def forward(self, tgt_input_ids, tgt_token_type_ids, tgt_attention_mask, encoder_out, encoder_padding_mask, incremental_state=None):
        
        if tgt_attention_mask is None:
            tgt_attention_mask = torch.ones_like(tgt_input_ids)
        if tgt_token_type_ids is None:
            tgt_token_type_ids = torch.zeros_like(tgt_input_ids)

        x = self.embeddings(tgt_input_ids, tgt_token_type_ids)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        attn = None
        inner_states = [x]

        # Decoder layers
        for layer in self.layers:
            x, attn = layer(
                x, encoder_out, encoder_padding_mask,
                incremental_state, self_attn_mask=self.buffered_future_mask(x) if incremental_state is None else None)
            inner_states.append(x)
        x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)
        x = F.linear(x, self.embed_out)

        return x, {'attn': attn, 'inner_states': inner_states}

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        if not hasattr(self, '_future_mask') or self._future_mask is None or self._future_mask.device != tensor.device:
            self._future_mask = torch.triu(utils.fill_with_neg_inf(tensor.new(dim, dim)), 1)
        if self._future_mask.size(0) < dim:
            self._future_mask = torch.triu(utils.fill_with_neg_inf(self._future_mask.resize_(dim, dim)), 1)
        return self._future_mask[:dim, :dim]

class TransformerDecoderLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.embed_dim = args.decoder_embed_dim
        self.dropout = args.dropout
        self.relu_dropout = args.relu_dropout

        self.self_attn = MultiheadAttention(self.embed_dim, args.decoder_attention_heads, args.attention_dropout)
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.encoder_attn = MultiheadAttention(self.embed_dim, args.decoder_attention_heads, args.attention_dropout)
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, args.decoder_ffn_embed_dim)
        self.fc2 = nn.Linear(args.decoder_ffn_embed_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.constant_(self.fc1.bias, 0.)
        nn.init.constant_(self.fc2.bias, 0.)

    def forward(
        self, x, encoder_out, encoder_padding_mask, incremental_state, prev_self_attn_state=None,
        prev_attn_state=None, self_attn_mask=None, self_attn_padding_mask=None,
    ):
        residual = x
        x = self.self_attn_layer_norm(x)
        if prev_self_attn_state is not None:
            if incremental_state is None:
                incremental_state = {}
            prev_key, prev_value = prev_self_attn_state
            saved_state = {'prev_key': prev_key, 'prev_value': prev_value}
            self.self_attn._set_input_buffer(incremental_state, saved_state)

        x, _ = self.self_attn(
            query=x, key=x, value=x, key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state, need_weights=False, attn_mask=self_attn_mask)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x

        attn = None
        residual = x
        x = self.encoder_attn_layer_norm(x)
        if prev_attn_state is not None:
            if incremental_state is None:
                incremental_state = {}
            prev_key, prev_value = prev_attn_state
            saved_state = {'prev_key': prev_key, 'prev_value': prev_value}
            self.encoder_attn._set_input_buffer(incremental_state, saved_state)
        x, attn = self.encoder_attn(
            query=x, key=encoder_out, value=encoder_out, key_padding_mask=encoder_padding_mask,
            incremental_state=incremental_state, static_kv=True, need_weights=(not self.training))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x

        residual = x
        x = self.final_layer_norm(x)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.relu_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.final_layer_norm(x)
        return x, attn


