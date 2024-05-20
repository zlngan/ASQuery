import torch
from torch import nn
from torch.nn import functional as F

from .blocks import LayerNorm, get_sinusoid_encoding

import copy

class ActionHead(nn.Module):
    def __init__(self,
                n_embd: int,
                max_seq_len: int,
                num_classes: int,
                n_head: int,
                n_dec_layers: int,
                n_blocks: int,
                attn_pdrop=0.0,        # dropout rate for the attention map
                proj_pdrop=0.0,        # dropout rate for the projection / MLP
                path_pdrop=0.0,        # drop path rate
                with_feat_pos=False,
                with_query_pos=False,
                with_aux_outputs=False,
                with_aux_boundary=False,
                normalize_before=False,
                decoder_type='transformer',
                ):
        super().__init__()

        self.num_classes = num_classes
        self.n_dec_layers = n_dec_layers
        self.n_blocks = n_blocks
        self.n_head = n_head
        self.max_seq_len = max_seq_len
        self.with_aux_outputs = with_aux_outputs
        self.with_aux_boundary = with_aux_boundary
        self.decocer_type = decoder_type


        self.feat_pe = []
        for i in range(self.n_blocks):
            pos_embd = get_sinusoid_encoding(int(self.max_seq_len/(2**i)), n_embd) / (n_embd**0.5)
            self.register_buffer(f"feat_pe_{i}", pos_embd, persistent=False)
            self.feat_pe.append(pos_embd)

        self.query = nn.Embedding(num_classes+1, n_embd)
        self.query_pe = nn.Embedding(num_classes+1, n_embd)
        self.level_embd = nn.Embedding(n_blocks, n_embd)
        
        if decoder_type == 'transformer':
            self.decoder_layer = TransformerdecoderLayer(
                    n_embd = n_embd,
                    n_head = n_head,
                    n_blocks = n_blocks,
                    attn_pdrop = attn_pdrop,
                    proj_pdrop = proj_pdrop,
                    path_pdrop = path_pdrop,
                    with_query_pos = with_query_pos,
                    with_feat_pos = with_feat_pos,
                    normalize_before = normalize_before,
                    )

        self.decoder_layers = nn.ModuleList([copy.deepcopy(self.decoder_layer) for _ in range(n_dec_layers)])
            
        self.mask_embd = nn.Sequential(
                        nn.Conv1d(n_embd, n_embd//4, 1), nn.ReLU(inplace=True), 
                        nn.Conv1d(n_embd//4, n_embd, 1), nn.ReLU(inplace=True),
                        nn.Conv1d(n_embd, n_embd, 1)
                        )
        self.boundary_embd = nn.Sequential(
                        nn.Linear(n_embd, n_embd//4), nn.ReLU(inplace=True),
                        nn.Linear(n_embd//4, n_embd), nn.ReLU(inplace=True),
                        nn.Linear(n_embd, n_embd)
                        )

    def forward(self, mask_feat, feats, masks):
        masks = [x.unsqueeze(1) for x in masks]
        lvl_feats = []
        ### add level pe
        for i in range(self.n_blocks):
            lvl_feats.append(feats[i] + self.level_embd.weight[i][None, :, None])

        B = feats[0].size()[0]
        query = self.query.weight.transpose(0,1).unsqueeze(0).repeat(B,1,1) 
        query_pe = self.query_pe.weight.transpose(0,1).unsqueeze(0).repeat(B,1,1) 

        pred_mask_all = []
        pred_boundary_all = []

        for decoder_leyer in self.decoder_layers:
            pred_mask, pred_boundary = decoder_leyer\
                    (query, feats, masks, query_pe, self.feat_pe, mask_feat, self.forward_pred_mask, self.forward_pred_boundary)
            pred_mask_all += pred_mask
            pred_boundary_all += pred_boundary

        out_pred_mask = pred_mask_all if self.with_aux_outputs else pred_mask_all[-1]
        out_pred_boundary = pred_boundary_all if self.with_aux_boundary else pred_boundary_all[-1]        


        return out_pred_mask, out_pred_boundary


    def forward_pred_mask(self, action_query, mask_feat):    
        mask_embd = self.mask_embd(action_query)
        output_mask = torch.einsum('bcn, bcl -> bnl', mask_embd, mask_feat)

        return output_mask
    

    def forward_pred_boundary(self, boundary_query, mask_feat):
        boundary_embd = self.boundary_embd(boundary_query)
        boundary = torch.einsum('bc, bcl -> bl', boundary_embd, mask_feat)

        return boundary


class TransformerdecoderLayer(nn.Module):
    def __init__(self,
            n_embd,                # dimension of the input features
            n_head,               # number of attention heads
            n_blocks,
            n_hidden=None,         # dimension of the hidden layer in MLP
            activation="relu",     # nonlinear activation used in MLP, default GELU
            attn_pdrop=0.0,        # dropout rate for the attention map
            proj_pdrop=0.0,        # dropout rate for the projection / MLP
            path_pdrop=0.0,        # drop path rate
            with_query_pos=False,
            with_feat_pos=False,
            normalize_before=False,     
    ):
        super().__init__()
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_blocks = n_blocks
        self.with_query_pos = with_query_pos
        self.with_feat_pos = with_feat_pos
        n_hidden = n_embd * 4 if n_hidden is None else n_hidden

        self.cross_attn = CrossAttentionLayer(n_embd, n_head, attn_pdrop, path_pdrop, activation, with_query_pos, with_feat_pos, normalize_before)
        self.self_attn = SelfAttentionLayer(n_embd, n_head, attn_pdrop, path_pdrop, activation, with_query_pos, normalize_before)
        self.ffn = FFNLayer(n_embd, n_hidden, proj_pdrop, activation, normalize_before)

        self.cross_attn_layers = nn.ModuleList([copy.deepcopy(self.cross_attn) for _ in range(n_blocks)])
        self.self_attn_layers = nn.ModuleList([copy.deepcopy(self.self_attn) for _ in range(n_blocks)])
        self.ffn_layers = nn.ModuleList([copy.deepcopy(self.ffn) for _ in range(n_blocks)])


    def forward(self, query, feats, masks, query_pe, feat_pe, mask_feat, mask_func, boundary_func):
        pred_mask = []
        pred_boundary = []

        output_mask = mask_func(query[...,:-1], mask_feat)
        pred_mask.append(output_mask)

        output_boundary = boundary_func(query[...,-1], mask_feat)
        pred_boundary.append(output_boundary)

        for i in range(self.n_blocks-1, -1, -1):
            query = self.self_attn_layers[i](query, tgt_mask=None, tgt_key_padding_mask=None, query_pos=query_pe)
            query = self.cross_attn_layers[i](query, feats[i], None, ~masks[i].squeeze(1), feat_pe[i], query_pe)
            query = self.ffn_layers[i](query)

            output_mask = mask_func(query[...,:-1], mask_feat)
            pred_mask.append(output_mask)

            output_boundary = boundary_func(query[...,-1], mask_feat)
            pred_boundary.append(output_boundary)

        return pred_mask, pred_boundary


class CrossAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead, attn_pdrop=0.0, path_pdrop=0.0, activation="relu", with_query_pos=False, with_feat_pos=False, normalize_before=False):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=attn_pdrop, batch_first=True)
        self.norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(path_pdrop)
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self.with_query_pos = with_query_pos
        self.with_feat_pos = with_feat_pos

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos=None):
        return tensor + pos.to(tensor.device) if pos is not None else tensor
    
    def forward_post(self, tgt, memory,
                    memory_mask = None,
                    memory_key_padding_mask = None,
                    pos = None,
                    query_pos = None):
        query = self.with_pos_embed(tgt, query_pos) if self.with_query_pos else tgt
        key = self.with_pos_embed(memory, pos) if self.with_feat_pos else memory
        query = query.transpose(1,2)
        key = key.transpose(1,2)
        memory = memory.transpose(1,2)
        tgt2 = self.multihead_attn(query=query,
                                    key=key,
                                    value=memory, attn_mask=memory_mask,
                                    key_padding_mask=memory_key_padding_mask)[0]
        tgt2 = tgt2.transpose(1,2)
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        
        return tgt

    def forward_pre(self, tgt, memory,
                    memory_mask = None,
                    memory_key_padding_mask = None,
                    pos= None,
                    query_pos = None):
        tgt2 = self.norm(tgt)
        query = self.with_pos_embed(tgt2, query_pos) if self.with_query_pos else tgt2
        key = self.with_pos_embed(memory, pos) if self.with_feat_pos else memory
        query = query.transpose(1,2)
        key = key.transpose(1,2)
        memory = memory.transpose(1,2)
        tgt2 = self.multihead_attn(query=query,
                                    key=key,
                                    value=memory, attn_mask=memory_mask,
                                    key_padding_mask=memory_key_padding_mask)[0]
        tgt2 = tgt2.transpose(1,2)
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt, memory,
                memory_mask = None,
                memory_key_padding_mask = None,
                pos = None,
                query_pos = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, memory_mask,
                                    memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, memory_mask,
                                 memory_key_padding_mask, pos, query_pos)


class SelfAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead, attn_pdrop=0.0, path_pdrop=0.0,
                 activation="relu",  with_query_pos=False, normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=attn_pdrop, batch_first=True)
        self.norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(path_pdrop)
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self.with_query_pos = with_query_pos

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    def with_pos_embed(self, tensor, pos=None):
        pos_len = pos.shape[-1]
        if pos is None:
            return tensor
        if pos_len == tensor.shape[-1]:
            return tensor + pos.to(tensor.device)
        else:
            tensor[:,:,:pos_len] += pos
            return tensor

    def forward_post(self, tgt,
                     tgt_mask = None,
                     tgt_key_padding_mask = None,
                     query_pos = None):
        q = k = self.with_pos_embed(tgt, query_pos) if self.with_query_pos else tgt
        q = k = q.transpose(1,2)
        tgt2 = self.self_attn(query=q, key=k, value=tgt.transpose(1,2), attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2.transpose(1,2))
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt,
                    tgt_mask = None,
                    tgt_key_padding_mask = None,
                    query_pos = None):
        tgt2 = self.norm(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos) if self.with_query_pos else tgt2
        q = k = q.transpose(1,2)
        tgt2 = self.self_attn(q, k, value=tgt2.transpose(1,2), attn_mask=tgt_mask,
                                key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2.transpose(1,2))
        
        return tgt

    def forward(self, tgt,
                tgt_mask = None,
                tgt_key_padding_mask = None,
                query_pos = None):
        if self.normalize_before:
            return self.forward_pre(tgt, tgt_mask,
                                    tgt_key_padding_mask, query_pos)
        return self.forward_post(tgt, tgt_mask,
                                 tgt_key_padding_mask, query_pos)


class FFNLayer(nn.Module):
    def __init__(self, d_model, dim_feedforward=2048, path_pdrop=0.0,
                activation="relu", normalize_before=False):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(path_pdrop)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos=None):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt):
        tgt = tgt.transpose(1,2)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = tgt.transpose(1,2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = tgt2.transpose(1,2)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2.transpose(1,2))
        return tgt
    
    def forward(self, tgt):
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")



class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
