import torch
import torch.nn as nn
import math
from rotary_embedding_torch import RotaryEmbedding

LEN = 457
LEN_EOS = 459
LEN_FOR_GENERALIZATION = 722

############################################################
# the code for building transformer (building blocks) is from
# https://towardsdatascience.com/build-your-own-transformer-from-scratch-using-pytorch-84c850470dcb

# the way how sinusoidal embedding is calculated is from https://www.kaggle.com/code/iafoss/rna-starter-0-186-lb#Model
class PosEnc(nn.Module):
    """
    sinusoidal embeddings
    """
    def __init__(self, dim=192, M=10000, num_tokens=LEN_EOS):
        super().__init__()
        positions = torch.arange(num_tokens).unsqueeze(0)
        half_dim = dim // 2
        emb = math.log(M) / half_dim
        emb = torch.exp(torch.arange(half_dim) * (-emb))
        emb = positions[..., None] * emb[None, ...]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        self.pos = emb

    def forward(self, x):
        device = x.device
        pos = self.pos.to(device)
        res = x + pos
        return res


# https://towardsdatascience.com/build-your-own-transformer-from-scratch-using-pytorch-84c850470dcb
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            _MASKING_VALUE = -1e+30 if attn_scores.dtype == torch.float32 else -1e+4
            attn_scores = attn_scores.masked_fill(mask == 0, _MASKING_VALUE)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output

    def split_heads(self, x):
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))
        return output


class AttentionRotary(nn.Module):
    def __init__(self, d_model, num_heads, rotary_emb):
        super(AttentionRotary, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.rotary_emb = rotary_emb

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            _MASKING_VALUE = -1e+30 if attn_scores.dtype == torch.float32 else -1e+4
            attn_scores = attn_scores.masked_fill(mask == 0, _MASKING_VALUE)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output

    def split_heads(self, x):
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        Q = self.rotary_emb.rotate_queries_or_keys(Q)
        K = self.split_heads(self.W_k(K))
        K = self.rotary_emb.rotate_queries_or_keys(K)
        V = self.split_heads(self.W_v(V))

        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))
        return output


class CustomAttentionBPP(nn.Module):
    def __init__(self, d_model, num_heads=1):
        super(CustomAttentionBPP, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, bpp, V, mask=None):
        attn_scores = bpp.unsqueeze(1)
        _MASKING_VALUE = -1e+30 if attn_scores.dtype == torch.float32 else -1e+4
        attn_scores = attn_scores.masked_fill(attn_scores == 0, _MASKING_VALUE)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, _MASKING_VALUE)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output

    def split_heads(self, x):
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

    def forward(self, bpp, V, mask=None):
        V = self.split_heads(self.W_v(V))

        attn_output = self.scaled_dot_product_attention(bpp, V, mask)
        output = self.W_o(self.combine_heads(attn_output))
        return output


# https://towardsdatascience.com/build-your-own-transformer-from-scratch-using-pytorch-84c850470dcb
# gelu is used instead of relu
class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.gelu = nn.GELU()

    def forward(self, x):
        return self.fc2(self.gelu(self.fc1(x)))


# https://towardsdatascience.com/build-your-own-transformer-from-scratch-using-pytorch-84c850470dcb
# with minor modifications
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x


class EncoderLayerRotary(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout, rotary_emb):
        super(EncoderLayerRotary, self).__init__()
        self.self_attn = AttentionRotary(d_model, num_heads, rotary_emb)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x


# https://towardsdatascience.com/build-your-own-transformer-from-scratch-using-pytorch-84c850470dcb
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask, tgt_mask):
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x


class DecoderLayerRotary(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout, rotary_emb):
        super(DecoderLayerRotary, self).__init__()
        self.self_attn = AttentionRotary(d_model, num_heads, rotary_emb)
        self.cross_attn = AttentionRotary(d_model, num_heads, rotary_emb)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask, tgt_mask):
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x


# similar to DecoderLayer, but as cross_attn, it uses CustomAttentionBPP
class DecoderLayerTwo(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayerTwo, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = CustomAttentionBPP(d_model)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, bpp, mask):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        attn_output = self.cross_attn(bpp=bpp, V=x, mask=mask)
        x = self.norm2(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x

#########################################################
# models:


# first it is decoder layer to use bpp (with sinusoidal pos embeds), then uses rotary embeddings
# tgt, info1: seq_inds; info2: bpp; src or info3: struct_inds
class ModelThirtyTwo(nn.Module):
    def __init__(self, tgt_vocab=7, src_vocab=6, d_model=192, num_heads=6, num_layers=8,
                 d_ff=(192*4), dropout=0.1, num_tokens=LEN_EOS):
        super(ModelThirtyTwo, self).__init__()
        self.tgt_embedding = nn.Embedding(tgt_vocab, d_model)
        self.src_embedding = nn.Embedding(src_vocab, d_model)
        self.positional_enc = PosEnc(dim=d_model, num_tokens=num_tokens)
        self.rotary = RotaryEmbedding(dim=d_model//num_heads)
        self.decoder_one = DecoderLayerTwo(d_model, num_heads, d_ff, dropout)
        self.decoder = DecoderLayerRotary(d_model, num_heads, d_ff, dropout, self.rotary)
        self.encoder_layers = nn.ModuleList([EncoderLayerRotary(d_model, num_heads, d_ff, dropout, self.rotary) for _ in range(num_layers)])
        self.fc = nn.Linear(d_model, 2)

    def forward(self, data):
        tgt = data['info1']
        bpp = data['info2']
        src = data['info3']
        mask = data['mask']

        mask = mask.unsqueeze(1).unsqueeze(2)
        src = self.src_embedding(src)
        tgt = self.positional_enc(self.tgt_embedding(tgt))

        output = self.decoder_one(x=tgt, bpp=bpp, mask=mask)

        output = self.decoder(x=output, enc_output=src, src_mask=mask, tgt_mask=mask)
        for enc_layer in self.encoder_layers:
            output = enc_layer(output, mask)

        output = self.fc(output)
        return output


class ModelThirtyNine(nn.Module):
    def __init__(self, tgt_vocab=7, src_vocab=6, d_model=384, num_heads=6, num_layers=8, d_ff=384, dropout=0.1):
        super(ModelThirtyNine, self).__init__()
        self.tgt_embedding = nn.Embedding(tgt_vocab, d_model)
        self.src_embedding = nn.Embedding(src_vocab, d_model)
        self.positional_enc = RotaryEmbedding(dim=d_model//num_heads)
        self.decoder = DecoderLayerRotary(d_model, num_heads, d_ff, dropout, self.positional_enc)
        self.encoder_layers = nn.ModuleList([EncoderLayerRotary(d_model, num_heads, d_ff, dropout, self.positional_enc) for _ in range(num_layers)])
        self.fc = nn.Linear(d_model, 2)

    def forward(self, data):
        tgt = data['info1']
        src = data['info2']
        mask = data['mask']

        mask = mask.unsqueeze(1).unsqueeze(2)
        tgt = self.tgt_embedding(tgt)
        src = self.src_embedding(src)

        output = self.decoder(x=tgt, enc_output=src, src_mask=mask, tgt_mask=mask)
        for enc_layer in self.encoder_layers:
            output = enc_layer(output, mask)

        output = self.fc(output)
        return output



