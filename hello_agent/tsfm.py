import math
import torch
import torch.nn.functional as F
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self,d_model, max_len=5000,vocab_size=50257):
        super(PositionalEncoding, self).__init__()
        pos_embed = torch.arange(max_len).unsqueeze(1)
        self.matrix = torch.zeros(max_len,d_model)
        div_term = torch.exp(torch.arange(0,d_model,2) * (-math.log(10000.0) / d_model))
        self.matrix[:,0::2] = torch.sin(pos_embed*div_term)
        self.matrix[:,1::2] = torch.cos(pos_embed*div_term)
        self.matrix = self.matrix.unsqueeze(0)
        self.register_buffer("pe", self.matrix)
        self.embedding = nn.Embedding(num_embeddings=vocab_size,embedding_dim=d_model)
    def forward(self, x):
        length = x.size(1)
        x = self.embedding(x)
        x = x + self.pe[:, :length,:]
        return x




class MultiHeadAttention(nn.Module):
    def __init__(self,d_model,n_head):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_head == 0, "维度必须整除头数"
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_out = nn.Linear(d_model, d_model)

    def forward(self,query,key,value,attention_mask=None,padding_mask=None):
        bsz, tgt_len, d_model = query.size()
        _, src_len, n_ = key.size()

        Q = self.w_q(query).view(bsz, tgt_len,self.n_head, self.head_dim).transpose(1, 2)
        K = self.w_k(key).view(bsz, src_len, self.n_head, self.head_dim).transpose(1, 2)
        V = self.w_v(value).view(bsz, src_len, self.n_head, self.head_dim).transpose(1, 2)

        score = torch.matmul(Q, K.transpose(-1,-2)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            score = score.masked_fill(attention_mask,float("-inf"))
        if padding_mask is not None:
            score +=  score.masked_fill(padding_mask,float("-inf"))
        attn_score = F.softmax(score, dim=-1)
        output = torch.matmul(attn_score, V).transpose(1,2).contiguous().view(bsz, tgt_len, self.d_model)
        output = self.w_out(output)
        return output, attn_score




class PositionWiseFeedForward(nn.Module):
    def __init__(self,d_model,d_ff,dropout=0.1):
        super(PositionWiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()


    def forward(self,x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_head, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, n_head)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
    def forward(self, x, src_padding_mask=None):
        attn_output,attn = self.self_attn(x,x,x,attention_mask=None,padding_mask=src_padding_mask)
        x = self.layer_norm1(x + self.dropout(attn_output))

        ff_output = self.feed_forward(x)
        x = self.layer_norm2(x + self.dropout(ff_output))
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_head, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.masked_self_attn = MultiHeadAttention(d_model, n_head)
        self.cross_attn = MultiHeadAttention(d_model, n_head)
        self.feed_forward = PositionWiseFeedForward(d_model,d_ff)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.layer_norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value,attention_mask=None,src_padding_mask=None,tgt_padding_mask=None):
        self_attn_output,attn1 = self.masked_self_attn(query,query,query,attention_mask=attention_mask,padding_mask=tgt_padding_mask)
        query = self.layer_norm1(query + self.dropout(self_attn_output))

        cross_attn_output,attn2 = self.cross_attn(query,key,value,padding_mask=src_padding_mask)
        output = self.layer_norm2(query + self.dropout(cross_attn_output))

        ff_output = self.feed_forward(output)
        output = self.layer_norm3(output + self.dropout(ff_output))

        return output

class transformer(nn.Module):
    def __init__(self,d_model,n_head,d_ff,dropout=0.1):
        super(transformer,self).__init__()
        self.encoder = EncoderLayer(d_model,n_head,d_ff,dropout)
        self.decoder = DecoderLayer(d_model,n_head,d_ff,dropout)
        self.position = PositionalEncoding(d_model)
    def forward(self,src,tgt,src_padding_mask=None,tgt_padding_mask=None):
        src = self.position(src)
        tgt = self.position(tgt)
        attention_mask = torch.triu(torch.ones(tgt.shape[1],tgt.shape[1]),diagonal=1).to(torch.bool)

        encoder_output = self.encoder(src,src_padding_mask)
        decoder_output = self.decoder(
            tgt,
            encoder_output,
            encoder_output,
            attention_mask=attention_mask,
            src_padding_mask=src_padding_mask,
            tgt_padding_mask=tgt_padding_mask
        )
        return decoder_output


if __name__ == "__main__":
    tgt = torch.tensor([[1,2,3,4],[5,6,7,8]])
    tgt_padding_mask = torch.tensor([[False,False,True,True],[False,False,False,True]])[:,None,None,:]
    src = torch.tensor([[11,22,33],[44,55,66]])
    src_padding_mask = torch.tensor([[False,False,True],[False,False,False]])[:,None,None,:]

    model = transformer(32,4,64)

    output  = model(src,tgt,src_padding_mask,tgt_padding_mask)
    pass
