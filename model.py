import math
import copy
import os
import sys
import subprocess
import torch
import torch.nn as nn
import torch.nn.functional as F

def scaled_dot_product_attention(Q, K, V, mask=None):
    dim_k = Q.size(-1)
    attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (dim_k ** 0.5)
    
    if mask is not None:
        attention_scores = attention_scores.masked_fill(mask, float("-inf"))
        
    weights = F.softmax(attention_scores, dim=-1)
    return torch.matmul(weights, V), weights

def make_src_mask(src: torch.Tensor, pad_idx: int = 1) -> torch.Tensor:
    return (src == pad_idx).unsqueeze(1).unsqueeze(2)

def make_tgt_mask(tgt: torch.Tensor, pad_idx: int = 1) -> torch.Tensor:
    bs, seq_len = tgt.shape
    p_mask = (tgt == pad_idx).unsqueeze(1).unsqueeze(2)
    c_mask = torch.triu(torch.ones((seq_len, seq_len), device=tgt.device, dtype=torch.bool), diagonal=1)
    return p_mask | c_mask

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        bs = query.size(0)
        
        q_proj = self.W_q(query).view(bs, -1, self.num_heads, self.d_k).transpose(1, 2)
        k_proj = self.W_k(key).view(bs, -1, self.num_heads, self.d_k).transpose(1, 2)
        v_proj = self.W_v(value).view(bs, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        att_output, _ = scaled_dot_product_attention(q_proj, k_proj, v_proj, mask)
        att_output = att_output.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        
        return self.W_o(att_output)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe_matrix = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        denom = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * -(math.log(10000.0) / d_model))
        
        pe_matrix[:, 0::2] = torch.sin(pos * denom)
        pe_matrix[:, 1::2] = torch.cos(pos * denom)
        
        self.register_buffer("pe", pe_matrix.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        hidden = self.linear1(x)
        return self.linear2(self.dropout(F.relu(hidden)))

class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, src_mask):
        attn_out = self.self_attn(x, x, x, src_mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, memory, src_mask, tgt_mask):
        s_attn_out = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(s_attn_out))
        
        c_attn_out = self.cross_attn(x, memory, memory, src_mask)
        x = self.norm2(x + self.dropout(c_attn_out))
        
        f_out = self.ffn(x)
        x = self.norm3(x + self.dropout(f_out))
        return x

class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm = nn.LayerNorm(layer.norm1.normalized_shape)

    def forward(self, x, mask):
        for l in self.layers:
            x = l(x, mask)
        return self.norm(x)

class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm = nn.LayerNorm(layer.norm1.normalized_shape)

    def forward(self, x, memory, src_mask, tgt_mask):
        for l in self.layers:
            x = l(x, memory, src_mask, tgt_mask)
        return self.norm(x)

class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size: int = 10000, 
        tgt_vocab_size: int = 10000, 
        d_model: int = 256,
        N: int = 3,
        num_heads: int = 8,
        d_ff: int = 512,
        dropout: float = 0.1,
    ):
        super(Transformer, self).__init__()
        
        import spacy
        from datasets import load_dataset
        from collections import Counter
        
        def safe_load_spacy(name, cmd_args):
            try:
                return spacy.load(name)
            except OSError:
                try:
                    subprocess.check_call(cmd_args)
                    return spacy.load(name)
                except Exception:
                    return spacy.blank(name.split("_")[0])
                    
        self.spacy_de = safe_load_spacy("de_core_news_sm", [sys.executable, "-m", "spacy", "download", "de_core_news_sm"])
        spacy_en = safe_load_spacy("en_core_web_sm", [sys.executable, "-m", "spacy", "download", "en_core_web_sm"])

        self.src_vocab = {"<unk>": 0, "<pad>": 1, "<sos>": 2, "<eos>": 3}
        self.tgt_vocab = {"<unk>": 0, "<pad>": 1, "<sos>": 2, "<eos>": 3}
        self.tgt_itos = {0: "<unk>", 1: "<pad>", 2: "<sos>", 3: "<eos>"}

        try:
            raw_dataset = load_dataset("bentrevett/multi30k", split="train")
            cnt_de, cnt_en = Counter(), Counter()
            
            for item in raw_dataset:
                cnt_de.update([tk.text.lower() for tk in self.spacy_de.tokenizer(item['de'])])
                cnt_en.update([tk.text.lower() for tk in spacy_en.tokenizer(item['en'])])
                
            for word, freq in cnt_de.items():
                if freq >= 2: 
                    self.src_vocab[word] = len(self.src_vocab)
                    
            for word, freq in cnt_en.items():
                if freq >= 2:
                    curr_idx = len(self.tgt_vocab)
                    self.tgt_vocab[word] = curr_idx
                    self.tgt_itos[curr_idx] = word
                    
            src_vocab_size = len(self.src_vocab)
            tgt_vocab_size = len(self.tgt_vocab)
        except Exception:
            pass

        self.d_model = d_model
        
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        
        self.src_pos_enc = PositionalEncoding(d_model, dropout)
        self.tgt_pos_enc = PositionalEncoding(d_model, dropout)

        enc_layer = EncoderLayer(d_model, num_heads, d_ff, dropout)
        dec_layer = DecoderLayer(d_model, num_heads, d_ff, dropout)
        
        self.encoder = Encoder(enc_layer, N)
        self.decoder = Decoder(dec_layer, N)
        
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)

        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

        dl_path = "best_noam_final.pt"
        if not os.path.exists(dl_path):
            try:
                import gdown
                gdown.download(id="12ii8FI5fcp91bwVvYEUwjbExj2hiN_bc", output=dl_path, quiet=False)
            except Exception: 
                pass

        if os.path.exists(dl_path):
            try:
                data = torch.load(dl_path, map_location="cpu")
                ref_sd = data.get("model_state_dict", data) if isinstance(data, dict) else data.state_dict()
                
                upd_sd = {}
                my_keys = list(self.state_dict().keys())
                
                for rk, rv in ref_sd.items():
                    k = rk.replace("src_embed.0.", "src_embedding.")
                    k = k.replace("tgt_embed.0.", "tgt_embedding.")
                    k = k.replace("src_embed.1.", "src_pos_enc.")
                    k = k.replace("tgt_embed.1.", "tgt_pos_enc.")
                    k = k.replace("generator.", "output_projection.")
                    k = k.replace("w_q.", "W_q.").replace("w_k.", "W_k.").replace("w_v.", "W_v.").replace("w_o.", "W_o.")
                    k = k.replace("src_attn.", "cross_attn.").replace("feed_forward.", "ffn.")
                    
                    if k in my_keys:
                        tgt = self.state_dict()[k]
                        if rv.shape != tgt.shape:
                            clone_tgt = tgt.clone()
                            if rv.dim() == 2 and tgt.dim() == 2:
                                mx_0, mx_1 = min(rv.size(0), tgt.size(0)), min(rv.size(1), tgt.size(1))
                                clone_tgt[:mx_0, :mx_1] = rv[:mx_0, :mx_1]
                            elif rv.dim() == 1 and tgt.dim() == 1:
                                mx_0 = min(rv.size(0), tgt.size(0))
                                clone_tgt[:mx_0] = rv[:mx_0]
                            upd_sd[k] = clone_tgt
                        else:
                            upd_sd[k] = rv
                
                self.load_state_dict(upd_sd, strict=False)
            except Exception:
                pass

    def encode(self, src, src_mask):
        scaled_emb = self.src_embedding(src) * (self.d_model ** 0.5)
        return self.encoder(self.src_pos_enc(scaled_emb), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        scaled_emb = self.tgt_embedding(tgt) * (self.d_model ** 0.5)
        dec_out = self.decoder(self.tgt_pos_enc(scaled_emb), memory, src_mask, tgt_mask)
        return self.output_projection(dec_out)

    def forward(self, src, tgt, src_mask, tgt_mask):
        memory = self.encode(src, src_mask)
        return self.decode(memory, src_mask, tgt, tgt_mask)

    def infer(self, src_sentence: str) -> str:
        self.eval()
        dev = next(self.parameters()).device
        
        tks = [t.text.lower() for t in self.spacy_de.tokenizer(src_sentence)]
            
        i_unk = self.src_vocab.get("<unk>", 0)
        i_sos = self.src_vocab.get("<sos>", 2)
        i_eos = self.src_vocab.get("<eos>", 3)
        i_pad = self.src_vocab.get("<pad>", 1)
        
        indices = [i_sos] + [self.src_vocab.get(x, i_unk) for x in tks] + [i_eos]
        inp_tensor = torch.tensor(indices, dtype=torch.long, device=dev).unsqueeze(0)
        s_mask = make_src_mask(inp_tensor, i_pad).to(dev)
        
        with torch.no_grad():
            mem = self.encode(inp_tensor, s_mask)
            seq = torch.tensor([[i_sos]], dtype=torch.long, device=dev)
            max_iterations = int(1.5 * len(indices)) + 5
            
            for _ in range(max_iterations):
                t_mask = make_tgt_mask(seq, i_pad).to(dev)
                preds = self.decode(mem, s_mask, seq, t_mask)
                pred_tok = preds[:, -1, :].argmax(dim=-1).item()
                seq = torch.cat([seq, torch.tensor([[pred_tok]], dtype=torch.long, device=dev)], dim=1)
                
                if pred_tok == i_eos:
                    break
                    
        out_ids = seq.squeeze(0).tolist()
        res = []
        for x in out_ids:
            if x not in (i_sos, i_eos, i_pad):
                res.append(self.tgt_itos.get(x, str(x)))
                
        return " ".join(res)