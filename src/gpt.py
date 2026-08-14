
import torch
import torch.nn as nn
from torch.nn import functional as F


class Head(nn.Module):
    def __init__(self, n_embedding, context_size, head_size, dropout_p):
        super().__init__()
        self.head_size = head_size
        self.query = nn.Linear(n_embedding, self.head_size, bias=False)
        self.key = nn.Linear(n_embedding, head_size, bias=False)
        self.value = nn.Linear(n_embedding, head_size, bias=False)
        self.drouput = nn.Dropout(dropout_p)

        self.register_buffer("tril", torch.tril(torch.ones(context_size, context_size)))

    def forward(self, x):
        B, T, C = x.shape

        q = self.query(x)  # (B, T, C)
        k = self.key(x)  # (B, T, C)

        w = q @ k.transpose(-2, -1) * self.head_size ** -0.5  # (B, T, C) @ (B, C, T) -> (B, T, T)
        w = w.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # (B, T, T)
        w = F.softmax(w, dim=-1)
 
        w = self.drouput(w)

        v = self.value(x) # (B, T, C)
        y = w @ v  # (B, T, T) @ (B, T, C) -> (B, T, C)
        return y


# decoder only transformer


class MultiHeadAttention(nn.Module):

    def __init__(self, n_embedding, context_size, num_heads, head_size, dropout_p):
        super().__init__()

        self.heads = nn.ModuleList([
            Head(n_embedding, context_size, head_size, dropout_p) 
            for _ in range(num_heads)
        ])
        self.projection = nn.Linear(n_embedding, n_embedding)
        self.droupout = nn.Dropout(dropout_p)

    def forward(self, x):
        x = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.droupout(self.projection(x))

class FeedForward(nn.Module):

    def __init__(self, n_embedding, dropout_p, inner_dimension_scale=4):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(n_embedding, int(inner_dimension_scale) * n_embedding),
            nn.ReLU(),
            nn.Linear( int(inner_dimension_scale) * n_embedding, n_embedding),  # projection
            nn.Dropout(dropout_p),
        )

    def forward(self, x):
        return self.ff(x)

class Block(nn.Module):

    def __init__(self, n_embedding, context_size, num_heads, dropout_p):
        super().__init__()
        head_size = n_embedding // num_heads
        self.sa_head = MultiHeadAttention(n_embedding, context_size, num_heads, head_size, dropout_p)
        self.ff = FeedForward(n_embedding, dropout_p)
        self.ln1 = nn.LayerNorm(n_embedding)
        self.ln2 = nn.LayerNorm(n_embedding)

    def forward(self, x):
        x = x + self.sa_head(self.ln1(x)) # (B, T, C)
        x = x + self.ff(self.ln2(x))  # (B, T, C)
        return x


class GPTLM(nn.Module):

    def __init__(self, num_heads, n_embedding, context_size, vocab_size, n_layer, dropout_p):
        super().__init__()
        self.n_layer = n_layer
        self.n_embedding = n_embedding
        self.context_size = context_size
        self.num_heads = num_heads
        self.vocab_size = vocab_size
        
        self.token_embedding_table = nn.Embedding(vocab_size, n_embedding)
        self.position_embedding_table = nn.Embedding(context_size, n_embedding)
        self.blocks = nn.Sequential(*[
            Block(n_embedding, context_size, num_heads, dropout_p)
            for _ in range(n_layer)
        ])
        self.ln = nn.LayerNorm(n_embedding)
        self.lm_head = nn.Linear(n_embedding, vocab_size)

        
    def forward(self, idx, targets=None):
        B, T = idx.shape
        device = idx.device

        tok_emb = self.token_embedding_table(idx) # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)

        x = tok_emb + pos_emb # (B, T, C)
        x = self.blocks(x) # (B, T, C)
        x = self.ln(x)
        logits = self.lm_head(x) # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.context_size:]
            logits, loss = self(idx_cond)

            logits = logits[:, -1, :] # (B, C)

            probs = F.softmax(logits, dim=-1) # (B, C)

            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)

            idx = torch.cat([idx, idx_next], dim=1) # (B, C + 1)
        return idx

