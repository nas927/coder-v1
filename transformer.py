import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RoPEPositionalEncoding(nn.Module):
    def __init__(self, head_dim):
        super().__init__()
        self.head_dim = head_dim
        
        # Créer les fréquences pour RoPE
        inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
        # on le stocke comme buffer → non entraînable mais transférable sur GPU/CPU
        self.register_buffer('inv_freq', inv_freq)
        
    def forward(self, seq_len, device):
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb

class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)
        
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
        self.rope = RoPEPositionalEncoding(self.head_dim)
        
    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.shape
        
        # Projections Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Appliquer RoPE
        rope_emb = self.rope(seq_len, x.device)
        q = self.apply_rope(q, rope_emb)
        k = self.apply_rope(k, rope_emb)
        
        # Transposer pour l'attention
        q = q.transpose(1, 2)  # [batch, num_heads, seq_len, head_dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Calcul de l'attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Masque causal pour decoder-only
        if mask is None:
            mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        
        scores.masked_fill_(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        
        # Recombiner les têtes
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        
        return self.out_proj(attn_output)
    
    def apply_rope(self, x, rope_emb):
        """Applique RoPE aux embeddings"""
        # x shape: [batch, seq_len, num_heads, head_dim]
        # rope_emb shape: [seq_len, head_dim]
        
        seq_len, head_dim = x.shape[1], x.shape[-1]
        
        # S'assurer que rope_emb a la bonne dimension
        if rope_emb.shape[-1] != head_dim:
            rope_emb = rope_emb[..., :head_dim]
        
        cos_emb = rope_emb.cos().unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, head_dim]
        sin_emb = rope_emb.sin().unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, head_dim]
        
        # Séparer les dimensions paires et impaires
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        
        # S'assurer que cos_emb et sin_emb ont les bonnes dimensions
        cos_half = cos_emb[..., ::2]
        sin_half = sin_emb[..., ::2]
        
        # Appliquer la rotation
        rotated_x1 = x1 * cos_half - x2 * sin_half
        rotated_x2 = x2 * cos_half + x1 * sin_half
        
        # Recombiner
        rotated_x = torch.zeros_like(x)
        rotated_x[..., ::2] = rotated_x1
        rotated_x[..., 1::2] = rotated_x2
        
        return rotated_x

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = SwiGLU(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Attention avec connexion résiduelle et normalisation
        attn_output = self.attention(self.norm1(x), mask)
        x = x + self.dropout(attn_output)
        
        # Feed forward avec connexion résiduelle et normalisation
        ff_output = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_output)
        
        return x

class TransformerDecoder(nn.Module):

    def __init__(self, vocab_size, d_model=512, num_heads=8, num_layers=6, d_ff=2048, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # Embedding des tokens
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Couches du transformer
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Normalisation finale et projection de sortie
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Initialisation des poids
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.xavier_uniform_(module.weight)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(self, input_ids, target=None):
        batch_size, seq_len = input_ids.shape
        
        # Embedding des tokens
        x = self.token_embedding(input_ids) * math.sqrt(self.d_model)
        
        # Passer à travers toutes les couches
        for layer in self.layers:
            x = layer(x)
        
        # Normalisation finale
        x = self.norm(x)
        
        # Projection vers le vocabulaire
        logits = self.lm_head(x)
        
        loss = None
        if target is not None:
            # Calcul de la loss pour l'entraînement
            shift_logits = logits[..., :-1, :].contiguous()
            shift_target = target[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)), 
                shift_target.view(-1),
                ignore_index=-100
            )
        
        return {
            'logits': logits,
            'loss': loss,
            'predictions': torch.argmax(logits, dim=-1)
        }
    
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int=50, temperature: float=1.0, top_k: int=1, top_p: float=0.5):
        """Génération de texte"""
        self.eval()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Forward pass
                outputs = self.forward(input_ids)
                logits = outputs['logits']
                
                # Prendre le dernier token
                next_token_logits = logits[:, -1, :] / temperature
                
                # Top-k sampling si spécifié
                if top_k is not None:
                    v, _ = torch.topk(next_token_logits, top_k)
                    next_token_logits[next_token_logits < v[:, [-1]]] = float('-inf')
            
                # Top-p (nucleus) sampling
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                    # Supprimer les tokens dont la probabilité cumulée dépasse top_p
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0

                    indices_to_remove = sorted_indices_to_remove.scatter(
                        dim=1,
                        index=sorted_indices,
                        src=sorted_indices_to_remove
                    )
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Échantillonnage
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Ajouter le nouveau token
                input_ids = torch.cat([input_ids, next_token], dim=1)
        
        return input_ids