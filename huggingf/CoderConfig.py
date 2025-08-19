from transformers import LlamaConfig

class CustomCoderConfig(LlamaConfig):
    model_type = "llama"  # Changé de "coder-v1" à "llama"

    def __init__(
        self,
        vocab_size=32000,
        hidden_size=768,        # Renommé de d_model
        num_attention_heads=32, # Renommé de num_heads  
        num_hidden_layers=32,   # Renommé de num_layers
        intermediate_size=4096, # Renommé de d_ff
        max_position_embeddings=4096, # Renommé de max_seq_len
        attention_dropout=0.2,  # Renommé de dropout
        **kwargs
    ):
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_hidden_layers=num_hidden_layers,
            intermediate_size=intermediate_size,
            max_position_embeddings=max_position_embeddings,
            attention_dropout=attention_dropout,
            **kwargs
        )
        
        # Garder vos anciens noms pour compatibilité
        self.d_model = hidden_size
        self.num_heads = num_attention_heads
        self.num_layers = num_hidden_layers
        self.d_ff = intermediate_size
        self.max_seq_len = max_position_embeddings
        self.dropout = attention_dropout