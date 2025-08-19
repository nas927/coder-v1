from huggingf.CoderConfig import CustomCoderConfig
from transformers import PreTrainedModel, PreTrainedTokenizerFast
from transformer import TransformerDecoder
import torch


class CustomCoderPreTrainedModel(PreTrainedModel):
    config_class = CustomCoderConfig
    base_model_prefix = "transformer"

class CustomCoderModel(CustomCoderPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.transformer = TransformerDecoder(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            d_ff=config.d_ff,
            max_seq_len=config.max_seq_len,
            dropout=config.dropout
        )
    
    def forward(self, input_ids, labels=None):
        return self.transformer(input_ids, labels)

    def generate(self, input_ids, max_new_tokens=50, temperature=0.8, top_k=50, top_p=0.9):
        return self.transformer.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )
    
def save_as_hf_model(model_path, save_directory):
    # Charger le modèle existant
    modelLoaded = torch.load(model_path, weights_only=False)
    # Créer la configuration
    config = CustomCoderConfig(
        vocab_size=32000,  # Ajustez selon votre vocabulaire
        d_model=768,
        num_heads=32,
        num_layers=32,
        d_ff=4096
    )
    
    # Créer le modèle HF
    model = CustomCoderModel(config)
    model.load_state_dict(model.state_dict())
    
    # Sauvegarder le modèle et la configuration
    model.save_pretrained(save_directory)
    config.save_pretrained(save_directory)
    
    # Sauvegarder le tokenizer
    tokenizer = PreTrainedTokenizerFast(tokenizer_file="./coder-v1.json")
    tokenizer.save_pretrained(save_directory)

# Utilisation
if __name__ == "__main__":
    save_as_hf_model(
        model_path="./checkpoint.pt",
        save_directory="./huggingf"
    )