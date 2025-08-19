from transformers import LlamaForCausalLM, LlamaConfig, AutoTokenizer
import torch

def save_as_hf_model(model_path, save_directory):
    # Charger votre modèle existant
    checkpoint = torch.load(model_path, weights_only=False)
    
    # Récupérer les poids selon le format de sauvegarde
    if isinstance(checkpoint, dict):
        # Si sauvegardé avec torch.save({'model': model.state_dict(), ...})
        if 'model' in checkpoint:
            model_state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            model_state_dict = checkpoint['state_dict']
        else:
            model_state_dict = checkpoint
    else:
        # Si sauvegardé directement avec torch.save(model, ...)
        model_state_dict = checkpoint.state_dict()
    
    # Créer une configuration Llama standard
    config = LlamaConfig(
        vocab_size=32000,
        hidden_size=768,           # votre d_model
        num_attention_heads=32,    # votre num_heads
        num_hidden_layers=32,      # votre num_layers
        intermediate_size=4096,    # votre d_ff
        max_position_embeddings=4096, # votre max_seq_len
        torch_dtype="float32",
        # Paramètres Llama standards
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        attention_dropout=0.0,
        hidden_dropout=0.2,
        tie_word_embeddings=False
    )
    
    # Créer un modèle Llama vide
    model = LlamaForCausalLM(config)
    
    # IMPORTANT: Mapper vos poids vers l'architecture Llama
    # Vous devrez adapter cette partie selon votre architecture exacte
    
    # Debug: voir les clés disponibles
    print("Clés dans votre modèle:")
    for key in model_state_dict.keys():
        print(f"  {key}: {model_state_dict[key].shape}")
    
    print("\nClés attendues par Llama:")
    for key in model.state_dict().keys():
        print(f"  {key}: {model.state_dict()[key].shape}")
    
    # Exemple de mapping (à adapter selon votre TransformerDecoder) :
    new_state_dict = {}
    
    for key, value in model_state_dict.items():
        # Adapter les noms de vos couches vers les noms Llama
        if 'embedding' in key:
            new_key = key.replace('embedding', 'model.embed_tokens')
        elif 'output_projection' in key:
            new_key = key.replace('output_projection', 'lm_head')
        # Ajoutez d'autres mappings selon votre architecture
        else:
            new_key = f"model.{key}"  # Préfixer avec "model."
            
        new_state_dict[new_key] = value
    
    # Charger les poids adaptés
    model.load_state_dict(new_state_dict, strict=False)
    
    # Sauvegarder
    model.save_pretrained(save_directory)
    config.save_pretrained(save_directory)
    
    # Sauvegarder le tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(save_directory)
        tokenizer.save_pretrained(save_directory)
    except:
        print("Attention: Tokenizer non trouvé, vous devrez l'ajouter manuellement")

# Utilisation
if __name__ == "__main__":
    save_as_hf_model(
        model_path="./checkpoint.pt",
        save_directory="./huggingf_compatible"
    )