import preprocess
import torch.optim as optim
import torch.nn.functional as F
from transformer import *

def init_model():
    tokenizer = preprocess.tokenize()
    vocab_size = len(tokenizer)  # Taille du vocabulaire GPT-2
    model = TransformerDecoder(
        vocab_size=vocab_size,
        d_model=768,
        num_heads=32,
        num_layers=32,
        d_ff=4096,
        dropout=0.2
    )

    optimizer = optim.AdamW(model.parameters(), lr=4.2e-4, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)

    return model, optimizer, scheduler, tokenizer

def launch_training(model, optimizer, scheduler, tokenizer):
    dataset = preprocess.load_data()
    datas = preprocess.encode_data(tokenizer, dataset)

    print(f"Nombre de paramètres: {sum(p.numel() for p in model.parameters()):,}")
    # Forward pass
    for epochs in range(2):
        print("Epochs : ", epochs)
        for i in range(0, len(datas["input"]["input_ids"])):
            input_tensor = datas["input"]["input_ids"][i].unsqueeze(0)
            labels_tensor = datas["output"]["input_ids"][i].unsqueeze(0)

            outputs = model(input_tensor, labels=labels_tensor)
            loss = outputs['loss']
            predictions = outputs['predictions']
            logits = outputs['logits']
            print(f"Loss: {loss.item():.4f}")
            print(f"Logits shape: {outputs['logits'].shape}")
            print(f"Predictions shape: {predictions.shape}")

            probs = F.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)
            print("prédiction : ", preds[0])
            print("Mots prédits : ",  preprocess.decode_data(tokenizer, preds[0].tolist()))
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
        print("Mise à jour des gardients")
        # Mise à jour des poids
        optimizer.step()
        scheduler.step()
        print("Sauvegarde du modèle")
        torch.save(model, "checkpoint.pt")


if __name__ == "__main__":
    model, optimizer, scheduler, tokenizer = init_model()
    launch_training(model, optimizer, scheduler, tokenizer)

