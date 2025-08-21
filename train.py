import preprocess
import torch.optim as optim
import torch.nn.functional as F
from transformer import *
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=3, help="Nombre d'epochs default 3")
parser.add_argument("--batch_size", type=int, default=5, help="Nombre de batch dans 1 epoch default 5")
args = parser.parse_args()

def early_stopping(epochs: int, average_loss: float, best_loss: float, patience: int, patience_counter: int, model: TransformerDecoder):
    # Early stopping check
    if average_loss < best_loss:
        best_loss = average_loss
        patience_counter = 0
        # Sauvegarder le meilleur modèle
        print("Sauvegarde du meilleur modèle")
        torch.save(model, "best_model.pt")
    else:
        patience_counter += 1
        print(f"Patience: {patience_counter}/{patience}")
    if patience_counter >= patience:
        print(f"Early stopping après {epochs} epochs")
        print(f"Meilleure loss: {best_loss:.4f}")
        return False
    return True



def init_model():
    tokenizer = preprocess.tokenize()
    vocab_size: int = len(tokenizer)  # Taille du vocabulaire Mistral + data token spéciaux
    model: TransformerDecoder = TransformerDecoder(
        vocab_size=vocab_size,
        d_model=768,
        num_heads=32,
        num_layers=32,
        d_ff=4096,
        dropout=0.2
    )
    model.to(device)
    optimizer: torch.optim.adamw.AdamW = optim.AdamW(model.parameters(), lr=4.2e-4, weight_decay=0.01)
    scheduler: torch.optim.lr_scheduler.CosineAnnealingLR = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)

    return model, optimizer, scheduler, tokenizer

def launch_training(model, optimizer, scheduler, tokenizer):
    dataset: list[str] = preprocess.load_data()
    datas: dict = preprocess.encode_data(tokenizer, dataset)
    batches: list[torch.Tensor] = preprocess.ret_batch(datas, batch_size=args.batch_size)

    print(f"Nombre de paramètres: {sum(p.numel() for p in model.parameters()):,}")

    # Early stopping parameters
    # Stopper si pas d'évolution durant 10 epochs
    best_loss: float = float('inf')
    patience: int = 10
    patience_counter: int = 0

    # Forward pass
    for epochs in range(args.epochs):
        total_loss: float = 0
        num_batches: int = 0

        print("Epochs : ", epochs)
        for index_batch, batch in enumerate(batches):
            print("batch n° : ", index_batch)
            input_tensor: torch.Tensor = batch.to(device)
            output_tensor: torch.Tensor = input_tensor.clone().to(device)

            # Remplacer les tokens de padding par -100
            pad_mask: torch.Tensor = (output_tensor == tokenizer.pad_token_id)
            output_tensor[pad_mask] = -100

            outputs: dict = model(input_tensor, input_tensor)
            loss = outputs['loss']
            predictions = outputs['predictions']
            logits = outputs['logits']
            print(f"Loss: {loss.item():.4f}")
            print(f"Logits shape: {outputs['logits'].shape}")
            print(f"Predictions shape: {predictions.shape}")

            # Pas d'intérêt je fournis le batch entier
            # last_logits = logits[0, -1, :]   # shape = [vocab_size]
            # pred_id = torch.argmax(last_logits).item()
            # print("Token décodé : ", preprocess.decode_data(tokenizer, pred_id))

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            # Mise à jour des poids
            print("Mise à jour des gradients")
            optimizer.step()
            scheduler.step()

        average_loss: float = total_loss / num_batches
        print(f"Loss moyenne de l'epoch {epochs}: {average_loss:.4f}")

        if not early_stopping(epochs, average_loss, best_loss, patience, patience_counter, model):
            break


if __name__ == "__main__":
    model, optimizer, scheduler, tokenizer = init_model()
    launch_training(model, optimizer, scheduler, tokenizer)

