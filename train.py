import preprocess
import torch.optim as optim
import torch.nn.functional as F
from transformer import *
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=3, help="Nombre d'epochs default 3")
parser.add_argument("--batch_size", type=int, default=5, help="Nombre de batch dans 1 epoch default 5")
args = parser.parse_args()

def init_model():
    tokenizer = preprocess.tokenize()
    vocab_size = tokenizer.get_vocab_size()  # Taille du vocabulaire GPT-2
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
    batches = preprocess.ret_batch(datas, batch_size=args.batch_size)

    print(f"Nombre de paramètres: {sum(p.numel() for p in model.parameters()):,}")
    # Forward pass
    for epochs in range(args.epochs):
        print("Epochs : ", epochs)
        for index_batch, batch in enumerate(batches):
            print("Mise à jour des gardients batch n° : ", index_batch)
            for i in range(len(batch['input'])):
                input_tensor = torch.tensor([batch['input'][i].ids])
                output_tensor = torch.tensor([batch['output'][i].ids])
                outputs = model(input_tensor, output_tensor)
                loss = outputs['loss']
                predictions = outputs['predictions']
                logits = outputs['logits']
                print(f"Loss: {loss.item():.4f}")
                print(f"Logits shape: {outputs['logits'].shape}")
                print(f"Predictions shape: {predictions.shape}")

                last_logits = logits[0, -1, :]   # shape = [vocab_size]
                pred_id = torch.argmax(last_logits).item()
                print("Token décodé : ", preprocess.decode_data(tokenizer, [[pred_id]]))

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
            # Mise à jour des poids
            optimizer.step()
            scheduler.step()
            print("Sauvegarde du modèle")
            torch.save(model, "checkpoint.pt")


if __name__ == "__main__":
    model, optimizer, scheduler, tokenizer = init_model()
    launch_training(model, optimizer, scheduler, tokenizer)

