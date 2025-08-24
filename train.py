from colorama.initialise import reset_all
import preprocess
import torch.optim as optim
import torch.nn.functional as F
from transformer import *
import argparse
from colorama import init, Style, Back, Fore


init()
parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=3, help="Nombre d'epochs default 3")
parser.add_argument("--batch_size", type=int, default=5, help="Nombre de batch dans 1 epoch default 5")
parser.add_argument("--model_path", type=str, default="best_model.pt", help="Chemin de sauvegarde du model")
parser.add_argument("--dataset", type=str, default="./all-in-one.txt", help="Chemin du dataset.txt")
parser.add_argument("--max_length", type=int, default=0, help="Maximum de séquence avant de couper")
parser.add_argument("--lora", type=int, default=0, help="Entrainer avec lora activé")
parser.add_argument("--lora-only", type=int, default=1, help="Entrainer seulement les couches d'attentions")
parser.add_argument("--lora-r", type=int, default=8, help="low rank lora")
parser.add_argument("--lora-alpha", type=int, default=16, help="Alpha lora")
parser.add_argument("--lora-path", type=str, default="best_model_lora.pt", help="Chemin de sauvegarde du model lora")
parser.add_argument("--d_model", type=int, default=2048, help="d_model size  doit être multiple de num_heads")
parser.add_argument("--d_ff", type=int, default=5504, help="feed forward size")
parser.add_argument("--device", type=str, default="cpu", help="Vous voulez utilisez quoi comme matériel pour l'entrainement")
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else args.device)
print(Fore.GREEN + "Device : " + torch.cuda.get_device_name(torch.cuda.current_device()) if torch.cuda.is_available() else "Device : CPU")
print(Style.RESET_ALL)

def early_stopping(epochs: int, average_loss: float, best_loss: float, patience: int, patience_counter: int, model: TransformerDecoder):
    # Early stopping check
    if average_loss < best_loss:
        best_loss = average_loss
        patience_counter = 0
        # Sauvegarder le meilleur modèle
        if args.lora and args.lora_only:
            print(Fore.GREEN + f"Sauvegarde du meilleur modèle lora uniquement : {args.lora_path}" + Style.RESET_ALL)
            torch.save(
                {k: v for k, v in model.state_dict().items() if "lora" in k}, 
                args.lora_path
            )
        else:
            print(Fore.GREEN + f"Sauvegarde du meilleur modèle : {args.model_path}" + Style.RESET_ALL)
            torch.save(model, args.model_path)
    else:
        patience_counter += 1
        print(f"Patience: {patience_counter}/{patience}")
    if patience_counter >= patience:
        print(Fore.YELLOW + f"Early stopping après {epochs} epochs")
        print(Fore.Green + f"Meilleure loss: {best_loss:.4f}" + Style.RESET_ALL)
        return False
    return True

def init_model():
    tokenizer = preprocess.tokenize()
    vocab_size: int = len(tokenizer)  # Taille du vocabulaire Mistral + data token spéciaux
    model: TransformerDecoder = TransformerDecoder(
        vocab_size=vocab_size,
        d_model=args.d_model,
        num_heads=32,
        num_layers=32,
        d_ff=args.d_ff,
        dropout=0.2,
        lora={"enabled": args.lora, "r": args.lora_r, "alpha": args.lora_alpha}
    )
    model.to(device)
    if args.lora:
        print(Fore.YELLOW + f"Lora est activé rank = {args.lora_r} alpha = {args.lora_alpha} lora-only = {args.lora_only}" + Style.RESET_ALL)
        if args.lora_only:  
            # Tout geler
            for param in model.parameters():
                param.requires_grad = False

        # Débloquer LoRA
        for name, param in model.named_parameters():
            if "lora" in name.lower():  # lora_A et lora_B
                param.requires_grad = True

    optimizer: torch.optim.adamw.AdamW  = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=4.2e-4, weight_decay=0.01
    )
    scheduler: torch.optim.lr_scheduler.CosineAnnealingLR = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scheduler: torch.optim.lr_scheduler.CosineAnnealingLR = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    return model, optimizer, scheduler, tokenizer

def launch_training(model, optimizer, scheduler, tokenizer):
    dataset: list[str] = preprocess.load_data(args.dataset)
    batches: list[torch.Tensor] = preprocess.ret_batch(tokenizer, dataset, batch_size=args.batch_size, max_length=args.max_length)
    scaler = torch.amp.GradScaler()

    print(Back.GREEN + Fore.WHITE + f"Nombre de paramètres: {sum(p.numel() for p in model.parameters()):,}" + Style.RESET_ALL)
    print(f"Nombre d'epochs: {args.epochs}")

    # Early stopping parameters
    # Stopper si pas d'évolution durant 10 epochs
    best_loss: float = float('inf')
    patience: int = 10
    patience_counter: int = 0

    # Forward pass
    for epochs in range(args.epochs):
        total_loss: float = 0
        num_batches: int = 0

        print(Back.WHITE + Fore.BLACK + "Epochs : ", epochs, Style.RESET_ALL)
        for index_batch, batch in enumerate(batches):
            print(Back.WHITE + Fore.BLACK + "batch n° : ", index_batch, Style.RESET_ALL)
            print(Fore.BLUE + "Taille des séquences du batch : ", len(batch[0]), Style.RESET_ALL)
            input_tensor: torch.Tensor = torch.tensor(torch.as_tensor(batch['input_ids']).detach().clone().ids).to(device)
            output_tensor: torch.Tensor = input_tensor.clone()
            

            # Remplacer les tokens de padding par -100
            pad_mask: torch.Tensor = (output_tensor == tokenizer.pad_token_id)
            output_tensor[pad_mask] = -100
            optimizer.zero_grad()
            with torch.autocast(device_type='cuda', dtype=torch.float16):
              outputs: dict = model(input_tensor, output_tensor)
            loss = outputs['loss']
            predictions = outputs['predictions']
            logits = outputs['logits']
            print(f"Loss: {loss.item():.4f}")
            print(f"Logits shape: {outputs['logits'].shape}")
            print(f"Predictions shape: {predictions.shape}")
            # Perplexity comme métrique doit être entre 1 et 300 pour un modèle raisonnable
            perplexity = torch.exp(loss)
            if perplexity > 300:
                print(Fore.RED + "Perplexity : ", perplexity.item(), Style.RESET_ALL)
            else:
                print(Fore.GREEN + "Perplexity : ", perplexity.item(), Style.RESET_ALL)

            total_loss += loss.item()
            num_batches += 1

            # Pas d'intérêt je fournis le batch entier
            # last_logits = logits[0, -1, :]   # shape = [vocab_size]
            # pred_id = torch.argmax(last_logits).item()
            # print("Token décodé : ", preprocess.decode_data(tokenizer, pred_id))

            # Backward pass
            scaler.scale(loss).backward()
            # Mise à jour des poids
            print("Mise à jour des gradients")
            scaler.step(optimizer)
            scaler.update()

        scheduler.step()
        average_loss: float = total_loss / num_batches
        print(f"Loss moyenne de l'epoch {epochs}: {average_loss:.4f}")

        if not early_stopping(epochs, average_loss, best_loss, patience, patience_counter, model):
            break


if __name__ == "__main__":
    model, optimizer, scheduler, tokenizer = init_model()
    launch_training(model, optimizer, scheduler, tokenizer)


    

