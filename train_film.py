import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from lib.utils import get_device, makedirs
from lib.create_latent_ode_model import create_LatentODE_model
from lib.read_tacro import extract_gen_tac_film, TacroFilmDataset, collate_fn_tacro_film
from torch.distributions.normal import Normal

def main():
    parser = argparse.ArgumentParser('Latent ODE FiLM Training')
    parser.add_argument('--exp', type=str, default='exp_film_run', help="Experiment folder to load data from")
    parser.add_argument('--load', type=str, required=True, help="Path to pre-trained base model checkpoint (.ckpt)")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate for FiLM")
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=64)
    # Important base model args to reconstruct architecture
    parser.add_argument('-l', '--latents', type=int, default=6)
    parser.add_argument('--rec-dims', type=int, default=20)
    parser.add_argument('--rec-layers', type=int, default=1)
    parser.add_argument('--gen-layers', type=int, default=1)
    parser.add_argument('-u', '--units', type=int, default=100)
    parser.add_argument('-g', '--gru-units', type=int, default=100)
    parser.add_argument('--z0-encoder', type=str, default='odernn')
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 1. Load the Paired Data
    print("Loading FiLM paired dataset...")
    data_dict_train, _ = extract_gen_tac_film(
        file_path=[f"exp_run_all/{args.exp}/virtual_cohort_film_train.csv"]
    )
    data_dict_test, _ = extract_gen_tac_film(
        file_path=[f"exp_run_all/{args.exp}/virtual_cohort_film_test.csv"]
    )

    train_dataset = TacroFilmDataset(data_dict_train)
    test_dataset = TacroFilmDataset(data_dict_test)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                              collate_fn=lambda x: collate_fn_tacro_film(x, device))
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, 
                             collate_fn=lambda x: collate_fn_tacro_film(x, device))

    # 2. Instantiate Base Model
    input_dim = 1
    obsrv_std = torch.Tensor([0.01]).to(device)
    z0_prior = Normal(torch.Tensor([0.0]).to(device), torch.Tensor([1.]).to(device))
    
    # We use your create_LatentODE_model (Assuming it constructs a LatentODE object)
    print("Reconstructing base model...")
    model = create_LatentODE_model(args, input_dim, z0_prior, obsrv_std, device)

    # 3. Load Weights
    print(f"Loading weights from {args.load}...")
    checkpoint = torch.load(args.load, map_location=device)
    model.load_state_dict(checkpoint['state_dict'], strict=False) # Strict False to allow uninitialized FiLM weights

    # 4. Freeze Base Model, Unfreeze FiLM
    for param in model.parameters():
        param.requires_grad = False
    
    for param in model.film_gamma.parameters():
        param.requires_grad = True
    for param in model.film_beta.parameters():
        param.requires_grad = True

    # 5. Setup Optimizer (Only for FiLM parameters)
    optimizer = optim.Adamax(
        list(model.film_gamma.parameters()) + list(model.film_beta.parameters()), 
        lr=args.lr
    )
    criterion = nn.MSELoss()

    # 6. Training Loop
    print("Starting FiLM training...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            # Forward extrapolation
            pred_v2, _ = model.get_reconstruction_extrapolation(
                data_v1=batch["observed_data_v1"],
                time_steps_v1=batch["observed_tp_v1"],
                time_steps_v2=batch["tp_to_predict_v2"],
                dose_v1=batch["dose_v1"],
                dose_v2=batch["dose_v2"],
                static_v1=batch["static_v1"],
                n_traj_samples=1
            )
            
            # Reshape predictions to match target
            # pred_v2 is [n_traj_samples, batch, seq, dims], squeeze n_traj
            pred_v2 = pred_v2.squeeze(0) 
            target_v2 = batch["data_to_predict_v2"]

            loss = criterion(pred_v2, target_v2)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()

        if epoch % 10 == 0:
            # Quick evaluation
            model.eval()
            test_loss = 0
            with torch.no_grad():
                for batch in test_loader:
                    pred_v2, _ = model.get_reconstruction_extrapolation(
                        data_v1=batch["observed_data_v1"],
                        time_steps_v1=batch["observed_tp_v1"],
                        time_steps_v2=batch["tp_to_predict_v2"],
                        dose_v1=batch["dose_v1"],
                        dose_v2=batch["dose_v2"],
                        static_v1=batch["static_v1"]
                    )
                    test_loss += criterion(pred_v2.squeeze(0), batch["data_to_predict_v2"]).item()
                    
            print(f"Epoch {epoch:04d} | Train MSE: {train_loss/len(train_loader):.6f} | Test MSE: {test_loss/len(test_loader):.6f}")

    # Save specifically the tuned model
    save_path = args.load.replace(".ckpt", "_film.ckpt")
    torch.save({
        'args': args,
        'state_dict': model.state_dict()
    }, save_path)
    print(f"FiLM tuned model saved to {save_path}")

if __name__ == '__main__':
    main()