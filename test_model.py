import argparse
import torch
from lib.utils import get_ckpt_model, compute_loss_all_batches
from lib.parse_datasets import parse_datasets
from lib.create_latent_ode_model import create_LatentODE_model
import os

def test_model(args):
    """
    Loads a trained model and evaluates it on a test dataset.
    """
    # Create the model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_obj = parse_datasets(args, device)
    input_dim = data_obj["input_dim"]

    obsrv_std = torch.Tensor([0.01]).to(device)
    z0_prior = torch.distributions.Normal(torch.Tensor([0.0]).to(device), torch.Tensor([1.]).to(device))

    model = create_LatentODE_model(args, input_dim, z0_prior, obsrv_std, device)

    # Load the trained model
    if args.load:
        get_ckpt_model(args.load, model, device)
        print(f"Loaded model from {args.load}")
    else:
        raise ValueError("A model path must be provided via the --load argument.")

    # Evaluate the model
    with torch.no_grad():
        test_res = compute_loss_all_batches(
            model,
            data_obj["test_dataloader"],
            args,
            n_batches=data_obj["n_test_batches"],
            experimentID=None,
            device=device,
            n_traj_samples=10,
            kl_coef=1.0,
            data_obj=data_obj
        )

    # Print the results
    print("--- Test Results ---")
    for key, value in test_res.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: {value.item()}")
        else:
            print(f"{key}: {value}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Latent ODE Tester')
    parser.add_argument('--dataset', type=str, required=True, help="Dataset to load.")
    parser.add_argument('--load', type=str, required=True, help="Path to the trained model checkpoint.")

    # Arguments for dataset parsing
    parser.add_argument('--train_datasets', type=str, default='pccp', help='Datasets to use for training, separated by commas')
    parser.add_argument('--test_datasets', type=str, default='pccp', help='Datasets to use for testing, separated by commas')
    parser.add_argument('--train_data_path', type=str, help='Path to the training data file')
    parser.add_argument('--test_data_path', type=str, help='Path to the testing data file')
    parser.add_argument('--pharmac_path', type=str, help='Path to the pharmacokinetics data file')
    parser.add_argument('--auc_path', type=str, help='Path to the AUC data file')
    parser.add_argument('--exp', type=str, default="", help='Experiment ID for gen_tac')
    parser.add_argument('--seed', type = int, default = 15, help="Fix seed for reproducibility")
    parser.add_argument('--n',  type=int, default=100, help="Size of the dataset")
    parser.add_argument('--batch-size', type=int, default=200)
    parser.add_argument('--latent-ode', action='store_true', help="Run Latent ODE seq2seq model")
    parser.add_argument('-l', '--latents', type=int, default=6, help="Size of the latent state")
    parser.add_argument('--rec-dims', type=int, default=20, help="Dimensionality of the recognition model (ODE or RNN).")
    parser.add_argument('--rec-layers', type=int, default=1, help="Number of layers in ODE func in recognition ODE")
    parser.add_argument('--gen-layers', type=int, default=1, help="Number of layers in ODE func in generative ODE")
    parser.add_argument('-u', '--units', type=int, default=100, help="Number of units per layer in ODE func")
    parser.add_argument('-g', '--gru-units', type=int, default=100, help="Number of units per layer in each of GRU update networks")
    parser.add_argument('--z0-encoder', type=str, default='odernn', help="Type of encoder for Latent ODE model: odersenn or rnn")
    parser.add_argument('--classif', action='store_true', help="Include binary classification loss -- used for Physionet dataset for hospiral mortality")
    parser.add_argument('--extrap', action='store_true', help="Set extrapolation mode. If this flag is not set, run interpolation mode.")
    parser.add_argument('--timepoints', type=int, default=100, help="Total number of time-points")
    parser.add_argument('--max-t',  type=float, default=5., help="We subsample points in the interval [0, args.max_tp]")
    parser.add_argument('--noise-weight', type=float, default=0.04, help="Noise amplitude for generated traejctories")

    args = parser.parse_args()

    test_model(args)