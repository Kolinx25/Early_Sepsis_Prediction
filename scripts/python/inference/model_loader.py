import torch

from models.tcn import SepsisTCN


MODEL_PATH = "models/best_tcn_y12.pt"


def load_model():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SepsisTCN(input_channels=69).to(device)

    checkpoint = torch.load(MODEL_PATH, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()

    print("TCN model loaded")

    print("Checkpoint epoch:", checkpoint["epoch"])

    print("Validation AUPRC:", checkpoint["best_auprc"])

    return model, device
