import torch

from inference.model_loader import load_model


model, device = load_model()


def predict_risk(sequence):
    """
    Input:

        sequence:
            list or array

            shape:
            (12,69)


    Output:

        probability of sepsis
        within next 12 hours

    """

    X = torch.tensor(sequence, dtype=torch.float32)

    X = X.unsqueeze(0)

    X = X.to(device)

    with torch.no_grad():
        logits = model(X)

        probability = torch.sigmoid(logits)

    return float(probability.cpu().item())
