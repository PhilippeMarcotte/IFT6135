import torch


def save_model(model, epoch, trainingLosses, trainingAccuracies, validationLosses, validationAccuracies):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'trainingLosses': trainingLosses,
        'trainingAccuracies': trainingAccuracies,
        'validationLosses': validationLosses,
        'validationAccuracies': validationAccuracies
    }, "checkpoint.pth")

def load_model():
    return torch.load("checkpoint.pth")