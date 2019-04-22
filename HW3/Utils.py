import torch


def save_model(model, optimizer, epoch, trainingLoss, validationLoss):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'trainingLoss': trainingLoss,
        'validationLoss': validationLoss,
    }, "checkpoint.pth")


def load_model():
    return torch.load("checkpoint.pth")
