from model.cnn import cnn
import torch


def master():
    print('Initializing master model')
    global_model = cnn()
    torch.save(global_model.state_dict(), "global/round_0.pt")

