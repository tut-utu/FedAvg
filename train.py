import copy
import torch
from model.client import client
from multiprocessing import Process, Manager
from model.cnn import cnn
from torch.utils.data import DataLoader


def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def train(clients: list, t: int, data: list, B: int, E: int, lr):
    """

    :param clients: list of clients
    :param t: number of round
    :param data: list of data
    :return: NA
    """
    process_list = []
    with Manager() as manager:
        state_list = manager.list()
        for i in clients:
            p = Process(target=client, args=(t, data[i], state_list, i, B, E, lr))
            p.start()
            process_list.append(p)

        for p in process_list:
            p.join()

        new_global_mode = average_weights(state_list)
        torch.save(new_global_mode, "global/round_%s.pt" % t)


def test(t: int, data):
    global_model = cnn()
    global_model.load_state_dict(torch.load("global/round_%s.pt" % t), strict=True)
    test_iter = DataLoader(data,
                           batch_size=124,
                           shuffle=False,
                           num_workers=1)

    global_model.eval()
    correct = 0
    with torch.no_grad():
        for X, y in test_iter:
            y_hat = global_model(X)
            pred = y_hat.argmax(dim=1, keepdim=True)
            correct += pred.eq(y.view_as(pred)).sum().item()
    accuracy = correct / len(test_iter.dataset)
    print("Accuracy for round %s : %s" % (t, accuracy))
