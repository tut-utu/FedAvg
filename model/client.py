from model.cnn import cnn
import torch
from torch.utils.data import DataLoader
import torch.nn as nn


def c_train(net, train_iter, lr):
    """
    Train a client model
    :return:
    """
    loss = nn.CrossEntropyLoss()
    net.train()
    trainer = torch.optim.SGD(net.parameters(), lr=lr)
    loss_total = 0
    for X, y in train_iter:
        l = loss(net(X), y)
        loss_total += l.item()
        trainer.zero_grad()
        l.backward()
        trainer.step()

    return loss_total


def client(t: int, data, state_list, c_number: int, B: int = 100, E: int = 2, lr=0.01):
    """
    Open a client and train locally
    :param c_number: index of client
    :param t: round
    :param data: train data
    :param state_list: global model
    :param B: local batch size
    :param E: local epoch
    :param lr: learning rate
    :return: NA
    """
    print('Initializing client model %s' % c_number)
    client_model = cnn()
    # load global model
    client_model.load_state_dict(torch.load("global/round_%s.pt" % (t - 1)), strict=True)
    train_iter = DataLoader(data,
                            batch_size=B,
                            shuffle=True,
                            num_workers=1)
    for i in range(E):
        # local epoch E
        loss = c_train(client_model, train_iter, lr)
        print("Client %s , round %s, training epoch %s loss: %s" % (c_number, t, i, loss))
    state_list.append(client_model.state_dict())
