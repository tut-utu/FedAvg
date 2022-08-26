from model.master import master
from torchvision.datasets import MNIST
from torchvision import transforms
import torch
import random
from train import train, test


def main():
    k = 5
    # number of clients
    train_round = 3
    C = 0.6  # percent of participate clients
    B = 100  # local batch size
    E = 2  # local epoch
    lr = 0.01  # learning rate

    master()

    transform = transforms.Compose([transforms.ToTensor()])
    mnist_trainset = MNIST(root='./data', train=True, download=True,
                           transform=transform)  # read training/testing dataset
    mnist_testset = MNIST(root='./data', train=False, download=True, transform=transform)

    len_data = int(len(mnist_trainset) / k)
    train_set = torch.utils.data.random_split(mnist_trainset,
                                              [len_data for _ in range(k)])  # separate training data to each client
    test(0, mnist_testset)
    for n in range(1, train_round + 1):
        m = max(int(C * k), 1)  # number of clients in this round
        S_t = random.sample([i for i in range(k)], k=m)
        print("round %s, participate clients: " % n, *S_t)
        train(S_t, n, train_set, B, E, lr)
        test(n, mnist_testset)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
