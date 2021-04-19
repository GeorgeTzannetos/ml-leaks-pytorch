from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler
import sys

# We split the total dataset to four equal parts. Half of the dataset is Dshadow, from which half is Dshadow_train
# used for training the shadow model, and the other half is Dshadow_out for evaluation. The second half of the total
# dataset is Dtarget with is used for attack evaluation. The target model is trained on the half of that Dtarget_train,
# which serve as the members of target's training data, and the other half is the Dtarget_out as the non-member data
# points.


def subsetloader(ls_indices, start, end, trainset, batch_size):
    """
    Function that takes a list of indices and a certain split with start and end, creates a randomsampler and returns
    a subset dataloader with this sampler
    """
    ids = ls_indices[start:end]
    sampler = SubsetRandomSampler(ids)
    loader = DataLoader(trainset, batch_size=batch_size, sampler=sampler)
    return loader


#  The main dataloader used. Can return 4 differents dataloaders for each different split based on the paper's
#  methodology and a testloader, for both CIFAR10 and MNIST.


def dataloader(dataset="cifar", batch_size_train=8, batch_size_test=1000, split_dataset="shadow_train"):
    """
    Dataloader function that returns dataloader of a subset for train and test data of CIFAR10 or MNIST.
    """
    try:
        if dataset == "cifar":

            transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
            trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
            testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
            testloader = DataLoader(testset, batch_size=batch_size_test, shuffle=False)

        elif dataset == "mnist":

            transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize((0.1307,), (0.3081,))])
            trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
            testset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
            testloader = DataLoader(testset, batch_size=batch_size_test, shuffle=False)
        else:
            raise NotAcceptedDataset

    except NotAcceptedDataset:
        print('Dataset Error. Choose "cifar" or "mnist"')
        sys.exit()

    total_size = len(trainset)
    split1 = total_size // 4
    split2 = split1 * 2
    split3 = split1 * 3

    indices = [*range(total_size)]

    if split_dataset == "shadow_train":
        return subsetloader(indices, 0, split1, trainset, batch_size_train)

    elif split_dataset == "shadow_out":
        return subsetloader(indices, split1, split2, trainset, batch_size_train)

    elif split_dataset == "target_train":
        return subsetloader(indices, split2, split3, trainset, batch_size_train)

    elif split_dataset == "target_out":
        return subsetloader(indices, split3, total_size, trainset, batch_size_train)

    else:
        return testloader


# Just a simple custom exception that is raised when the dataset argument is not accepted


class NotAcceptedDataset(Exception):
    """Not accepted dataset as argument"""
    pass


