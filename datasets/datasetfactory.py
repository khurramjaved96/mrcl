import torchvision.transforms as transforms
import datasets.celeba as celeb
import datasets.omniglot as om


class DatasetFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_dataset(name, train=True, path=None, background=True, all=False):

        if name == "omniglot":
            train_transform = transforms.Compose(
                [transforms.Resize((84, 84)),
                 transforms.ToTensor()])
            if path is None:
                return om.Omniglot("../data/omni", background=background, download=True, train=train,
                                   transform=train_transform, all=all)
            else:
                return om.Omniglot(path, background=background, download=True, train=train,
                                   transform=train_transform, all=all)

        elif name == "celeba":
            train_transform = transforms.Compose(
                [transforms.Resize((84, 84)),
                 transforms.ToTensor()])
            if train:
                if path is None:
                    return celeb.CelebA("../data/celeba", 'train','identity', transform=train_transform, download=True)
                else:
                    return celeb.CelebA(path, 'train', 'identity', transform=train_transform, download=True)
            else:
                assert(False)
                return celeb.CelebA("../data/celeba", 'val', 'identity', download=True)


        else:
            print("Unsupported Dataset")
            assert False
