import torchvision.transforms as transforms
import datasets.celeba as celeb
import datasets.omniglot as om
import datasets.imagenet as im

class DatasetFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_dataset(name, train=True, path=None, background=True, all=False, RLN=None):

        if name == "omniglot":
            train_transform = transforms.Compose(
                [transforms.Resize((32, 32)),
                 transforms.ToTensor()])
            if path is None:
                return om.Omniglot("../data/omni", background=background, download=True, train=train,
                                   transform=train_transform, all=all)
            else:
                return om.Omniglot(path, background=background, download=True, train=train,
                                   transform=train_transform, all=all)

        elif name == "imagenet":
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
            train_transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
            if train:
                return im.ImageNet(path, split='train', download=True, transform=train_transform)
            else:
                return im.ImageNet(path, split='val', download=False, transform=train_transform)

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

        elif name == "celeba-linear":
            train_transform = transforms.Compose(
                [transforms.Resize((84, 84)),
                 transforms.ToTensor()])
            if train:
                if path is None:
                    return celeb.CelebA_Linear("../data/celeba", 'valid','attr', transform=train_transform, download=True, RLN=RLN)
                else:
                    return celeb.CelebA_Linear(path, 'valid', 'attr', transform=train_transform, download=True, RLN=RLN)
            else:
                if path is None:
                    return celeb.CelebA_Linear("../data/celeba", 'test', 'attr', transform=train_transform,
                                               download=True, RLN=RLN)
                else:
                    return celeb.CelebA_Linear(path, 'test', 'attr', transform=train_transform, download=True,
                                               RLN=RLN)



        else:
            print("Unsupported Dataset")
            assert False
