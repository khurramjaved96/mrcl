import errno
import hashlib
import os
import os.path
import random
from collections import namedtuple
import logging
logger = logging.getLogger('experiment')
from torch.nn import functional as F
import numpy as np
import copy

transition = namedtuple('transition', 'state, next_state, action, reward, is_terminal')
import torch

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def freeze_layers(layers_to_freeze, maml):

    for name, param in maml.named_parameters():
        param.learn = True

    for name, param in maml.net.named_parameters():
        param.learn = True

    frozen_layers = []
    for temp in range(layers_to_freeze * 2):
        frozen_layers.append("net.vars." + str(temp))

    for name, param in maml.named_parameters():
        if name in frozen_layers:
            logger.info("RLN layer %s", str(name))
            param.learn = False

    list_of_names = list(filter(lambda x: x[1].learn, maml.named_parameters()))

    for a in list_of_names:
        logger.info("TLN layer = %s", a[0])

def log_accuracy(maml, my_experiment, iterator_test, device, writer, step):
    correct = 0
    torch.save(maml.net, my_experiment.path + "learner.model")
    for img, target in iterator_test:
        with torch.no_grad():
            img = img.to(device)
            target = target.to(device)
            logits_q = maml.net(img, vars=None)
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            correct += torch.eq(pred_q, target).sum().item() / len(img)
    writer.add_scalar('/metatrain/test/classifier/accuracy', correct / len(iterator_test), step)
    logger.info("Test Accuracy = %s", str(correct / len(iterator_test)))


class replay_buffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.location = 0
        self.buffer = []

    def add(self, *args):
        # Append when the buffer is not full but overwrite when the buffer is full
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(transition(*args))
        else:
            self.buffer[self.location] = transition(*args)

        # Increment the buffer location
        self.location = (self.location + 1) % self.buffer_size

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def sample_trajectory(self, batch_size):
        initial_index = random.randint(0, len(self.buffer) - batch_size)
        return self.buffer[initial_index: initial_index + batch_size]


class ReservoirSampler:
    def __init__(self, windows, buffer_size=5000):
        self.buffer = []
        self.location = 0
        self.buffer_size = buffer_size
        self.window = windows
        self.total_additions = 0

    def add(self, *args):
        self.total_additions += 1
        stuff_to_add = transition(*args)

        M = len(self.buffer)
        if M < self.buffer_size:
            self.buffer.append(stuff_to_add)
        else:
            i = random.randint(0, min(self.total_additions, self.window))
            if i < self.buffer_size:
                self.buffer[i] = stuff_to_add
        self.location = (self.location + 1) % self.buffer_size

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def sample_trajectory(self, batch_size):
        initial_index = random.randint(0, len(self.buffer) - batch_size)
        return self.buffer[initial_index: initial_index + batch_size]


def iterator_sorter(trainset, no_sort=True, random=True, pairs=False, classes=10):
    if no_sort:
        return trainset

    order = list(range(len(trainset.data)))
    np.random.shuffle(order)

    trainset.data = trainset.data[order]
    trainset.targets = np.array(trainset.targets)
    trainset.targets = trainset.targets[order]

    sorting_labels = np.copy(trainset.targets)
    sorting_keys = list(range(20, 20 + classes))
    if random:
        if not pairs:
            np.random.shuffle(sorting_keys)

    print("Order = ", [x - 20 for x in sorting_keys])
    for numb, key in enumerate(sorting_keys):
        if pairs:
            np.place(sorting_labels, sorting_labels == numb, key - (key % 2))
        else:
            np.place(sorting_labels, sorting_labels == numb, key)

    indices = np.argsort(sorting_labels)
    # print(indices)

    trainset.data = trainset.data[indices]
    trainset.targets = np.array(trainset.targets)
    trainset.targets = trainset.targets[indices]
    # print(trainset.targets)
    # print(trainset.targets )

    return trainset


def iterator_sorter_omni(trainset, no_sort=True, random=True, pairs=False, classes=10):
    return trainset


def remove_classes(trainset, to_keep):
    # trainset.data = trainset.data[order]
    trainset.targets = np.array(trainset.targets)
    # trainset.targets = trainset.targets[order]

    indices = np.zeros_like(trainset.targets)
    for a in to_keep:
        indices = indices + (trainset.targets == a).astype(int)
    indices = np.nonzero(indices)
    # logger.info(trainset.data[0])
    trainset.data = trainset.data[indices]
    trainset.targets = np.array(trainset.targets)
    trainset.targets = trainset.targets[indices]

    return trainset


def remove_classes_omni(trainset, to_keep):

    trainset = copy.deepcopy(trainset)
    # trainset.data = trainset.data[order]
    trainset.targets = np.array(trainset.targets)
    # trainset.targets = trainset.targets[order]

    indices = np.zeros_like(trainset.targets)
    for a in to_keep:
        indices = indices + (trainset.targets == a).astype(int)
    indices = np.nonzero(indices)
    trainset.data = [trainset.data[i] for i in indices[0]]
    # trainset.data = trainset.data[indices]
    trainset.targets = np.array(trainset.targets)
    trainset.targets = trainset.targets[indices]

    return trainset


def check_integrity(fpath, md5):
    if not os.path.isfile(fpath):
        return False
    md5o = hashlib.md5()
    with open(fpath, 'rb') as f:
        # read in 1MB chunks
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            md5o.update(chunk)
    md5c = md5o.hexdigest()
    if md5c != md5:
        return False
    return True


def download_url(url, root, filename, md5):
    from six.moves import urllib

    root = os.path.expanduser(root)
    fpath = os.path.join(root, filename)

    try:
        os.makedirs(root)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise

    # downloads file
    if os.path.isfile(fpath) and check_integrity(fpath, md5):
        print('Using downloaded and verified file: ' + fpath)
    else:
        try:
            print('Downloading ' + url + ' to ' + fpath)
            urllib.request.urlretrieve(url, fpath)
        except:
            if url[:5] == 'https':
                url = url.replace('https:', 'http:')
                print('Failed download. Trying https -> http instead.'
                      ' Downloading ' + url + ' to ' + fpath)
                urllib.request.urlretrieve(url, fpath)


def list_dir(root, prefix=False):
    """List all directories at a given root
    Args:
        root (str): Path to directory whose folders need to be listed
        prefix (bool, optional): If true, prepends the path to each result, otherwise
            only returns the name of the directories found
    """
    root = os.path.expanduser(root)
    directories = list(
        filter(
            lambda p: os.path.isdir(os.path.join(root, p)),
            os.listdir(root)
        )
    )

    if prefix is True:
        directories = [os.path.join(root, d) for d in directories]

    return directories


def list_files(root, suffix, prefix=False):
    """List all files ending with a suffix at a given root
    Args:
        root (str): Path to directory whose folders need to be listed
        suffix (str or tuple): Suffix of the files to match, e.g. '.png' or ('.jpg', '.png').
            It uses the Python "str.endswith" method and is passed directly
        prefix (bool, optional): If true, prepends the path to each result, otherwise
            only returns the name of the files found
    """
    root = os.path.expanduser(root)
    files = list(
        filter(
            lambda p: os.path.isfile(os.path.join(root, p)) and p.endswith(suffix),
            os.listdir(root)
        )
    )

    if prefix is True:
        files = [os.path.join(root, d) for d in files]

    return files


def resize_image(img, factor):
    '''

    :param img:
    :param factor:
    :return:
    '''
    img2 = np.zeros(np.array(img.shape) * factor)

    for a in range(0, img.shape[0]):
        for b in range(0, img.shape[1]):
            img2[a * factor:(a + 1) * factor, b * factor:(b + 1) * factor] = img[a, b]
    return img2



def get_run(arg_dict, rank=0):
    # print(arg_dict)
    combinations =[]

    if isinstance(arg_dict["seed"], list):
        combinations.append(len(arg_dict["seed"]))


    for key in arg_dict.keys():
        if isinstance(arg_dict[key], list) and not key=="seed":
            combinations.append(len(arg_dict[key]))

    total_combinations = np.prod(combinations)
    selected_combinations = []
    for base in combinations:
        selected_combinations.append(rank%base)
        rank = int(rank/base)

    counter=0
    result_dict = {}

    result_dict["seed"] = arg_dict["seed"]
    if isinstance(arg_dict["seed"], list):
        result_dict["seed"] = arg_dict["seed"][selected_combinations[0]]
        counter += 1
    #

    for key in arg_dict.keys():
        if key !="seed":
            result_dict[key] = arg_dict[key]
            if isinstance(arg_dict[key], list):
                result_dict[key] = arg_dict[key][selected_combinations[counter]]
                counter+=1

    logger.info("Parameters %s", str(result_dict))
    # 0/0
    return result_dict

import torch
#
def construct_set(iterators, sampler, steps, shuffle=True):
    x_traj = []
    y_traj = []

    x_rand = []
    y_rand = []


    id_map = list(range(sampler.capacity - 1))
    if shuffle:
        random.shuffle(id_map)

    for id, it1 in enumerate(iterators):
        id_mapped = id_map[id]
        for inner in range(steps):
            x, y = sampler.sample_batch(it1, id_mapped, 10)
            x_traj.append(x)
            y_traj.append(y)
        #
        x, y = sampler.sample_batch(it1, id_mapped, 10)
        x_rand.append(x)
        y_rand.append(y)

    x_rand = torch.stack([torch.cat(x_rand)])
    y_rand = torch.stack([torch.cat(y_rand)])

    x_traj = torch.stack(x_traj)
    y_traj = torch.stack(y_traj)


    return x_traj, y_traj, x_rand, y_rand
