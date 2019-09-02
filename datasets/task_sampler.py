import copy
import logging

import numpy as np
import torch

logger = logging.getLogger("experiment")


class SamplerFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_sampler(dataset, tasks, trainset, testset=None, capacity=None):
        if "omni" in dataset:
            return OmniglotSampler(tasks, trainset, testset)
        elif "celeba" in dataset:
            return CelebaSampler(tasks, trainset)



class CelebaSampler:
    # Class to sample tasks
    def __init__(self, tasks, trainset):
        self.tasks = tasks
        self.task_sampler = SampleCeleba(trainset)
        self.task_sampler.add_complete_iteraetor(list(range(0, int(len(self.tasks)))))

    def get_complete_iterator(self):
        return self.task_sampler.complete_iterator

    def sample_random(self):
        return self.task_sampler.get([np.random.choice(self.tasks)])

    def filter_upto(self, task):
        return self.task_sampler.filter_upto(task)

    def sample_task_no_cache(self, t, train=True):
        return self.task_sampler.get_no_cache(t)

    def sample_task(self, t):
        return self.task_sampler.get(t)

    def sample_tasks(self, t, train=False):
        # assert(false)
        dataset = self.task_sampler.get_task_trainset(t)
        train_iterator = torch.utils.data.DataLoader(dataset,
                                                     batch_size=1,
                                                     shuffle=True, num_workers=1)
        return train_iterator

class SampleCeleba:

    def __init__(self, trainset):
        self.task_iterators = []
        self.trainset = trainset
        self.iterators = {}
        self.test_iterators = {}

    def add_complete_iteraetor(self, tasks):
        dataset = self.get_task_trainset(tasks)
        train_iterator = torch.utils.data.DataLoader(dataset,
                                                     batch_size=12,
                                                     shuffle=True, num_workers=1)
        self.complete_iterator = train_iterator


    def add_task_iterator(self, task):
        dataset = self.get_task_trainset([task])

        train_iterator = torch.utils.data.DataLoader(dataset,
                                                     batch_size=1,
                                                     shuffle=True)
        self.iterators[task] = train_iterator
        print("Task %d has been added to the list" % task)
        return train_iterator

    def get_no_cache(self, tasks):

        dataset = self.get_task_trainset(tasks)

        train_iterator = torch.utils.data.DataLoader(dataset,
                                                     batch_size=12,
                                                     shuffle=True, num_workers=1)
        return train_iterator


    def get(self, tasks):

        for task in tasks:
            if task in self.iterators:
                return self.iterators[task]
            else:
                return self.add_task_iterator(task)


    def get_task_trainset(self, task):


        trainset = copy.deepcopy(self.trainset)

        class_labels = trainset.identity.squeeze().numpy()

        indices = np.zeros_like(class_labels)
        for a in task:
            indices = indices + (class_labels == a).astype(int)
        indices = np.nonzero(indices)[0]


        trainset.identity = trainset.identity[indices]
        trainset.filename = [trainset.filename[x] for x in indices]
        trainset.attr = trainset.attr[indices]

        return trainset

    def filter_upto(self, task):

        trainset = copy.deepcopy(self.trainset)
        trainset.data = trainset.data[trainset.data['target'] <= task]

        return trainset



class OmniglotSampler:
    # Class to sample tasks
    def __init__(self, tasks, trainset, testset):
        self.tasks = tasks
        self.task_sampler = SampleOmni(trainset, testset)
        self.task_sampler.add_complete_iteraetor(list(range(0, int(len(self.tasks)))))

    def get_complete_iterator(self):
        return self.task_sampler.complete_iterator

    def get_another_complete_iterator(self):
        return self.task_sampler.another_complete_iterator

    def sample_random(self):
        return self.task_sampler.get([np.random.choice(self.tasks)])

    def filter_upto(self, task):
        return self.task_sampler.filter_upto(task)

    def sample_task(self, t, train=True):
        return self.task_sampler.get(t, train)

    def sample_task_no_cache(self, t, train=True):
        return self.task_sampler.get_no_cache(t, train)

    def sample_tasks(self, t, train=False):
        # assert(false)
        dataset = self.task_sampler.get_task_trainset(t, train)
        train_iterator = torch.utils.data.DataLoader(dataset,
                                                     batch_size=1,
                                                     shuffle=True, num_workers=1)
        return train_iterator

class SampleOmni:

    def __init__(self, trainset, testset):
        self.task_iterators = []
        self.trainset = trainset
        self.testset = testset
        self.iterators = {}
        self.test_iterators = {}

    def add_complete_iteraetor(self, tasks):
        dataset = self.get_task_trainset(tasks, True)
        # dataset = self.get_task_testset(tasks)
        train_iterator = torch.utils.data.DataLoader(dataset,
                                                     batch_size=10,
                                                     shuffle=True, num_workers=1)
        self.complete_iterator = train_iterator
        logger.info("Len of complete iterator = %d", len(self.complete_iterator) * 256)

        train_iterator2 = torch.utils.data.DataLoader(dataset,
                                                      batch_size=1,
                                                      shuffle=True, num_workers=1)

        self.another_complete_iterator = train_iterator2

    def add_task_iterator(self, task, train):
        dataset = self.get_task_trainset([task], train)

        train_iterator = torch.utils.data.DataLoader(dataset,
                                                     batch_size=1,
                                                     shuffle=True, num_workers=1)
        self.iterators[task] = train_iterator
        print("Task %d has been added to the list" % task)
        return train_iterator


    def get_no_cache(self, tasks, train):

        dataset = self.get_task_trainset(tasks, train)

        train_iterator = torch.utils.data.DataLoader(dataset,
                                                     batch_size=10,
                                                     shuffle=True, num_workers=1)
        return train_iterator


    def get(self, tasks, train):
        if train:
            for task in tasks:
                if task in self.iterators:
                    return self.iterators[task]
                else:
                    return self.add_task_iterator(task, True)
        else:
            for task in tasks:
                if tasks in self.test_iterators:
                    return self.test_iterators[task]
                else:
                    return self.add_task_iterator(task, False)

    def get_task_trainset(self, task, train):

        if train:
            trainset = copy.deepcopy(self.trainset)
        else:
            trainset = copy.deepcopy(self.testset)
        class_labels = np.array([x[1] for x in trainset._flat_character_images])

        indices = np.zeros_like(class_labels)
        for a in task:
            indices = indices + (class_labels == a).astype(int)
        indices = np.nonzero(indices)

        trainset._flat_character_images = [trainset._flat_character_images[i] for i in indices[0]]
        trainset.data = [trainset.data[i] for i in indices[0]]
        trainset.targets = [trainset.targets[i] for i in indices[0]]

        return trainset

    def get_task_testset(self, task):

        trainset = copy.deepcopy(self.testset)
        class_labels = np.array([x[1] for x in trainset._flat_character_images])

        indices = np.zeros_like(class_labels)
        for a in task:
            indices = indices + (class_labels == a).astype(int)
        indices = np.nonzero(indices)

        trainset._flat_character_images = [trainset._flat_character_images[i] for i in indices[0]]
        trainset.data = [trainset.data[i] for i in indices[0]]
        trainset.targets = [trainset.targets[i] for i in indices[0]]

        return trainset

    def filter_upto(self, task):

        trainset = copy.deepcopy(self.trainset)
        trainset.data = trainset.data[trainset.data['target'] <= task]

        return trainset
