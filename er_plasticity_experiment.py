import logging

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

import datasets.datasetfactory as df
import model.modelfactory as mf
import utils.utils as utils
from configs.default import er_argparse
from experiment.experiment import experiment
from model import learner
from tqdm import tqdm

logger = logging.getLogger('experiment')


def main():
    p = er_argparse.ERParser()
    total_seeds = len(p.parse_known_args()[0].seed)
    rank = p.parse_known_args()[0].rank
    all_args = vars(p.parse_known_args()[0])

    args = utils.get_run(vars(p.parse_known_args()[0]), rank)

    utils.set_seed(args["seed"])

    my_experiment = experiment(args["name"], args, "../results/", commit_changes=args["commit"],
                               rank=int(rank / total_seeds),
                               seed=total_seeds)
    my_experiment.results["all_args"] = all_args

    writer = SummaryWriter(my_experiment.path + "tensorboard")

    logger = logging.getLogger('experiment')

    total_classes = 4

    keep = np.random.choice(list(range(650)), total_classes, replace=False)
    # np.random.shuffle(keep)



    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    results_mem_size = {}

    mem_size = args["memory"]
    lr = args["lr"]

    config = mf.ModelFactory.get_model("na", args["dataset"]+"-er")
    my_model = learner.Learner(config)

    my_model = my_model.to(device)

    opt = torch.optim.SGD(my_model.parameters(), lr=lr, weight_decay=0.0001)
    import module.replay as rep
    res_sampler = rep.ReservoirSampler(mem_size)
    keep =[]
    for cycle in range(0, 2):
        keep_old = keep
        keep = np.random.choice(list(range(650)), total_classes, replace=False)
        keep_test = np.concatenate((keep,keep_old))
        dataset = utils.remove_classes_omni(
            df.DatasetFactory.get_dataset("omniglot", train=True, background=False, path=args["dataset_path"]), keep)
        iterator_sorted = torch.utils.data.DataLoader(
            utils.iterator_sorter_omni(dataset, False, classes=total_classes),
            batch_size=1,
            shuffle=False, num_workers=2)
        dataset = utils.remove_classes_omni(
            df.DatasetFactory.get_dataset("omniglot", train=True, background=False, path=args["dataset_path"]),
            keep_test)
        iterator = torch.utils.data.DataLoader(dataset, batch_size=32,
                                               shuffle=False, num_workers=1)

        for img, y in tqdm(iterator_sorted):
            orig_img, orig_y = img.to(device), y.to(device)
            if mem_size > 0:
                res_sampler.update_buffer(zip(img, y))
                res_sampler.update_observations(len(img))
                img = img.to(device)
                y = y.to(device)
                img2, y2 = res_sampler.sample_buffer(16)
                img2 = img2.to(device)
                y2 = y2.to(device)

                img = torch.cat([img, img2], dim=0)
                y = torch.cat([y, y2], dim=0)
                # print(y)
            else:
                img = img.to(device)
                y = y.to(device)

            if args["mask"] and cycle==1:
                pred = my_model(img)
                opt.zero_grad()
                loss = F.cross_entropy(pred, y)
                loss.backward()
                mask_list = []
                for weights in my_model.parameters():
                    # print("Printing grad")
                    # print(weights.grad)
                    grad_vector = weights.grad.flatten().detach().cpu().numpy()
                    grad_vector = np.sort(np.abs(grad_vector))
                    # print(grad_vector)
                    mask_val = grad_vector[int(len(grad_vector)*0.90)]
                    mask = (torch.abs(weights.grad.detach())>mask_val).int()
                    mask_list.append(mask)
                opt.zero_grad()
                pred = my_model(orig_img)
                # print(torch.argmax(pred, axis=1), orig_y)
                loss = F.cross_entropy(pred, orig_y)
                loss.backward()
                for counter, weights in enumerate(my_model.parameters()):
                    weights.grad = weights.grad*mask_list[counter]
                    # print("Grad = ", weights.grad)
                    # print("Same grad = ", (weights.grad>0).int().sum())
                    # print("Opposite grad = ", (weights.grad < 0).int().sum())
                    # print("Total weight = ", (weights.grad>-10000).int().sum())
                    # print("/n")
                    # print(weights.grad)
                    weights.data = weights.data - args["lr"]*weights.grad


            else:
                pred = my_model(img)
                opt.zero_grad()
                loss = F.cross_entropy(pred, y)
                loss.backward()
                opt.step()

        logger.info("Result after one epoch for LR = %f", lr)
        correct = 0
        for img, target in iterator:
            img = img.to(device)
            target = target.to(device)
            logits_q = my_model(img, vars=None, bn_training=False, feature=False)

            pred_q = (logits_q).argmax(dim=1)

            correct += torch.eq(pred_q, target).sum().item() / len(img)

        logger.info("Accuracy = %s", str(correct / len(iterator)))

        accuracy = str(correct / len(iterator))
        my_experiment.results["Accuracy"] = accuracy
        my_experiment.store_json()
        writer.close()


if __name__ == '__main__':
    main()
