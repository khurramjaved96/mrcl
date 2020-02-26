import logging
import random

import torch
from torch.utils.tensorboard import SummaryWriter

import configs.classification.classification_parser as reg_parser
import datasets.datasetfactory as df
import model.modelfactory as mf
import utils.utils as utils
from experiment.experiment import experiment
from model.meta_classification import MetaLearingClassification

logger = logging.getLogger('experiment')


def main():
    p = reg_parser.Parser()
    total_seeds = len(p.parse_known_args()[0].seed)
    rank = p.parse_known_args()[0].rank
    all_args = vars(p.parse_known_args()[0])
    print("All args = ", all_args)

    args = utils.get_run(vars(p.parse_known_args()[0]), rank)

    utils.set_seed(args["seed"])

    my_experiment = experiment(args["name"], args, "../results/", commit_changes=False,
                               rank=int(rank / total_seeds),
                               seed=total_seeds)
    my_experiment.results["all_args"] = all_args

    writer = SummaryWriter(my_experiment.path + "tensorboard")

    logger = logging.getLogger('experiment')

    print("Selected args", args)

    # Using first 963 classes of the omniglot as the meta-training set
    args["classes"] = list(range(963))

    args["traj_classes"] = args["classes"]

    dataset = df.DatasetFactory.get_dataset(args["dataset"], background=True, train=True, all=True, path=args["path"])

    config = mf.ModelFactory.get_model("na", args["dataset"])

    gpu_to_use = rank % args["gpus"]
    if torch.cuda.is_available():
        device = torch.device('cuda:' + str(gpu_to_use))
        logger.info("Using gpu : %s", 'cuda:' + str(gpu_to_use))
    else:
        device = torch.device('cpu')

    maml = MetaLearingClassification(args, config).to(device)

    inner_step = 0
    for step in range(args["epoch"]):
        logger.info(" \n #### STEP %d #### \n", step)
        list_of_tasks = list(range(len(args["classes"])))
        labels = list(range(len(args["classes"])))
        random.shuffle(list_of_tasks)
        list_of_tasks = list_of_tasks[0:args["capacity"]]
        labels = labels[0:args["capacity"]]
        if step % 200 == 199:
            torch.save(maml.net, my_experiment.path + "net.model")
        if step % 5 == 4:

            inner_step += 1
            x_spt, y_spt = maml.sample_training_data(list_of_tasks, labels, dataset.images_data,
                                                     20, limit=20000)
            logger.info("Evaluation x_spt shape %s", str(x_spt.shape))
            x_spt, y_spt = x_spt.to(device), y_spt.to(device)

            maml.update_TLN(x_spt.unsqueeze(1), y_spt.unsqueeze(1))

            accuracy = maml.eval_accuracy_images(x_spt.view(-1, 100, 1, 84, 84), y_spt.view(-1, 100))
            logger.info("%d Evaluation accuracy =  %f", step, accuracy)
            writer.add_scalar('/metatrain/train/valaccuracy', accuracy,
                              step)

        else:

            for current_tasks in range(0, len(list_of_tasks), args["tasks"]):
                inner_step += 1
                ids_spt = list_of_tasks[current_tasks:current_tasks + args["tasks"]]
                labels_spt = labels[current_tasks:current_tasks + args["tasks"]]

                ids_qry = list_of_tasks[0:current_tasks + args["tasks"]]
                labels_qry = labels[0:current_tasks + args["tasks"]]

                x_spt, y_spt = maml.sample_training_data(ids_spt, labels_spt, dataset.images_data,
                                                         args["update_step"])

                x_qry, y_qry = maml.sample_training_data(ids_qry, labels_qry, dataset.images_data,
                                                         args["update_step"])

                x_qry, y_qry = torch.cat([x_qry, x_spt]), torch.cat([y_qry, y_spt])

                x_qry, x_spt = x_qry.unsqueeze(0), x_spt.unsqueeze(1)

                y_spt, y_qry = y_spt.unsqueeze(1), y_qry.unsqueeze(0)
                # print(x_spt.shape, y_spt.shape, x_qry.shape, y_qry.shape)

                x_spt, y_spt, x_qry, y_qry = x_spt.to(device), y_spt.to(device), x_qry.to(device), y_qry.to(device)
                #
                accs, loss = maml(x_spt, y_spt, x_qry, y_qry)

                writer.add_scalar('/metatrain/train/accuracy', loss[-1].cpu().detach().item(), inner_step)
                writer.add_scalar('/metatrain/train/runningaccuracy', accs[-1],
                                  inner_step)

                maml.update_TLN(x_spt, y_spt)

                if inner_step % 100 == 0:
                    logger.info("%d Accuracy before %f, after %f", current_tasks, accs[0], accs[-1])
                    logger.info("%d Loss before %s, after %s", current_tasks, str(loss[0].cpu().detach().item()),
                                str(loss[-1].cpu().detach().item()))

            logger.info("%d Accuracy after = %f", step, accs[-1])
            writer.add_scalar('/metatrain/train/loss_end', loss[-1].cpu().detach().item(), step)
            writer.add_scalar('/metatrain/train/accuracy_end', accs[-1], step)
            #

            # # Evaluation during training for sanity checks
            # if step % 40 == 39:
            #     writer.add_scalar('/metatrain/train/accuracy', accs[-1], step)
            #     logger.info('step: %d \t training acc %s', step, str(accs))
            # if step % 300 == 299:
            #     utils.log_accuracy(maml, my_experiment, iterator_test, device, writer, step)
            #     utils.log_accuracy(maml, my_experiment, iterator_train, device, writer, step)


#
if __name__ == '__main__':
    main()
