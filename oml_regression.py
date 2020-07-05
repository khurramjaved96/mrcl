import logging

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

import configs.regression.reg_parser as reg_parser
import datasets.task_sampler as ts
import model.modelfactory as mf
from experiment.experiment import experiment
from model.meta_learner import MetaLearnerRegression
from utils import utils

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

    tasks = list(range(400))

    sampler = ts.SamplerFactory.get_sampler("Sin", tasks, None, capacity=args["capacity"] + 1)

    model_config = mf.ModelFactory.get_model(args["model"], "Sin", input_dimension=args["capacity"] + 1,
                                             output_dimension=1,
                                             width=args["width"])

    context_backbone_config = None

    gpu_to_use = rank % args["gpus"]
    if torch.cuda.is_available():
        device = torch.device('cuda:' + str(gpu_to_use))
        logger.info("Using gpu : %s", 'cuda:' + str(gpu_to_use))
    else:
        device = torch.device('cpu')

    metalearner = MetaLearnerRegression(args, model_config, context_backbone_config).to(device)

    tmp = filter(lambda x: x.requires_grad, metalearner.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    logger.info('Total trainable tensors: %d', num)
    #
    running_meta_loss = 0
    adaptation_loss = 0
    loss_history = []
    adaptation_loss_history = []
    adaptation_running_loss_history = []
    meta_steps_counter = 0
    LOG_INTERVAL = 50

    for step in range(args["epoch"]):

        if step % LOG_INTERVAL == 0:
            logger.debug("####\t STEP %d \t####", step)


        net = metalearner.net

        meta_steps_counter += 1

        t1 = np.random.choice(tasks, args["tasks"], replace=False)

        iterators = []
        for t in t1:
            iterators.append(sampler.sample_task([t]))

        x_traj_meta, y_traj_meta, x_rand_meta, y_rand_meta = utils.construct_set(iterators, sampler,
                                                                                 steps=args[
                                                                                     "update_step"])

        if torch.cuda.is_available():
            x_traj_meta, y_traj_meta, x_rand_meta, y_rand_meta = x_traj_meta.to(device), y_traj_meta.to(
                device), x_rand_meta.to(
                device), y_rand_meta.to(
                device)

        meta_loss = metalearner(x_traj_meta, y_traj_meta, x_rand_meta, y_rand_meta)
        loss_history.append(meta_loss[-1].detach().cpu().item())

        running_meta_loss = running_meta_loss * 0.97 + 0.03 * meta_loss[-1].detach().cpu()
        running_meta_loss_fixed = running_meta_loss / (1 - (0.97 ** (meta_steps_counter)))
        writer.add_scalar('/metatrain/train/accuracy', meta_loss[-1].detach().cpu(), meta_steps_counter)
        writer.add_scalar('/metatrain/train/runningaccuracy', running_meta_loss_fixed,
                          meta_steps_counter)

        if step % LOG_INTERVAL == 0:
            if running_meta_loss > 0:
                logger.info("Running meta loss = %f", running_meta_loss_fixed.item())

            with torch.no_grad():
                t1 = np.random.choice(tasks, args["tasks"], replace=False)

                iterators = []
                for t in t1:
                    iterators.append(sampler.sample_task([t]))

                x_traj, y_traj, x_rand, y_rand = utils.construct_set(iterators, sampler, steps=args["update_step"])

                if torch.cuda.is_available():
                    x_traj, y_traj, x_rand, y_rand = x_traj.to(device), y_traj.to(device), x_rand.to(device), y_rand.to(
                        device)

                logits = net(x_rand[0], vars=None)
                logits_select = []
                for no, val in enumerate(y_rand[0, :, 1].long()):
                    logits_select.append(logits[no, val])
                logits = torch.stack(logits_select).unsqueeze(1)

                current_adaptation_loss = F.mse_loss(logits, y_rand[0, :, 0].unsqueeze(1))
                adaptation_loss_history.append(current_adaptation_loss.detach().item())
                adaptation_loss = adaptation_loss * 0.97 + current_adaptation_loss.detach().cpu().item() * 0.03
                adaptation_loss_fixed = adaptation_loss / (1 - (0.97 ** (step + 1)))
                adaptation_running_loss_history.append(adaptation_loss_fixed)

                logger.info("Adaptation loss = %f", current_adaptation_loss)

                if step % LOG_INTERVAL == 0:
                    logger.info("Running adaptation loss = %f", adaptation_loss_fixed)
                writer.add_scalar('/learn/test/adaptation_loss', current_adaptation_loss, step)

        if (step + 1) % (LOG_INTERVAL * 500) == 0:
            if not args["no_save"]:
                torch.save(metalearner.net, my_experiment.path + "net.model")
            dict_names = {}
            for (name, param) in metalearner.net.named_parameters():
                dict_names[name] = param.adaptation

            my_experiment.add_result("Layers meta values", dict_names)
            my_experiment.add_result("Meta loss", loss_history)
            my_experiment.add_result("Adaptation loss", adaptation_loss_history)
            my_experiment.add_result("Running adaption loss", adaptation_running_loss_history)
            my_experiment.store_json()


if __name__ == '__main__':
    main()
