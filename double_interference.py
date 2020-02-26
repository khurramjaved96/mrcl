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

    tasks = list(range(2000))

    sampler = ts.SamplerFactory.get_sampler("Sin", tasks, None, capacity=args["capacity"] + 1)

    config = mf.ModelFactory.get_model(args["model"], "Sin", input_dimension=args["capacity"] + 1, output_dimension=1,
                                       width=args["width"])

    gpu_to_use = rank % args["gpus"]
    if torch.cuda.is_available():
        device = torch.device('cuda:' + str(gpu_to_use))
        logger.info("Using gpu : %s", 'cuda:' + str(gpu_to_use))
    else:
        device = torch.device('cpu')

    replay_buffer = utils.replay_buffer(100)
    metalearner = MetaLearnerRegression(args, config).to(device)

    tmp = filter(lambda x: x.requires_grad, metalearner.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    logger.info('Total trainable tensors: %d', num)

    loss = 0
    adaptation_loss = 0
    loss_history = []
    adaptation_loss_history = []
    adaptation_running_loss_history = []

    adaptation_loss_after = 0
    adaptation_loss_history_after = []
    adaptation_running_loss_history_after = []

    for step in range(args["epoch"]):
        if step % 5 == 0:
            logger.warning("####\t STEP %d \t####", step)

        lrs = args["update_lr"]
        if step == 0:
            for name, param in metalearner.named_parameters():
                logger.info("Name = %s, learn = %s", name, str(param.learn))

        t1 = np.random.choice(tasks, args["tasks"], replace=False)

        iterators = []
        for t in t1:
            iterators.append(sampler.sample_task([t]))

        x_traj, y_traj, x_rand, y_rand = utils.construct_set(iterators, sampler, steps=args["update_step"])


        replay_buffer.add([x_traj, y_traj, x_rand, y_rand])

        if torch.cuda.is_available():
            x_traj, y_traj, x_rand, y_rand = x_traj.to(device), y_traj.to(device), x_rand.to(device), y_rand.to(device)
        x_rand_current = x_rand
        y_rand_current = y_rand
        net = metalearner.net
        for k in range(len(x_traj)):

            logits = net(x_traj[k], vars=None, bn_training=False)
            logits_select = []
            for no, val in enumerate(y_traj[k, :, 1].long()):
                logits_select.append(logits[no, val])
            logits = torch.stack(logits_select).unsqueeze(1)
            loss_temp = F.mse_loss(logits, y_traj[k, :, 0].unsqueeze(1))
            # if k < 10:

            grad = metalearner.clip_grad(
                torch.autograd.grad(loss_temp, list(filter(lambda x: x.learn, list(net.parameters())))))

            counter = 0
            for (name, p) in net.named_parameters():
                if "meta" in name or "neuro" in name:
                    pass
                if p.learn:
                    g = grad[counter]
                    # print(g)
                    mask = net.meta_plasticity[counter]
                    # print(g.shape, p.shape, mask.shape)
                    if metalearner.plasticity:
                        if metalearner.sigmoid:
                            p.data -= lrs * g * torch.sigmoid(mask)
                        else:
                            p.data -= lrs * g * mask
                    else:
                        p.data -= lrs * g
                    counter += 1
                else:
                    pass

        with torch.no_grad():
            logits = net(x_rand[0], vars=None, bn_training=False)
            # print("Logits = ", logits)
            logits_select = []
            for no, val in enumerate(y_rand[0, :, 1].long()):
                logits_select.append(logits[no, val])
            logits = torch.stack(logits_select).unsqueeze(1)
            # print("Logits = ", logits)
            # print("Targets = ", y_rand[0, :, 0].unsqueeze(1))
            current_adaptation_loss = F.mse_loss(logits, y_rand[0, :, 0].unsqueeze(1))
            adaptation_loss_history.append(current_adaptation_loss.detach().item())
            adaptation_loss = adaptation_loss * 0.97 + current_adaptation_loss.detach().cpu().item() * 0.03
            adaptation_running_loss_history.append(adaptation_loss)
            # adaptation_loss /= (1-(0.85**(step+1)))
            if step % 5 == 0:
                logger.info("Running adaptation loss = %f", adaptation_loss)
            # logger.info("Adaptation loss = %f", loss_q.item())
            writer.add_scalar('/learn/train/accuracy', current_adaptation_loss, step)
            # lr_results[lrs].append(loss_q.item())

        if not args["no_meta"]:
            for meta_sleep in range(args["frequency"]):
                t1 = np.random.choice(tasks, args["tasks"], replace=False)

                iterators = []
                for t in t1:
                    iterators.append(sampler.sample_task([t]))

                # x_traj, y_traj, x_rand, y_rand = utils.construct_set(iterators, sampler, steps=args["update_step"])

                x_traj, y_traj, x_rand, y_rand = replay_buffer.sample(1)[0]

                if torch.cuda.is_available():
                    x_traj, y_traj, x_rand, y_rand = x_traj.to(device), y_traj.to(device), x_rand.to(device), y_rand.to(
                        device)

                meta_loss = metalearner(x_traj, y_traj, x_rand, y_rand)
                loss_history.append(meta_loss[-1].detach().cpu().item())
                if metalearner.meta_optim is not None:
                    metalearner.meta_optim.optimizer_step()
                if not args["no_plasticity"]:
                    metalearner.meta_optim_plastic.optimizer_step()
                # if not args["no_neuro"]:
                #     metalearner.meta
                #
                loss = loss * 0.97 + 0.03 * meta_loss[-1]
                # loss /= (1-(0.85**(step+1)))
                writer.add_scalar('/metatrain/train/accuracy', meta_loss[-1], step)
                writer.add_scalar('/metatrain/train/runningaccuracy', loss, step)

                if step % 5 == 0:
                    logger.info("Running meta-loss = %f", loss.item())
                if step % 20 == 0:
                    logger.debug('Meta-training loss: Before adaptation: %f \t After adaptation: %f', meta_loss[0].item(),
                                 meta_loss[-1].item())
        with torch.no_grad():
            logits = net(x_rand_current[0], vars=None, bn_training=False)
            # print("Logits = ", logits)
            logits_select = []
            for no, val in enumerate(y_rand_current[0, :, 1].long()):
                logits_select.append(logits[no, val])
            logits = torch.stack(logits_select).unsqueeze(1)
            # print("Logits = ", logits)
            # print("Targets = ", y_rand[0, :, 0].unsqueeze(1))
            current_adaptation_loss_after = F.mse_loss(logits, y_rand_current[0, :, 0].unsqueeze(1))
            adaptation_loss_history_after.append(current_adaptation_loss_after.detach().item())
            adaptation_loss_after = adaptation_loss_after * 0.97 + current_adaptation_loss_after.detach().cpu().item() * 0.03
            adaptation_running_loss_history_after.append(adaptation_loss_after)
            # adaptation_loss /= (1-(0.85**(step+1)))
            if step % 5 == 0:
                logger.info("Running adaptation after loss = %f", adaptation_loss_after)
            # logger.info("Adaptation loss = %f", loss_q.item())
            writer.add_scalar('/learn/train/accuracyafter', current_adaptation_loss_after, step)
            # lr_results[lrs].append(loss_q.item())


        if step % 100 == 0:
            torch.save(metalearner.net, my_experiment.path + "net.model")
            dict_names = {}
            for (name, param) in metalearner.net.named_parameters():
                dict_names[name] = param.learn
            my_experiment.add_result("Layers meta values", dict_names)
            my_experiment.add_result("Meta loss", loss_history)
            my_experiment.add_result("Adaptation loss", adaptation_loss_history)
            my_experiment.add_result("Running adaption loss", adaptation_running_loss_history)
            my_experiment.store_json()


#
if __name__ == '__main__':
    main()
#
#
