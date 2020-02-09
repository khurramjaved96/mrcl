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
import matplotlib.pyplot as plt
logger = logging.getLogger('experiment')


def main():
    p = reg_parser.Parser()
    total_seeds = len(p.parse_known_args()[0].seed)
    rank = p.parse_known_args()[0].rank
    all_args = vars(p.parse_known_args()[0])
    print("All args = ", all_args)

    args = utils.get_run(vars(p.parse_known_args()[0]), rank)

    if args['model_path'] is not None:
        # args_old = args
        args_temp_temp, layers_learn = utils.load_run(args['model_path'])
        # args["no_meta"] = args_old["no_meta"]
        # args["model_path"] = args_old["model_path"]
        # args["name"] = args_old["name"]

    utils.set_seed(args["seed"])

    my_experiment = experiment(args["name"], args, "../results/", commit_changes=False,
                               rank=int(rank / total_seeds),
                               seed=total_seeds)
    my_experiment.results["all_args"] = all_args

    writer = SummaryWriter(my_experiment.path + "tensorboard")

    logger = logging.getLogger('experiment')

    print("Selected args", args)

    tasks = list(range(2000))
    # tasks = list(range(20))

    sampler = ts.SamplerFactory.get_sampler("Sin", tasks, None, capacity=args["capacity"] + 1)

    config = mf.ModelFactory.get_model(args["model"], "Sin", in_channels=args["capacity"] + 1, num_actions=1,
                                       width=args["width"])

    gpu_to_use = rank % args["gpus"]
    if torch.cuda.is_available():
        device = torch.device('cuda:' + str(gpu_to_use))
        logger.info("Using gpu : %s", 'cuda:' + str(gpu_to_use))
    else:
        device = torch.device('cpu')

    replay_buffer = utils.replay_buffer(30)
    metalearner = MetaLearnerRegression(args, config).to(device)

    if args['model_path'] is not None:
        metalearner.net = torch.load(args['model_path'] + "/net.model",
                                     map_location="cpu").to(device)

        for (name, param) in metalearner.net.named_parameters():
            if name in layers_learn:
                param.learn = layers_learn[name]
                print(name, layers_learn[name])

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
    LOG_INTERVAL = 1
    for name, param in metalearner.named_parameters():
        logger.info("Name = %s, learn = %s", name, str(param.learn))
    #
    # logger.warning("Using SIGMOID Context plasticity")
    # logger.warning("Resseting model parameters")
    # metalearner.net.reset_vars()
    for step in range(args["epoch"]):
        # logger.warning("ONLY 20 FUNCTIONS")
        if args["sanity"]:
            logger.warning("Reloading model")
            metalearner.load_model(args, config)
            metalearner.net = metalearner.net.to(device)
        if step % LOG_INTERVAL == 0:
            logger.warning("####\t STEP %d \t####", step)

        adaptation_lr = args["update_lr"]

        t1 = np.random.choice(tasks, args["tasks"], replace=False)

        iterators = []
        for t in t1:
            iterators.append(sampler.sample_task([t]))

        x_traj, y_traj, x_rand, y_rand = utils.construct_set(iterators, sampler, steps=args["update_step"], shuffle=False)

        x_traj_cpu, y_traj_cpu, x_rand_cpu, y_rand_cpu = x_traj, y_traj, x_rand, y_rand
        if torch.cuda.is_available():
            x_traj, y_traj, x_rand, y_rand = x_traj.to(device), y_traj.to(device), x_rand.to(device), y_rand.to(device)
        #
        net = metalearner.net
        a_old = None
        for meta_counter, k in enumerate(range(len(x_traj))):
            logits = net(x_traj[k], vars=None, bn_training=False, log=bool(not (step + meta_counter)))
            logits_select = []
            for no, val in enumerate(y_traj[k, :, 1].long()):
                logits_select.append(logits[no, val])
            logits = torch.stack(logits_select).unsqueeze(1)
            loss_temp = F.mse_loss(logits, y_traj[k, :, 0].unsqueeze(1))
            # if k < 10:

            grad = metalearner.clip_grad(
                torch.autograd.grad(loss_temp, list(filter(lambda x: x.learn, list(net.parameters())))))

            list_of_context = None
            if metalearner.context:
                list_of_context = metalearner.net.forward_plasticity(x_traj[k], log=bool(not (step + meta_counter)))


            a = list_of_context[8].cpu().detach().numpy().flatten()
            # print(x_traj[k][0])
            if a_old is not None:
                print("Prod = ", np.sum(a*a_old))
                print(np.corrcoef(a, a_old))
            else:
                a_old = a
            # histo, b = np.histogram(a)
            # #
            # # print(histo)
            # # print(b)t
            plt.hist(a)
            plt.ylim(0,100000)
            plt.savefig(my_experiment.path + str(meta_counter)+'.png', format="png")
            plt.clf()
            plt.close()
            a_img = a.reshape(400,400)
            # a_img = (a_img > 0).astype(np.int)
            plt.imshow(a_img)
            plt.savefig(my_experiment.path + str(meta_counter)+"box.png", format="png")
            plt.clf()
            plt.close()
            a_positive = np.sum((a>0).astype(np.int))/(400*400)
            print("Total positive = ", a_positive * 100)


                # print(a.shape)
        assert(False)
                # metalearner.inner_update(net, grad, adaptation_lr, list_of_context, log=bool(not (step + meta_counter)))





#
#
if __name__ == '__main__':
    main()
#
#
