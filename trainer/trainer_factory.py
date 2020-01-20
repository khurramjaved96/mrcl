''' Incremental-Classifier Learning
 Authors : Khurram Javed, Muhammad Talha Paracha
 Maintainer : Khurram Javed
 Lab : TUKL-SEECS R&D Lab
 Email : 14besekjaved@seecs.edu.pk '''

import trainer.metatrainer as rf

class TrainerFactory():
    def __init__(self):
        pass

    @staticmethod
    def get_trainer(trainer, train_iterator, myModel, optimizer, args):

        if trainer == "meta":
            return rf.MetaTrainer(train_iterator, myModel, args.cuda, optimizer, args)
