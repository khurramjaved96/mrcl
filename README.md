# (05 July, 2020) Major bug fix and refactoring log:

1. Fixed a bug that resulted in incorrect meta-gradients.
2. Refactored the code. It should be easier to understand and modify now. 
3. Significantly improved results on both omniglot and sine benchmark by fixing the bug. By using a linear PLN layer -- as suggested by S. Beaulieu et.al (2020) -- it is possible to get the same results as ANML (S. Beaulieu 2020) without using any neuromodulation layers. 
4. The bug fix also makes the optimization more robust to hyper-parameter changes. The omniglot results hold for a wide range of meta-learning and inner learning rates.
5. Added new pretrained models in the google drive. Check mrcl_trained_models/Omniglot_updated. There are eight pre-trained models, with different hyper-parameters. You can look at hyper-parameters in the metadata.json file. The old model will no longer work withe new code. If you want to use the old models, checkout an older commit of the repo. 

A discussion on the changes: https://github.com/khurramjaved96/mrcl/issues/15

Reference: Beaulieu, Shawn, et al. "Learning to continually learn." ECAI (2020).
# OML (Online-aware Meta-learning) ~ NeurIPS19

Paper : https://arxiv.org/abs/1905.12588

<div>
<img src="utils/overview_1.png" alt="Overall system architecture for learning representations" width="100% align="middle">
                                                                                                  </div>                                                                                        


### Learning OML Representations
To learn representations for omnigtot run the following command:
``` bash
python oml_omniglot.py --update_lr 0.03 --meta_lr 1e-4 --name OML_Omniglot/ --tasks 3 --update_step 5 --steps 700000 --rank 0
```

This will store the learned model at ../results/DDMonthYYYY/Omniglot/0.0001/oml_omniglot)


### Evaluating Representations learned by OML
We provide trained models at https://drive.google.com/drive/folders/1vHHT5kxtgx8D4JHYg25iA-C31O5OjAQz?usp=sharing which can be used to evaluate performance on the continual learning benchmarks. 

To evaluate performance on test trajectories of omniglot run: 
``` bash
python evaluate_omniglot.py --model-path path_to_model/learner.model --name Omniglot_evaluation/  --schedule 10:50:100:200:600
```

Exclude the --test argument to get result on training trajectories (Used to measure forgetting). 

Results will be stored in a json file in "../results/DDMonthYYYY/Omniglot/eval/Omni_test_traj_0"

### Visualizing Representations

To visualize representations for different omniglot models, run 

``` bash
python visualize_representations.py --name OML_rep_study --model ./trained_models/split_omniglot_oml.model
```

### Results

#### Classification Results
The accuracy curve averaged over 50 runs as we learn more classes sequentially. The error bars represent 95% confidence intervals drawn using 1,000 bootstraps. We report results on both the training trajectory (left) and a held out dataset that has the same classes as the training trajectory (right).
![alt text](utils/classification_1.png "Method Overview")
 Online updates starting from OML are capable of learning 200 classes with little to no forgetting. Other representations, such as pretraining and SR-NN suffer from noticeable forgetting on the other hand. OML also generalizes better than the other methods on the unseen held out set. Note that the Oracle, learned using multiple, IID passes over the trajectory, represents an upper bound on the performance, reflecting the inherent inaccuracy when training on an increasing number of classes. 
#### Regression Results
Mean squared error across all 10 regression tasks. The x-axis in (a) corresponds to seeing all data points of samples for class 1, then class 2 and so on. These learning curves are averaged over 50 runs, with error bars representing 95% confidence interval drawn by 1,000 bootstraps.
![alt text](utils/regression_1.png "Method Overview")
We can see that the representation trained on iid data---pretraining---is not effective for online updating. Notice that in the final prediction accuracy in (b), pretraining and SR-NN representations have accurate predictions for task 10, but high error for earlier tasks. OML, on the other hand, has a slight skew in error towards later tasks in learning but is largely robust.

### References
1. Meta-learning code has been taken and modified from : https://github.com/dragen1860/MAML-Pytorch
2. For EWC, MER, and ER-Reservoir experiments, we modify the following implementation to be able to load our models : https://github.com/mattriemer/MER
