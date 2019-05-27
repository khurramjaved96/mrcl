# mrcl
Under blind review at NeurIPS19 

## Learning Representations
To learn representations for omnigtot run the following command:
``` bash
python mrcl_classification.py --rln 6 --update_lr 0.03 --name mrcl_omniglot --update_step 20 --steps 40000
```

This will store the learned model at ../results/DDMonthYYYY/Omniglot/0.0001/mrcl_omniglot)

``` bash
python mrcl_regression.py --update-step 40 --meta_lr 0.0001 --update_lr 0.003 --tasks 10 --capacity 10 --width 300 --rln 6
```
This will store the learned model at ../results/DDMonthYYYY/Sin/0.0001/mrcl_regression)

We also provide trained mrcl models used to report results for the incremental sine and split omniglot benchmark in the paper in trained_models/incremental_sine.model and trained_models/split_omniglot.model.

## Continual Learning Evaluation

