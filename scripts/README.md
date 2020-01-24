Misc of note for scripts in this directory.

* `run_split_pred_expr.screen`

```
myhdf5=/vol/bmd/yanyul/UKB/predicted_expression/predicted_expression.ukb_imp_x_ctimp_Whole_Blood.h5
myyaml=/vol/bmd/yanyul/GitHub/ptrs-tf/misc_files/split_ptrs_ukb.yaml
prefixout=ukb_imp_x_ctimp_Whole_Blood
screen -dmS split bash run_split_pred_expr.screen $myhdf5 $myyaml $prefixout
```

* `run_least_squared.screen` 

```
myhdf5=/vol/bmd/yanyul/UKB/predicted_expression_tf2/ukb_imp_x_ctimp_Whole_Blood_Chinese.hdf5
myyaml=/vol/bmd/yanyul/GitHub/ptrs-tf/misc_files/data_scheme.yaml
outprefix=least_square_model-ukb_imp_x_ctimp_Whole_Blood_Chinese
batchsize=8096
screen -dmS least_square bash run_least_squared.screen $myhdf5 $myyaml $outprefix $batchsize 
```
