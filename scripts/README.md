Misc of note for scripts in this directory.

* `run_split_pred_expr.screen`

```
myhdf5=/vol/bmd/yanyul/UKB/predicted_expression/predicted_expression.ukb_imp_x_ctimp_Whole_Blood.h5
myyaml=/vol/bmd/yanyul/GitHub/ptrs-tf/misc_files/split_ptrs_ukb.yaml
prefixout=ukb_imp_x_ctimp_Whole_Blood
screen -dmS split-$prefixout bash run_split_pred_expr.screen $myhdf5 $myyaml $prefixout
```

```
# t2d traits
myhdf5=/vol/bmd/yanyul/UKB/predicted_expression/predicted_expression.ukb_imp_x_ctimp_Whole_Blood.h5
myyaml=/vol/bmd/yanyul/GitHub/ptrs-tf/misc_files/split_ptrs_ukb_t2d.yaml
prefixout=ukb_imp_x_ctimp_Whole_Blood_t2d
screen -dmS split-$prefixout bash run_split_pred_expr.screen $myhdf5 $myyaml $prefixout

# mesa cau
myhdf5=/vol/bmd/yanyul/UKB/predicted_expression/predicted_expression.ukb_imp_x_CAU.h5
myyaml=/vol/bmd/yanyul/GitHub/ptrs-tf/misc_files/split_ptrs_ukb_t2d.yaml
prefixout=ukb_imp_x_MESA_CAU_t2d
screen -dmS split-$prefixout bash run_split_pred_expr.screen $myhdf5 $myyaml $prefixout

# mesa afhi
myhdf5=/vol/bmd/yanyul/UKB/predicted_expression/predicted_expression.ukb_imp_x_AFHI.h5
myyaml=/vol/bmd/yanyul/GitHub/ptrs-tf/misc_files/split_ptrs_ukb_t2d.yaml
prefixout=ukb_imp_x_MESA_AFHI_t2d
screen -dmS split-$prefixout bash run_split_pred_expr.screen $myhdf5 $myyaml $prefixout
```

* `run_least_squared.screen` 

```
myhdf5=/vol/bmd/yanyul/UKB/predicted_expression_tf2/ukb_imp_x_ctimp_Whole_Blood_Chinese.hdf5
myyaml=/vol/bmd/yanyul/GitHub/ptrs-tf/misc_files/data_scheme.yaml
outprefix=least_square_model-ukb_imp_x_ctimp_Whole_Blood_Chinese
batchsize=8096
screen -dmS least_square bash run_least_squared.screen $myhdf5 $myyaml $outprefix $batchsize 
```

Or by population

```
pop=ukb_imp_x_ctimp_Whole_Blood_Chinese
myhdf5=/vol/bmd/yanyul/UKB/predicted_expression_tf2/$pop.hdf5
myyaml=/vol/bmd/yanyul/GitHub/ptrs-tf/misc_files/data_scheme.yaml
outprefix=least_square_model-$pop
batchsize=8096
screen -dmS least_square bash run_least_squared.screen $myhdf5 $myyaml $outprefix $batchsize 
```

Running nnPTRS

```
screen -dmS baseline bash submit_train_baselinePTRS_ctimp_whole_blood.screen 1 > submit_train_baselinePTRS_ctimp_whole_blood.log 2>&1
screen -dmS cnn bash submit_train_cnnPTRS_ctimp_whole_blood.screen 2 > submit_train_cnnPTRS_ctimp_whole_blood.log 2>&1
screen -dmS mlp bash submit_train_mlpPTRS_ctimp_whole_blood.screen 3 > submit_train_mlpPTRS_ctimp_whole_blood.log 2>&1
```

Running nnPTRS in residual mode
 
```
screen -dmS baseline bash submit_train_baselinePTRS_ctimp_whole_blood.screen 1 residual > submit_train_baselinePTRS_ctimp_whole_blood.log 2>&1
screen -dmS cnn bash submit_train_cnnPTRS_ctimp_whole_blood.screen 2 residual > submit_train_cnnPTRS_ctimp_whole_blood.log 2>&1
screen -dmS mlp bash submit_train_mlpPTRS_ctimp_whole_blood.screen 3 residual > submit_train_mlpPTRS_ctimp_whole_blood.log 2>&1
```

Running nnPTRS in residual mode with suffix
 
```
screen -dmS baseline bash submit_train_baselinePTRS_ctimp_whole_blood.screen 1 residual .with_adam > submit_train_baselinePTRS_ctimp_whole_blood.log 2>&1
screen -dmS cnn bash submit_train_cnnPTRS_ctimp_whole_blood.screen 2 residual .with_adam > submit_train_cnnPTRS_ctimp_whole_blood.log 2>&1
screen -dmS mlp bash submit_train_mlpPTRS_ctimp_whole_blood.screen 3 residual .with_adam > submit_train_mlpPTRS_ctimp_whole_blood.log 2>&1
```

Running nnPTRS in residual mode with suffix
 
```
screen -dmS baseline bash submit_train_baselinePTRS_ctimp_whole_blood.screen 1 residual .with_adam_and_universal_normalizer 
screen -dmS cnn bash submit_train_cnnPTRS_ctimp_whole_blood.screen 2 residual .with_adam_and_universal_normalizer 
screen -dmS mlp bash submit_train_mlpPTRS_ctimp_whole_blood.screen 3 residual .with_adam_and_universal_normalizer 
```


