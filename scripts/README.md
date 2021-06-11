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

About `run_split_pred_expr_new.screen`. Here we make improvement on having nest list on populations so that we could pre-specify the validation, test, and training set (in order and by chunk).
This script itself is just the same as `run_split_pred_expr.screen` but the output is at `/vol/bmd/yanyul/UKB/predicted_expression_tf2_new` to avoid overwrite.

```
# GTEx Whole_Blood
myhdf5=/vol/bmd/yanyul/UKB/predicted_expression/predicted_expression.ukb_imp_x_ctimp_Whole_Blood.h5
myyaml=/vol/bmd/yanyul/GitHub/ptrs-tf/misc_files/split_ptrs_ukb_new.yaml
prefixout=ukb_imp_x_ctimp_Whole_Blood
screen -dmS split-$prefixout bash run_split_pred_expr_new.screen $myhdf5 $myyaml $prefixout

# MESA CAU
myhdf5=/vol/bmd/yanyul/UKB/predicted_expression/predicted_expression.ukb_imp_x_CAU.h5
myyaml=/vol/bmd/yanyul/GitHub/ptrs-tf/misc_files/split_ptrs_ukb_new.yaml
prefixout=ukb_imp_x_MESA_CAU
screen -dmS split-$prefixout bash run_split_pred_expr_new.screen $myhdf5 $myyaml $prefixout

# MESA AFHI
myhdf5=/vol/bmd/yanyul/UKB/predicted_expression/predicted_expression.ukb_imp_x_AFHI.h5
myyaml=/vol/bmd/yanyul/GitHub/ptrs-tf/misc_files/split_ptrs_ukb_new.yaml
prefixout=ukb_imp_x_MESA_AFHI
screen -dmS split-$prefixout bash run_split_pred_expr_new.screen $myhdf5 $myyaml $prefixout

```

New update: we updated the individual list for African and Indian. So, we re-extract them. 
For MESA ALL, we extract everything since we don't have anything before.
Save at `/vol/bmd/yanyul/UKB/predicted_expression_tf2_new_updated_indivs`

```
# GTEx Whole_Blood
myhdf5=/vol/bmd/yanyul/UKB/predicted_expression/predicted_expression.ukb_imp_x_ctimp_Whole_Blood.h5
myyaml=/vol/bmd/yanyul/GitHub/ptrs-tf/misc_files/split_ptrs_ukb_new_updated_indivs.yaml
prefixout=ukb_imp_x_ctimp_Whole_Blood
screen -dmS split-$prefixout bash run_split_pred_expr_new_updated_indivs.screen $myhdf5 $myyaml $prefixout

# MESA CAU
myhdf5=/vol/bmd/yanyul/UKB/predicted_expression/predicted_expression.ukb_imp_x_CAU.h5
myyaml=/vol/bmd/yanyul/GitHub/ptrs-tf/misc_files/split_ptrs_ukb_new_updated_indivs.yaml
prefixout=ukb_imp_x_MESA_CAU
screen -dmS split-$prefixout bash run_split_pred_expr_new_updated_indivs.screen $myhdf5 $myyaml $prefixout

# MESA AFHI
myhdf5=/vol/bmd/yanyul/UKB/predicted_expression/predicted_expression.ukb_imp_x_AFHI.h5
myyaml=/vol/bmd/yanyul/GitHub/ptrs-tf/misc_files/split_ptrs_ukb_new_updated_indivs.yaml
prefixout=ukb_imp_x_MESA_AFHI
screen -dmS split-$prefixout bash run_split_pred_expr_new_updated_indivs.screen $myhdf5 $myyaml $prefixout

# MESA ALL
myhdf5=/vol/bmd/yanyul/UKB/predicted_expression/predicted_expression.ukb_imp_x_ALL.h5
myyaml=/vol/bmd/yanyul/GitHub/ptrs-tf/misc_files/split_ptrs_ukb_new_updated_indivs_all.yaml
prefixout=ukb_imp_x_MESA_ALL
screen -dmS split-$prefixout bash run_split_pred_expr_new_updated_indivs.screen $myhdf5 $myyaml $prefixout
```

New updates: we split African and Caribbean. So, need to extract Carribean.

```
# GTEx Whole_Blood
myhdf5=/vol/bmd/yanyul/UKB/predicted_expression/predicted_expression.ukb_imp_x_ctimp_Whole_Blood.h5
myyaml=/vol/bmd/yanyul/GitHub/ptrs-tf/misc_files/split_ptrs_ukb_new_updated2_indivs.yaml
prefixout=ukb_imp_x_ctimp_Whole_Blood
screen -dmS split-$prefixout bash run_split_pred_expr_new_updated2_indivs.screen $myhdf5 $myyaml $prefixout

# MESA CAU
myhdf5=/vol/bmd/yanyul/UKB/predicted_expression/predicted_expression.ukb_imp_x_CAU.h5
myyaml=/vol/bmd/yanyul/GitHub/ptrs-tf/misc_files/split_ptrs_ukb_new_updated2_indivs.yaml
prefixout=ukb_imp_x_MESA_CAU
screen -dmS split-$prefixout bash run_split_pred_expr_new_updated2_indivs.screen $myhdf5 $myyaml $prefixout

# MESA AFHI
myhdf5=/vol/bmd/yanyul/UKB/predicted_expression/predicted_expression.ukb_imp_x_AFHI.h5
myyaml=/vol/bmd/yanyul/GitHub/ptrs-tf/misc_files/split_ptrs_ukb_new_updated2_indivs.yaml
prefixout=ukb_imp_x_MESA_AFHI
screen -dmS split-$prefixout bash run_split_pred_expr_new_updated2_indivs.screen $myhdf5 $myyaml $prefixout

# MESA ALL
myhdf5=/vol/bmd/yanyul/UKB/predicted_expression/predicted_expression.ukb_imp_x_ALL.h5
myyaml=/vol/bmd/yanyul/GitHub/ptrs-tf/misc_files/split_ptrs_ukb_new_updated2_indivs.yaml
prefixout=ukb_imp_x_MESA_ALL
screen -dmS split-$prefixout bash run_split_pred_expr_new_updated2_indivs.screen $myhdf5 $myyaml $prefixout
```