# Preparing HDF5 files for training PTRS.

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

# Updated splitting (only affect testing but not training)
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

# Another updated splitting (make the training cohort of PTRS match PRS exactly)
# GTEx Whole_Blood with matched British splitting 
myhdf5=/lambda_stor/data/yanyul/washington_UKB/predicted_expression/predicted_expression.ukb_imp_x_ctimp_Whole_Blood.h5
myyaml=/lambda_stor/data/yanyul/GitHub/ptrs-tf/misc_files/split_ptrs_ukb_split_british.yaml
prefixout=ukb_imp_x_ctimp_Whole_Blood
screen -dmS split-$prefixout bash run_split_pred_expr_split_british.screen $myhdf5 $myyaml $prefixout
```

# Training

All scripts with names `run_elastic_net_*.screen`. The file name indicates the type of model and the predicted expression being used.
If there is `_pt` in the file name, it means that clumping and thresholding models are trained instead of elastic nets. 

# Evaluating the performance

All scripts with names `pred_*.screen`. 


