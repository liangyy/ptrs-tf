Misc of note for scripts in this directory.

* `run_split_pred_expr.screen`

```
myhdf5=/vol/bmd/yanyul/UKB/predicted_expression/predicted_expression.ukb_imp_x_ctimp_Whole_Blood.h5
myyaml=/vol/bmd/yanyul/GitHub/ptrs-tf/misc_files/split_ptrs_ukb.yaml
prefixout=ukb_imp_x_ctimp_Whole_Blood
screen -dmS split bash run_split_pred_expr.screen $myhdf5 $myyaml $prefixout
```

