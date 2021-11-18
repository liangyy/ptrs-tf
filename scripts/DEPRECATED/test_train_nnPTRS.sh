# ARGS1: GPU ID

OUTPREFIX=/vol/bmd/yanyul/UKB/tmp/test_train_nnPTRS

# computing environment setup
source /vol/bmd/yanyul/miniconda3/etc/profile.d/conda.sh
conda activate tensorflow
export CUDA_VISIBLE_DEVICES=$1
# source /home/yanyul/tensorflow_env.sh

# pre-specifics
WORKDIR=/vol/bmd/yanyul/GitHub/ptrs-tf/

# code chunk
cd $WORKDIR

python code/train_nnPTRS.py \
  --training-set /vol/bmd/yanyul/UKB/predicted_expression_tf2/ukb_imp_x_ctimp_Whole_Blood_British.hdf5 \
  --data-scheme-yaml /vol/bmd/yanyul/GitHub/ptrs-tf/misc_files/data_scheme.yaml \
  --model-type MLP \
  --batch-size 256 \
  --valid-and-test-size 512 \
  --output-prefix $OUTPREFIX \
  --phase-yaml /vol/bmd/yanyul/GitHub/ptrs-tf/misc_files/test_phase.yaml \
  > $OUTPREFIX.log 2>&1
  
  # --model-yaml /vol/bmd/yanyul/GitHub/ptrs-tf/misc_files/mlp_ptrs.yaml \
  # --num-epoch 20 \
  
