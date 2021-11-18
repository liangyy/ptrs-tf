# ARGS1: GPU ID
# ARGS2: OUTPREFIX
# ARGS3: model YAML
# ARGS4: phase YAML
# ARGS5: model type
# ARGS6: training set
# ARGS7: residual mode (if yes, say 'residual' here)

OUTDIR=/vol/bmd/yanyul/UKB/ptrs-tf/models/cnnPTRS
OUTPREFIX=/vol/bmd/yanyul/UKB/ptrs-tf/models/cnnPTRS/$2
MODELYAML=$3
PHASEYAML=$4
MODELTYPE=$5
TRAINSET=$6

if [[ ! -z $7 && $7=='residual' ]]
then 
  residualmode="--residual-mode"
  OUTPREFIX=$OUTPREFIX"-residual-mode"
else
  residualmode=""
fi

mkdir -p $OUTDIR


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
  --training-set $TRAINSET \
  --data-scheme-yaml /vol/bmd/yanyul/GitHub/ptrs-tf/misc_files/data_scheme.yaml \
  --model-type $MODELTYPE \
  --batch-size 512 \
  --valid-and-test-size 4096 \
  --output-prefix $OUTPREFIX \
  --phase-yaml $PHASEYAML \
  --model-yaml $MODELYAML \
  $residualmode \
  > $OUTPREFIX.log 2>&1
  
  # --model-yaml $MODELYAML \
  # --num-epoch 20 \
  
