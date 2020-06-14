from acl20_repro import PREMASKED_DATA, PRETRAINED_MODELS, PRETRAINED_MODEL_CONFIG_JSON, PAPER_TASK_TO_INTERNAL

# NOTE: https://chrisdonahue.com/gdrive-wget
_CMD_TEMPL = """
mkdir -p {eval_tmp_dir}/data
mkdir -p {eval_tmp_dir}/models/{model_tag}

# Download pre-masked data
wget -nc --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id={data_id}' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\\1\\n/p')&id={data_id}" -O {eval_tmp_dir}/data/{data_tag}_test.pkl && rm -rf /tmp/cookies.txt

# Download pre-trained model
wget -nc --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id={model_id}' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\\1\\n/p')&id={model_id}" -O {eval_tmp_dir}/models/{model_tag}/pytorch_model.bin && rm -rf /tmp/cookies.txt
wget -nc --no-check-certificate 'https://docs.google.com/uc?export=download&id=15JnXi7L6LeEB2fq4dFK2WRvDKyX46hVi' -O {eval_tmp_dir}/models/{model_tag}/config.json

# NOTE: train_ilm.py won't load weights unless it sees this file
touch {eval_tmp_dir}/models/{model_tag}/step.pkl

python train_ilm.py \\
    eval \\
    {eval_tmp_dir}/models/{model_tag} \\
    {eval_tmp_dir}/data \\
    --mask_cls {mask_cls} \\
    --task {task} \\
    --data_no_cache \\
    --eval_only \\
    --eval_examples_tag {data_tag}_test \\
    --eval_batch_size 4 \\
    --eval_sequence_length 256 \\
    --eval_skip_naive_incomplete
"""

if __name__ == '__main__':
  import os
  import sys

  try:
    eval_tmp_dir = os.environ['ILM_DIR']
  except:
    eval_tmp_dir = '/tmp/ilm'
  eval_tmp_dir = os.path.join(eval_tmp_dir, 'eval_repro')

  dataset, model_type, infill_type = sys.argv[1:]

  data_tag = '{}_{}'.format(dataset[:3], infill_type)
  model_tag = '{}_{}'.format(dataset[:3], model_type)

  mask_url = PREMASKED_DATA['test'][data_tag]
  model_url = PRETRAINED_MODELS[model_tag]

  if 'lyr' in model_tag:
    mask_cls = 'ilm.mask.hierarchical.MaskHierarchicalVerse'
  else:
    mask_cls = 'ilm.mask.hierarchical.MaskHierarchical'

  task = PAPER_TASK_TO_INTERNAL[model_tag.split('_')[-1].replace('scratch', '')]

  print(_CMD_TEMPL.format(
    eval_tmp_dir=eval_tmp_dir,
    data_tag=data_tag,
    model_tag=model_tag,
    data_id=mask_url.split('=')[-1],
    model_id=model_url.split('=')[-1],
    mask_cls=mask_cls,
    task=task
  ))
