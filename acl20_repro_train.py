from acl20_repro import PREMASKED_DATA, PRETRAINED_MODELS, PRETRAINED_MODEL_CONFIG_JSON, PAPER_TASK_TO_INTERNAL

# NOTE: https://chrisdonahue.com/gdrive-wget
_CMD_TEMPL = """
mkdir -p {train_tmp_dir}/data

# Download pre-masked training data
wget -nc --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id={train_data_id}' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\\1\\n/p')&id={train_data_id}" -O {train_tmp_dir}/data/{data_tag}_train.pkl && rm -rf /tmp/cookies.txt
# Download pre-masked validation data
wget -nc --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id={valid_data_id}' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\\1\\n/p')&id={valid_data_id}" -O {train_tmp_dir}/data/{data_tag}_valid.pkl && rm -rf /tmp/cookies.txt

python train_ilm.py \\
    train_{data_tag}_{model_type} \\
    {train_tmp_dir}/train_{data_tag}_{model_type} \\
    {train_tmp_dir}/data \\
    --seed 0 \\
    --mask_cls {mask_cls} \\
    --task {task} \\
    --data_loader_num_workers 4 \\
    --train_examples_tag {data_tag}_train \\
    --train_batch_size 8 \\
    --train_batch_accumulation 3 \\
    --train_sequence_length 256 \\
    --train_skip_naive_incomplete \\
    --eval_examples_tag {data_tag}_valid \\
    --eval_max_num_examples 512 \\
    --eval_batch_size 8 \\
    --eval_sequence_length 256 \\
    --eval_skip_naive_incomplete \\
    {extra_args}
"""

if __name__ == '__main__':
  import os
  import sys

  try:
    train_tmp_dir = os.environ['ILM_DIR']
  except:
    train_tmp_dir = '/tmp/ilm'
  train_tmp_dir = os.path.join(train_tmp_dir, 'train_repro')

  dataset, model_type = sys.argv[1:]

  data_tag = dataset[:3]
  train_data_url = PREMASKED_DATA['train']['{}_mixture'.format(data_tag)]
  valid_data_url = PREMASKED_DATA['valid']['{}_mixture'.format(data_tag)]

  if 'lyr' in dataset:
    mask_cls = 'ilm.mask.hierarchical.MaskHierarchicalVerse'
  else:
    mask_cls = 'ilm.mask.hierarchical.MaskHierarchical'

  task = PAPER_TASK_TO_INTERNAL[model_type.replace('scratch', '')]
  scratch = 'scratch' in model_type

  extra_args = ''
  if 'scratch' in model_type:
    extra_args += ' --train_from_scratch'
  if data_tag == 'abs':
    extra_args += ' --train_num_epochs 20'
  elif data_tag == 'lyr':
    extra_args += ' --train_num_epochs 2'

  print(_CMD_TEMPL.format(
    train_tmp_dir=train_tmp_dir,
    data_tag=data_tag,
    model_type=model_type,
    train_data_id=train_data_url.split('=')[-1],
    valid_data_id=valid_data_url.split('=')[-1],
    mask_cls=mask_cls,
    task=task,
    extra_args=extra_args
  ))
