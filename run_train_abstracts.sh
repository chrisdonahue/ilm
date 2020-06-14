export CUDA_VISIBLE_DEVICES="0"
DATASET=abstracts
TRAIN_DIR=./train/${DATASET}
rm -rf ${TRAIN_DIR}
python train_ilm.py \
        abstracts \
	${TRAIN_DIR} \
        --data_num_threads 8 \
	--data_style abstract \
	--eval_batch_size 21 \
	--eval_data_fp ./data/abstracts/valid.txt \
	--eval_sequence_length 128 \
	--mask_document_p 0.075 \
	--mask_firstword_p 0. \
	--mask_lastword_p 0. \
	--mask_leadingwords_p 0. \
	--mask_ngram_p 0.075 \
	--mask_paragraph_p 0.075 \
	--mask_sentence_p 0.15 \
	--mask_word_p 0.075 \
	--seed 0 \
	--train_batch_size 21 \
	--train_data_fp ./data/abstracts/train.txt \
	--train_eval_secs 360 \
	--train_from_scratch  \
	--train_lm  \
	--train_num_tasks_per_example 4 \
	--train_sequence_length 128 \
	--train_summary_secs 180
