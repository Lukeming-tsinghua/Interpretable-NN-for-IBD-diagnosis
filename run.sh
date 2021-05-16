# train CNN model
#python -u train.py\
#	--lr 1e-3\
#       	--batch_size 128\
#	--cuda 1

# finetune PTM
#python -u finetune.py\
#	--lr 5e-5\
#	--save_config roberta\
#       	--batch_size 16\
#	--model_config hfl/chinese-roberta-wwm-ext\
#	--class_num 2\
#	--cuda 1

# distill CNN model
python -u distill.py \
	--lr 1e-3\
	--save_config CNN-distill-\
       	--batch_size 128\
	--epoch_num 30\
	--cuda 1
