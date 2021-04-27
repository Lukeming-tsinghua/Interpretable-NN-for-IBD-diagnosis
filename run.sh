# train CNN model
#python -u train.py\
#       	--batch_size 128\
#	--cuda 1

# finetune PTM
python -u finetune.py\
	--lr 5e-5\
	--save_config roberta\
       	--batch_size 16\
	--model_config hfl/chinese-roberta-wwm-ext\
	--class_num 2\
	--cuda 1

