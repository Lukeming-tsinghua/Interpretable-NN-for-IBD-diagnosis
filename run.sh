# train CNN model
#python train.py

# finetune PTM
python finetune.py\
	--lr 5e-5\
	--save_config roberta\
       	--batch_size 16\
	--model_config hfl/chinese-roberta-wwm-ext\
	--class_num 2\
	--cuda 0

