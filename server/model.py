import re
import torch
import torchtext
import torchtext.data
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from utils.model import CNN
from utils.data import DataIterator
from utils.utils import clean_text
from captum.attr import LayerIntegratedGradients, TokenReferenceBase
from captum.attr import *
from captum.attr import configure_interpretable_embedding_layer, remove_interpretable_embedding_layer

from LAC import LAC

from transformers import BertTokenizer
from transformers import BertConfig
from transformers import BertPreTrainedModel
from transformers import BertModel
from transformers import BertForSequenceClassification

class CNNPredictionModel:
    def __init__(self, model_path="static/CNN29",seq_length=256):
        self.seq_length = seq_length
        self.initialize(model_path)

    def initialize(self, model_path):
        print("initial tokenizer...")
        self.tokenizer = DataIterator().tokenizer
        self.PAD_IND = self.tokenizer.vocab.stoi['<pad>']
        self.token_reference = TokenReferenceBase(reference_token_idx=self.PAD_IND)
        print("initial inference model...")
        self.model = torch.load(model_path).cpu().eval()
        print("initial attribution method ... ")
        self.lig = LayerIntegratedGradients(self.model, self.model.embedding)

    def predict(self , text):
        words = self.tokenizer.preprocess(clean_text(text))
        if len(words) < self.seq_length:
            words += ['<pad>'] * (self.seq_length - len(words))
        elif len(words) > self.seq_length:
            words = words[:self.seq_length]
        tokens = [self.tokenizer.vocab.stoi[word] for word in words]
        tokens = torch.LongTensor(tokens).unsqueeze(0)
        reference_tokens = self.token_reference.generate_reference(self.seq_length, device='cpu').unsqueeze(0)
        pred = self.model(tokens)
        plabel = int(torch.argmax(pred, 1))
        pred = pred.tolist()[0]

        unpad_index = [idx for idx,word in enumerate(words) if word != '<pad>']
        unpad_words = [word for word in words if word != '<pad>']
        attributions = []
        for label in range(len(pred)):
            attributions.append(list(self.attribute(tokens, reference_tokens, label, unpad_index)))
        return unpad_words, pred, plabel, attributions

    def attribute(self, tokens, reference_tokens, target, unpad_index):
        attributions, delta = self.lig.attribute(tokens, reference_tokens, target=target,\
                                           return_convergence_delta=True)
        attributions = attributions.sum(dim=2).squeeze(0)
        attributions = attributions / torch.norm(attributions)
        attributions = attributions.cpu().detach().numpy()
        unpad_attributions = attributions[unpad_index]
        range_limit = np.max(np.abs(unpad_attributions))
        unpad_attributions /= range_limit
        return unpad_attributions

    
    def __repr__(self):
        return "prediction model for CNN"

    def __str__(self):
        return "prediction model for CNN"


class RobertaPredictionModel:
    def __init__(self, 
            model_path="static/chinese-roberta-wwm-ext5", 
            tokenizer_path="hfl/chinese-roberta-wwm-ext",
            lac_dict_path="static/addwords.txt"):
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.lac_dict_path = lac_dict_path
        self.initialize()

    def initialize(self):
        print("initial tokenizer...")
        self.tokenizer = BertTokenizer.from_pretrained(self.tokenizer_path)
        self.ref_token_id = self.tokenizer.pad_token_id
        self.sep_token_id = self.tokenizer.sep_token_id
        self.cls_token_id = self.tokenizer.cls_token_id
        print("initial inference model...")
        self.model = BertForSequenceClassification.from_pretrained(self.model_path,num_labels=3).cpu().eval()
        self.model.zero_grad()
        print("initial lac model...")
        self.lac = LAC(mode="seg")
        self.lac.load_customization(self.lac_dict_path, sep="\t")
        print("initial interpretable embedding layers ...")
        self.interpretable_embedding1 = configure_interpretable_embedding_layer(self.model, 'bert.embeddings.word_embeddings')
        self.interpretable_embedding2 = configure_interpretable_embedding_layer(self.model, 'bert.embeddings.token_type_embeddings')
        self.interpretable_embedding3 = configure_interpretable_embedding_layer(self.model, 'bert.embeddings.position_embeddings')
        remove_interpretable_embedding_layer(self.model, self.interpretable_embedding1)
        remove_interpretable_embedding_layer(self.model, self.interpretable_embedding2)
        remove_interpretable_embedding_layer(self.model, self.interpretable_embedding3)



    def model_predict(self, inputs, token_type_ids=None, position_ids=None, attention_mask=None):
        return self.model(inputs, token_type_ids=token_type_ids,
                 position_ids=position_ids, attention_mask=attention_mask, )[0]

    def forward_func(self, inputs, token_type_ids=None, position_ids=None, attention_mask=None):
        pred = self.model_predict(inputs,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask)
        return F.softmax(pred, dim=1)

    def construct_input_ref_pair(self, text, ref_token_id, sep_token_id, cls_token_id):
        text_ids = self.tokenizer.encode(text, add_special_tokens=False)
        input_ids = [cls_token_id] + text_ids + [sep_token_id]
        ref_input_ids = [cls_token_id] + [ref_token_id] * len(text_ids) + [sep_token_id]
        return torch.tensor([input_ids]), torch.tensor([ref_input_ids]), len(text_ids)

    def construct_input_ref_token_type_pair(self, input_ids, sep_ind=0):
        seq_len = input_ids.size(1)
        token_type_ids = torch.tensor([[0 if i <= sep_ind else 1 for i in range(seq_len)]])
        ref_token_type_ids = torch.zeros_like(token_type_ids)
        return token_type_ids, ref_token_type_ids

    def construct_input_ref_pos_id_pair(self, input_ids):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long)
        ref_position_ids = torch.zeros(seq_length, dtype=torch.long)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        ref_position_ids = ref_position_ids.unsqueeze(0).expand_as(input_ids)
        return position_ids, ref_position_ids

    def construct_attention_mask(self, input_ids):
        return torch.ones_like(input_ids)

    def construct_bert_sub_embedding(self, input_ids, ref_input_ids,
                                       token_type_ids, ref_token_type_ids,
                                       position_ids, ref_position_ids):
        input_embeddings = self.interpretable_embedding1.indices_to_embeddings(input_ids)
        ref_input_embeddings = self.interpretable_embedding1.indices_to_embeddings(ref_input_ids)
    
        input_embeddings_token_type = self.interpretable_embedding2.indices_to_embeddings(token_type_ids)
        ref_input_embeddings_token_type = self.interpretable_embedding2.indices_to_embeddings(ref_token_type_ids)
    
        input_embeddings_position_ids = self.interpretable_embedding3.indices_to_embeddings(position_ids)
        ref_input_embeddings_position_ids = self.interpretable_embedding3.indices_to_embeddings(ref_position_ids)
    
        return (input_embeddings, ref_input_embeddings), \
               (input_embeddings_token_type, ref_input_embeddings_token_type), \
               (input_embeddings_position_ids, ref_input_embeddings_position_ids)
    
    def word_level_spline(self, tokens, attributions_sum, lac):
        tokens = [each for each in tokens if each not in ('[CLS]','[SEP]','[UNK]')]
        cuts = [each for each in lac.run("".join(tokens)) if re.match("[\u4e00-\u9fa5]+", each)]
        attributions_word = []
        for cut in cuts:
            idxs = []
            for word in cut:
                if word in tokens:
                    idxs.append(tokens.index(word))
            if idxs:
                attr = [attributions_sum[idx] for idx in idxs]
                mean_attr = np.mean(attr)
                for idx in idxs:
                    attributions_sum[idx] = mean_attr
            else:
                mean_attr = 0
            attributions_word.append((cut, mean_attr))
        return attributions_sum, attributions_word

    def summarize_attributions(self, attributions):
        attributions = attributions.sum(dim=-1).squeeze(0)
        attributions = attributions / torch.norm(attributions)
        return attributions

    def predict(self, text):
        self.model.zero_grad()
        input_ids, ref_input_ids, sep_id = self.construct_input_ref_pair(text, self.ref_token_id, self.sep_token_id, self.cls_token_id)
        token_type_ids, ref_token_type_ids = self.construct_input_ref_token_type_pair(input_ids, sep_id)
        position_ids, ref_position_ids = self.construct_input_ref_pos_id_pair(input_ids)
        attention_mask = self.construct_attention_mask(input_ids)
        indices = input_ids[0].detach().tolist()
        tokens = self.tokenizer.convert_ids_to_tokens(indices)
        logits = self.forward_func(input_ids, \
              token_type_ids=token_type_ids, \
              position_ids=position_ids, \
              attention_mask=attention_mask)
        pred = int(logits.argmax(1))
        logits = logits.tolist()[0]

        self.interpretable_embedding1 = configure_interpretable_embedding_layer(self.model, 'bert.embeddings.word_embeddings')
        self.interpretable_embedding2 = configure_interpretable_embedding_layer(self.model, 'bert.embeddings.token_type_embeddings')
        self.interpretable_embedding3 = configure_interpretable_embedding_layer(self.model, 'bert.embeddings.position_embeddings')
        (input_embed, ref_input_embed), (token_type_ids_embed, ref_token_type_ids_embed), (position_ids_embed, ref_position_ids_embed) = self.construct_bert_sub_embedding(input_ids, ref_input_ids, \
                                                 token_type_ids=token_type_ids, ref_token_type_ids=ref_token_type_ids, \
                                                 position_ids=position_ids, ref_position_ids=ref_position_ids)
        
        lig = IntegratedGradients(self.forward_func)

        attr = []
        for label in range(len(logits)):
            attributions, delta = lig.attribute(inputs=(input_embed, token_type_ids_embed, position_ids_embed),
                                                baselines=(ref_input_embed, ref_token_type_ids_embed, ref_position_ids_embed),
                                                target=label,
                                                additional_forward_args=(attention_mask),
                                                return_convergence_delta=True)
            _, attribution_words = self.word_level_spline(tokens, self.summarize_attributions(attributions[0]).cpu().detach().numpy(),self.lac)
            _, attribution_position = self.word_level_spline(tokens, self.summarize_attributions(attributions[2]).cpu().detach().numpy(),self.lac)
            if len(attribution_words) != 0:
                words = [each[0] for each in attribution_words]
                attribution_merge = [attribution_words[i][1] + attribution_position[i][1] for i in range(len(attribution_words))]
                range_limit = np.max(np.abs(attribution_merge))
                attribution_merge /= range_limit
                attr.append(list(attribution_merge))
            else:
                words = []
                attr.append([])

        remove_interpretable_embedding_layer(self.model, self.interpretable_embedding1)
        remove_interpretable_embedding_layer(self.model, self.interpretable_embedding2)
        remove_interpretable_embedding_layer(self.model, self.interpretable_embedding3)

        return words, logits, pred, attr
    
    def __repr__(self):
        return "prediction model for Roberta"

    def __str__(self):
        return "prediction model for Roberta"

if __name__ == "__main__":
    print(RobertaPredictionModel.__name__)
    #model = RobertaPredictionModel()
    #texts="这是一个测试"
    #print(model.predict(texts))
