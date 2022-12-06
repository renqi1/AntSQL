import torch
import torch.nn as nn
from transformers import BertConfig, BertModel
from crf import CRF


class Bert1(nn.Module):
    
    def __init__(self, index, hidden_size, config, model):
        super(Bert1, self).__init__()
        model_config = BertConfig.from_pretrained(config)
        self.bert = BertModel.from_pretrained(model, config=model_config)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.index = index
        self.dropout = nn.Dropout(0.1)
        self.hidden_size = hidden_size
        self.classifier_cond_conn = nn.Linear(self.hidden_size, 3)
        self.classifier_conds_col = nn.Linear(self.hidden_size, 1)
        self.classifier_conds_op = nn.Linear(self.hidden_size, 8)
        self.classifier_sel_col = nn.Linear(self.hidden_size, 1)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = outputs[1]
        last_hidden_state = outputs[0]

        pooled_output = self.dropout(pooled_output)
        last_hidden_state = self.dropout(last_hidden_state)

        # out_cond_conn = self.classifier_cond_conn(pooled_output)
        out_conds_op = self.classifier_conds_op(pooled_output)

        cls_cols = last_hidden_state.gather(dim=1, index=self.index.unsqueeze(-1).unsqueeze(0).expand(
            last_hidden_state.shape[0], -1, last_hidden_state.shape[-1]))  # (batch, 52 , 768)

        out_conds_col = self.classifier_conds_col(cls_cols).squeeze(-1)
        out_sel_col = self.classifier_sel_col(cls_cols).squeeze(-1)

        return out_sel_col, out_conds_col, out_conds_op


class Bert2(nn.Module):

    def __init__(self, hidden_size, config, model):
        super(Bert2, self).__init__()
        model_config = BertConfig.from_pretrained(config)
        self.bert = BertModel.from_pretrained(model, config=model_config)
        for param in self.bert.parameters():
            param.requires_grad = True

        self.dropout = nn.Dropout(0.1)
        self.hidden_size = hidden_size
        self.classifier_conds_value = nn.Linear(self.hidden_size, 2)
        self.classifier_conn_op = nn.Linear(self.hidden_size, 3)
        self.crf = CRF(num_tags=3, batch_first=True)
        self.classifier_crf = nn.Linear(self.hidden_size, 3)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):

        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        pooled_output = outputs[1]
        last_hidden_state = outputs[0]
        pooled_output = self.dropout(pooled_output)
        last_hidden_state = self.dropout(last_hidden_state)

        out_conn_op = self.classifier_conn_op(pooled_output)

        conds_value = last_hidden_state[:, 1:64, :]
        out_conds_value = self.classifier_conds_value(conds_value)

        if labels is not None:
            logits = self.classifier_crf(conds_value)
            attention_mask = attention_mask[:, 1:64]
            labels = torch.LongTensor(labels.numpy())
            loss = self.crf(emissions=logits, tags=labels, mask=attention_mask)
            return -1 * loss, logits  # (loss), scores

        return out_conds_value, out_conn_op
