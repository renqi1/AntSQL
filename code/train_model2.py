import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from Bert import Bert2
from transformers import BertTokenizer
from dataset import read_train_set, BuildDataSet2, convert_examples2
from utils import get_columns

train_set_path = "../data/waic_nl2sql_train.jsonl"
pretrain_vocab_path = "../pretrain_model/ernie/vocab.txt"
pretrain_config_path = "../pretrain_model/ernie/bert_config.json"
pretrain_model_path = "../pretrain_model/ernie/pytorch_model.bin"
model2_save_path = "../my_model/model2_ernie.pkl"
hidden_size = 768
batch_size = 32
learning_rate = 0.00001

columns = get_columns()
train_examples = read_train_set(train_set_path)
print('train examples got')

tokenizer = BertTokenizer.from_pretrained(pretrain_vocab_path)

features = convert_examples2(train_examples, columns, tokenizer, max_length=128, train=True)
train_dataset = BuildDataSet2(features)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
print('load data finish')

model2 = Bert2(hidden_size=hidden_size, config=pretrain_config_path, model=pretrain_model_path).cuda()

loss_function = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model2.parameters(), lr=learning_rate)
labels_all = []
predict_all = []
model2.train()
for epoch in range(10):
    print('epoch', epoch)
    for epoch_batch, (input_ids, attention_mask, token_type_ids, conds_value, label_conn) in enumerate(train_loader):
        input_ids = Variable(input_ids).cuda()
        attention_mask = Variable(attention_mask).cuda()
        token_type_ids = Variable(token_type_ids).cuda()
        label_conn = Variable(label_conn).cuda()
        conds_value = Variable(conds_value).cuda().transpose(1, 2).reshape(-1, 63)

        out_conds_value, out_conn = model2(input_ids, attention_mask, token_type_ids)

        loss_conn = loss_function(out_conn, label_conn)
        out_conds_value = out_conds_value.transpose(1, 2).reshape(-1, 63)
        loss_cond_value = loss_function(out_conds_value, conds_value)

        total_loss = 5 * loss_cond_value + loss_conn

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if (epoch_batch + 1) % 20 == 0:
            print(epoch_batch + 1, '/', len(train_loader), '\t', loss_cond_value.item(), '\t', loss_conn.item())

        if (epoch_batch + 1) % 100 == 0:
            torch.save(model2.state_dict(), model2_save_path)
            print('model saved')
