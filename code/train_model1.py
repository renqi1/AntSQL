import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, random_split
from Bert import Bert1
from transformers import BertTokenizer
from dataset import read_train_set, BuildDataSet1, convert_examples1
from utils import encode_columns, get_cls_idx, get_columns
from sklearn import metrics

columns = get_columns()

train_set_path = "../data/waic_nl2sql_train.jsonl"
pretrain_vocab_path = "../pretrain_model/ernie/vocab.txt"
pretrain_config_path = "../pretrain_model/ernie/bert_config.json"
pretrain_model_path = "../pretrain_model/ernie/pytorch_model.bin"
model1_save_path = "../my_model/model1_ernie.pkl"
hidden_size = 768
batch_size = 24
learning_rate = 0.00001

train_examples = read_train_set(train_set_path)
print('train examples got')

# train_examples = [
#  ['涨幅最大的板块南方誉隆一年持有期混合a的净值', 13, [[1, 4, 7, 19], [1, 4, 7, 19]], 0],
#  ['华宝消费的净值现回撤快10个', 13, [[1, 4, 0, 4]], 0],
#  ['产品的净值', 13, None, 0],
#  ['光大保德信银发商机主题混合型证券投资基金的净值推荐其他好的板块', 13, [[1, 4, 0, 20]], 0],
#  ['短期看好广发睿升c的净值', 13, [[1, 4, 4, 9]], 0]]

tokenizer = BertTokenizer.from_pretrained(pretrain_vocab_path)
columns_encode, columns_segment = encode_columns(columns, tokenizer)
features = convert_examples1(train_examples, columns_encode, columns_segment, tokenizer, que_length=64, max_length=512)
train_dataset = BuildDataSet1(features)

train_size = int(0.95 * len(train_dataset))
test_size = len(train_dataset) - train_size
train_set, val_set = random_split(train_dataset, [train_size, test_size])
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
print('load data finish')

clsidx = get_cls_idx(columns)
index = torch.LongTensor(clsidx).cuda()
model1 = Bert1(index=index, hidden_size=hidden_size, config=pretrain_config_path, model=pretrain_model_path).cuda()

loss_function = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model1.parameters(), lr=learning_rate)

pre_all_cond_conn = []
pre_all_sel_col = []
pre_all_conds_col = []
pre_all_conds_op = []
label_all_cond_conn = []
label_all_sel_col = []
label_all_conds_col = []
label_all_conds_op = []
best_acc = 0
total_batch = 0


def model_evaluate(model, data_iter):
    model.eval()
    pre_all_sel_col = []
    pre_all_conds_col = []
    pre_all_conds_op = []
    label_all_sel_col = []
    label_all_conds_col = []
    label_all_conds_op = []
    with torch.no_grad():
        for epoch_batch, (input_ids, attention_mask, token_type_ids, label_sel, label_conn, label_conds_col, label_conds_op) in enumerate(data_iter):
            model1.train()
            input_ids = Variable(input_ids).cuda()
            attention_mask = Variable(attention_mask).cuda()
            token_type_ids = Variable(token_type_ids).cuda()

            out_sel_col, out_conds_col, out_conds_op = model1(input_ids, attention_mask, token_type_ids)

            pre_sel_col = torch.max(out_sel_col.data, 1)[1].cpu().numpy()
            pre_conds_col = torch.max(out_conds_col.data, 1)[1].cpu().numpy()
            pre_conds_op = torch.max(out_conds_op.data, 1)[1].cpu().numpy()

            # pre_all_cond_conn.extend(pre_cond_conn)
            pre_all_sel_col.extend(pre_sel_col)
            pre_all_conds_col.extend(pre_conds_col)
            pre_all_conds_op.extend(pre_conds_op)

            label_all_sel_col.extend(label_sel.tolist())
            label_all_conds_col.extend(label_conds_col.tolist())
            label_all_conds_op.extend(label_conds_op.tolist())

        dev_sel_col_acc = metrics.accuracy_score(label_all_sel_col, pre_all_sel_col)
        dev_conds_col_acc = metrics.accuracy_score(label_all_conds_col, pre_all_conds_col)
        dev_conds_op_acc = metrics.accuracy_score(label_all_conds_op, pre_all_conds_op)

        return dev_sel_col_acc, dev_conds_col_acc, dev_conds_op_acc


for epoch in range(5):
    print('epoch', epoch)
    for epoch_batch, (input_ids, attention_mask, token_type_ids, label_sel, label_conn, label_conds_col, label_conds_op) in enumerate(train_loader):
        total_batch += 1
        model1.train()
        input_ids = Variable(input_ids).cuda()
        attention_mask = Variable(attention_mask).cuda()
        token_type_ids = Variable(token_type_ids).cuda()
        label_sel_ = Variable(label_sel).cuda()
        label_conds_col_ = Variable(label_conds_col).cuda()
        label_conds_op_ = Variable(label_conds_op).cuda()

        out_sel_col, out_conds_col, out_conds_op = model1(input_ids, attention_mask, token_type_ids)

        loss_sel_col = loss_function(out_sel_col, label_sel_)
        loss_conds_col = loss_function(out_conds_col, label_conds_col_)
        loss_conds_op = loss_function(out_conds_op, label_conds_op_)

        total_loss = 2 * loss_sel_col + 10 * loss_conds_col + 0.5 * loss_conds_op
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        pre_sel_col = torch.max(out_sel_col.data, 1)[1].cpu().numpy()
        pre_conds_col = torch.max(out_conds_col.data, 1)[1].cpu().numpy()
        pre_conds_op = torch.max(out_conds_op.data, 1)[1].cpu().numpy()

        pre_all_sel_col.extend(pre_sel_col)
        pre_all_conds_col.extend(pre_conds_col)
        pre_all_conds_op.extend(pre_conds_op)

        label_all_sel_col.extend(label_sel.tolist())
        label_all_conds_col.extend(label_conds_col.tolist())
        label_all_conds_op.extend(label_conds_op.tolist())

        if (epoch_batch + 1) % 20 == 0:
            train_sel_acc = metrics.accuracy_score(pre_all_sel_col, label_all_sel_col)
            train_conds_col_acc = metrics.accuracy_score(pre_all_conds_col, label_all_conds_col)
            train_conds_op_acc = metrics.accuracy_score(pre_all_conds_op, label_all_conds_op)

            print(epoch_batch + 1, '/', len(train_loader), '\t',
                  loss_sel_col.item(), '\t', loss_conds_col.item(), '\t', loss_conds_op.item(), '\t',
                  train_sel_acc, '\t', train_conds_col_acc, '\t', train_conds_op_acc, '\t', )

        if (epoch_batch + 1) % 200 == 0 and total_batch > 0.8 * len(train_loader):
            dev_sel_col_acc, dev_conds_col_acc, dev_conds_op_acc = model_evaluate(model1, test_loader)

            if dev_conds_col_acc > best_acc:
                best_acc = dev_conds_col_acc
                improve = '*'
                torch.save(model1.state_dict(), model1_save_path)
            else:
                improve = ''

            print(dev_sel_col_acc, '\t', dev_conds_col_acc, '\t', dev_conds_op_acc, '\t', improve)

            pre_all_sel_col = []
            pre_all_conds_col = []
            pre_all_conds_op = []
            label_all_sel_col = []
            label_all_conds_col = []
            label_all_conds_op = []

    torch.save(model1.state_dict(), 'epoch_model1_ernie.pkl')
