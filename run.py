import os
import sys
import torch
import random as r
import numpy as np
import pandas as pd
import torch.nn as nn
import argparse
from tqdm import tqdm
from sklearn import metrics
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from torch.autograd import Variable
from torch.backends import cudnn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
import pickle
import glob
import math

torch.manual_seed(2019)
torch.cuda.manual_seed_all(2019)
np.random.seed(2019)
r.seed(2019)

cudnn.benchmark = False
cudnn.deterministic = True


class Normalize:
    @staticmethod
    def normalize(X_train, X_val, max_len):
        scaler = MinMaxScaler()
        X_train, X_val = X_train.reshape(X_train.shape[0], -1), X_val.reshape(X_val.shape[0], -1)
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_train, X_val = X_train.reshape(X_train.shape[0], max_len, -1), X_val.reshape(X_val.shape[0], max_len, -1)

        return X_train, X_val

    @staticmethod
    def inverse(X_train, X_val):
        scaler = MinMaxScaler()
        X_train = scaler.inverse_transform(X_train)
        X_val = scaler.inverse_transform(X_val)
        return X_train, X_val


class EarlyStopping:
    def __init__(self, patience=10, verbose=False, delta=0, path='checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
        
def sizeof_fmt(num, suffix='B'):
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)

def import_data(model_name, dataset, max_len):
    with open(os.path.join(args['data_dir'], f'{model_name}_{dataset}.p'), 'rb') as fp:
        pre_trained_dict = pickle.load(fp)

    pre_trained_vec = [pre_trained_dict[i] for i in pre_trained_dict.keys()]
    vec = np.zeros((len(pre_trained_vec), max_len, 768))

    for idx, doc in enumerate(pre_trained_vec):
        if doc.shape[0] <= max_len:
            vec[idx][:doc.shape[0], :] = doc
        else:
            vec[idx][:max_len, :] = doc[:max_len, :]

    print('positive example shape: ', vec.shape)
    TEXT_emb = vec
    assert TEXT_emb.shape == (len(pre_trained_vec), max_len, 768)
    print(sizeof_fmt(sys.getsizeof(TEXT_emb)))

    del vec
    labels = pd.read_excel(os.path.join(args['raw_dir'], 'labels.xlsx'))
    LABEL_emb = np.array(labels.TC_10_bin10.tolist())

    return TEXT_emb, LABEL_emb


def gelu_new(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


ACT2FN = {"gelu_new": gelu_new}


class BertConfig:
    def __init__(self, hidden_size=768, num_hidden_layers=4, num_attention_heads=1, intermediate_size=3072,
                 hidden_act="relu", hidden_dropout_prob=0.01, attention_probs_dropout_prob=0.01, seq_length=128,
                 initializer_range=0.02, layer_norm_eps=1e-12, output_attentions=True, output_hidden_states=False,
                 num_labels=2):
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.seq_length = seq_length
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states
        self.num_labels = num_labels


class BertLayerNorm(nn.LayerNorm):
    def __init__(self, hidden_size, eps):
        super().__init__(hidden_size, eps=eps)


class BertEmbeddings(nn.Module):
    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.position_embeddings = nn.Embedding(config.seq_length, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, inputs_embeds, position_ids=None):
        input_shape = inputs_embeds.size()[:-1]
        seq_length = input_shape[1]
        device = inputs_embeds.device

        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(input_shape)

        position_embeddings = self.position_embeddings(position_ids)
        embeddings = inputs_embeds + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return (context_layer, attention_probs) if self.output_attentions else (context_layer,)


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, hidden_states, attention_mask=None):
        self_outputs = self.self(hidden_states, attention_mask)
        attention_output = self.output(self_outputs[0], hidden_states)
        return (attention_output,) + self_outputs[1:]


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask=None):
        self_attention_outputs = self.attention(hidden_states, attention_mask)
        attention_output = self_attention_outputs[0]
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return (layer_output,) + self_attention_outputs[1:]


class BertEncoder(nn.Module):
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states

    def forward(self, hidden_states, attention_mask=None):
        all_hidden_states = ()
        all_attentions = ()

        for layer_module in self.layer:
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(hidden_states, attention_mask)
            hidden_states = layer_outputs[0]

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        return (hidden_states,) + (all_hidden_states,) + (all_attentions,)


class BertPooler(nn.Module):
    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertModel(nn.Module):
    def __init__(self, config):
        super(BertModel, self).__init__()
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)

    def forward(self, inputs_embeds, attention_mask=None, position_ids=None):
        input_shape = inputs_embeds.size()[:-1]
        device = inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)

        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(inputs_embeds=inputs_embeds, position_ids=position_ids)
        encoder_outputs = self.encoder(embedding_output, attention_mask=extended_attention_mask)
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        return (sequence_output, pooled_output) + encoder_outputs[1:]


class HTransformer(nn.Module):
    def __init__(self, config):
        super(HTransformer, self).__init__()
        self.config = config
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)

    def forward(self, x):
        outputs = self.bert(inputs_embeds=x)
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return (logits,) + outputs[2:]


def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=config.initializer_range)
    elif isinstance(module, BertLayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)
    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script arguments')
    
    # Adding arguments to the parser
    parser.add_argument('--data_dir', type=str, default='data/', help='Directory for data')
    parser.add_argument('--raw_dir', type=str, default='raw/', help='Directory for raw data')
    parser.add_argument('--output_dir', type=str, default='outputs/', help='Directory for output data')
    parser.add_argument('--verbose_output_dir', type=str, default='outputs/', help='Directory for verbose output data')
    parser.add_argument('--attention_output_dir', type=str, default='attentions/', help='Directory for attention outputs')
    parser.add_argument('--cuda_num', type=int, default=0, help='CUDA device number')
    
    parser.add_argument('--dataset_name', type=str, default='dataset', help='Name of the dataset')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--no_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--max_len', type=int, default=30, help='Maximum sequence length')
    parser.add_argument('--random_seeds', nargs='+', default=['1988'], help='List of random seeds')
    parser.add_argument('--model_name', nargs='+', default=['sciBERT'], help='List of model names')

    args = parser.parse_args()
    
    # Accessing the arguments
    data_dir = args.data_dir
    raw_dir = args.raw_dir
    output_dir = args.output_dir
    verbose_output_dir = args.verbose_output_dir
    attention_output_dir = args.attention_output_dir
    cuda_num = args.cuda_num
    
    dataset = args.dataset_name
    max_len = args.max_len
    lr = args.learning_rate
    no_epochs = args.no_epochs
    random_seeds = args.random_seeds
    models = args.model_name

    for model in models:
        model_name = model
        train_size = 'train_size'
        random_states = [int(number) for number in parser['random_seeds']]

        print('number of epochs:', no_epochs)
        print('dataset name:', dataset)
        print('initial random states:', random_states)

        gradient_clipping = 1.0
        train_batch = 512
        val_batch = 512

        TEXT_emb, LABEL_emb = import_data(model_name, dataset, max_len)
        config = BertConfig(seq_length=max_len, hidden_act='gelu_new')
        kfold = StratifiedKFold(n_splits=5, shuffle=True)

        df_all = pd.DataFrame()
        df_all_mcc = pd.DataFrame()
        best_mccs = []
        best_raws = []
        fold = 0

        X_train, X_test, y_train, y_test = train_test_split(TEXT_emb, LABEL_emb, test_size=0.1, random_state=1988, stratify=LABEL_emb)
        for fold in [0]:
            confusion_matrices = []
            aucs = []
            mccs = []
            train_confusion_matrices = []
            train_aucs = []
            train_mccs = []

            normalizer = Normalize()
            X_train, X_test = normalizer.normalize(X_train, X_test, max_len)

            tensor_train_x = torch.from_numpy(X_train).type(torch.FloatTensor)
            tensor_train_y = torch.from_numpy(y_train).type(torch.LongTensor)

            tensor_val_x = torch.from_numpy(X_test).type(torch.FloatTensor)
            tensor_val_y = torch.from_numpy(y_test).type(torch.LongTensor)

            training_set = TensorDataset(tensor_train_x, tensor_train_y)
            val_set = TensorDataset(tensor_val_x, tensor_val_y)

            trainloader = DataLoader(training_set, batch_size=train_batch, shuffle=True, num_workers=1)
            testloader = DataLoader(val_set, batch_size=val_batch, shuffle=False, num_workers=1)

            model = HTransformer(config=config)
            model.apply(init_weights)
            model.cuda(args['cuda_num'])
            model.to('cuda')

            opt = torch.optim.Adam(lr=lr, params=model.parameters())
            train_losses = []
            val_losses = []
            macro_f = []

            best_auc = 0.0

            for e in tqdm(range(no_epochs)):
                print('\n epoch', e)

                for i, data in enumerate(tqdm(trainloader)):
                    model.train(True)

                    opt.zero_grad()

                    inputs, labels = data
                    if inputs.size(1) > config.seq_length:
                        inputs = inputs[:, :config.seq_length, :]

                    if torch.cuda.is_available():
                        inputs, labels = Variable(inputs.cuda(args['cuda_num'])), labels.cuda(args['cuda_num'])

                    out = model(inputs)

                    weight = torch.tensor([5.0, 1.0]).cuda(args['cuda_num'])
                    loss = nn.CrossEntropyLoss(weight, reduction='mean')

                    output = loss(out[0], labels)
                    print('epoch', e, 'step', i, 'loss:', output.item(), 'num of postives', labels.sum())


                    train_loss_tol = float(output.cpu())

                    output.backward()

                    if gradient_clipping > 0.0:
                        nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)

                    opt.step()

                    del inputs, labels, out, output
                    torch.cuda.empty_cache()
                    train_losses.append(train_loss_tol)

                    with torch.no_grad():
                        model.train(False)
                        y_eval_pred = []
                        y_eval_true = []
                        y_eval_prob = []

                        y_train_pred = []
                        y_train_true = []
                        y_train_prob = []

                        train_attention_scores = torch.Tensor()
                        attention_scores = torch.Tensor()

                        for data in tqdm(trainloader):
                            inputs, labels = data
                            if inputs.size(1) > config.seq_length:
                                inputs = inputs[:, :config.seq_length, :]

                            if torch.cuda.is_available():
                                inputs, labels = Variable(inputs.cuda(args['cuda_num'])), labels.cuda(args['cuda_num'])

                            out = model(inputs)

                            sm = nn.Softmax(dim=1)
                            pred_prob = sm(out[0].cpu())

                            if config.output_attentions:
                                last_layer_attention = out[1][-1].cpu()
                                train_attention_scores = torch.cat((train_attention_scores, last_layer_attention))

                            predict = torch.argmax(pred_prob, axis=1)
                            labels = labels.cpu()
                            y_train_pred += predict.tolist()
                            y_train_true += labels.tolist()
                            y_train_prob += pred_prob.tolist()

                            del inputs, labels, out

                        for data in tqdm(testloader):
                            inputs, labels = data
                            if inputs.size(1) > config.seq_length:
                                inputs = inputs[:, :config.seq_length, :]

                            if torch.cuda.is_available():
                                inputs, labels = Variable(inputs.cuda(args['cuda_num'])), labels.cuda(args['cuda_num'])

                            out = model(inputs)
                            loss = nn.CrossEntropyLoss(weight, reduction='mean')
                            output = loss(out[0], labels)
                            val_loss_tol = float(output.cpu())
                            val_losses.append(val_loss_tol)

                            sm = nn.Softmax(dim=1)
                            pred_prob = sm(out[0].cpu())

                            if config.output_attentions:
                                last_layer_attention = out[1][-1].cpu()
                                attention_scores = torch.cat((attention_scores, last_layer_attention))

                            predict = torch.argmax(pred_prob, axis=1)
                            labels = labels.cpu()
                            y_eval_pred += predict.tolist()
                            y_eval_true += labels.tolist()
                            y_eval_prob += pred_prob.tolist()

                            del inputs, labels, out

                        train_acc = accuracy_score(y_train_true, y_train_pred)
                        train_f_score = f1_score(y_train_true, y_train_pred, average='macro')
                        train_matrix = confusion_matrix(y_train_true, y_train_pred, labels=[0, 1]).ravel()
                        train_fpr, train_tpr, _ = metrics.roc_curve(y_train_true, np.array(y_train_prob)[:, 1], pos_label=0)
                        train_step_auc = metrics.auc(train_fpr, train_tpr)
                        train_aucs.append(train_step_auc)
                        train_mccs.append(metrics.matthews_corrcoef(y_train_true, y_train_pred))
                        train_confusion_matrices.append(train_matrix)

                        print('Epoch:', e, 'step:', i, 'training accuracy:', train_acc)
                        print('Epoch:', e, 'step:', i, 'training AUC:', train_step_auc)

                        acc = accuracy_score(y_eval_true, y_eval_pred)
                        f_score = f1_score(y_eval_true, y_eval_pred, average='macro')
                        confusion_matrices.append(confusion_matrix(y_eval_true, y_eval_pred, labels=[0, 1]).ravel())
                        fpr, tpr, _ = metrics.roc_curve(y_eval_true, np.array(y_eval_prob)[:, 1], pos_label=0)
                        step_auc = metrics.auc(fpr, tpr)
                        aucs.append(step_auc)
                        mccs.append(metrics.matthews_corrcoef(y_eval_true, y_eval_pred))

                        print(classification_report(y_eval_true, y_eval_pred), acc, f_score, step_auc)
                        macro_f.append(f_score)

                        if config.output_attentions:
                            attention_dir = os.path.join(args['attention_output_dir'], f'{dataset}_{model_name}/')
                            os.makedirs(attention_dir, exist_ok=True)

                            if best_auc < train_step_auc:
                                best_auc = train_step_auc
                                glob_pattern = os.path.join(attention_dir, f'*_fold{fold}_size{train_size}_model{model_name}.pt')
                                for f in glob.glob(glob_pattern):
                                    os.remove(f)

                                doc_attention_score = [doc[0] for batch in attention_scores for doc in batch]
                                final_attention_scores = np.zeros((len(TEXT_emb), config.seq_length))

                                for idx, attention_matrix in enumerate(doc_attention_score):
                                    sum_attention_score = np.array(attention_matrix.cpu()).sum(axis=0)
                                    final_attention_scores[idx, :] = sum_attention_score

                                print(final_attention_scores.shape)
                                torch.save(attention_scores, os.path.join(attention_dir, f'epoch{e}_fold{fold}_size{train_size}_model{model_name}.pt'))

            df = pd.DataFrame({fold: [row for row in confusion_matrices]})
            df.to_csv(os.path.join(args['verbose_output_dir'], f'all_raw_{dataset}_h{config.num_attention_heads}_s{fold}_size{train_size}_model{model_name}.csv'), index=False)

            df_auc = pd.DataFrame({fold: aucs})
            df_auc.to_csv(os.path.join(args['verbose_output_dir'], f'all_auc_{dataset}_h{config.num_attention_heads}_s{fold}_size{train_size}_model{model_name}.csv'), index=False)

            df_mcc = pd.DataFrame({fold: mccs})
            df_mcc.to_csv(os.path.join(args['verbose_output_dir'], f'all_mcc_{dataset}_h{config.num_attention_heads}_s{fold}_size{train_size}_model{model_name}.csv'), index=False)

            df_loss = pd.DataFrame({fold: train_losses})
            df_loss.to_csv(os.path.join(args['verbose_output_dir'], f'all_loss_{dataset}_h{config.num_attention_heads}_s{fold}_size{train_size}_model{model_name}.csv'), index=False)

            df_train = pd.DataFrame({fold: [row for row in train_confusion_matrices]})
            df_train.to_csv(os.path.join(args['verbose_output_dir'], f'all_train_raw_{dataset}_h{config.num_attention_heads}_s{fold}_size{train_size}_model{model_name}.csv'), index=False)

            df_train_auc = pd.DataFrame({fold: train_aucs})
            df_train_auc.to_csv(os.path.join(args['verbose_output_dir'], f'all_train_auc_{dataset}_h{config.num_attention_heads}_s{fold}_size{train_size}_model{model_name}.csv'), index=False)

            df_train_mcc = pd.DataFrame({fold: train_mccs})
            df_train_mcc.to_csv(os.path.join(args['verbose_output_dir'], f'all_train_mcc_{dataset}_h{config.num_attention_heads}_s{fold}_size{train_size}_model{model_name}.csv'), index=False)

            max_ids = np.argwhere(np.array(train_mccs) == np.amax(train_mccs))[-1][0]
            best_mccs.append(mccs[max_ids])
            best_raws.append(confusion_matrices[max_ids])

            fold += 1
            del model

        df_all['result'] = best_raws
        df_all.to_csv(os.path.join(args['output_dir'], f'raw_{dataset}_hbm_{train_size}_model{model_name}.csv'), index=True)

        df_all_mcc['result'] = best_mccs
        df_all_mcc.to_csv(os.path.join(args['output_dir'], f'mcc_{dataset}_hbm_{train_size}_model{model_name}.csv'), index=True)