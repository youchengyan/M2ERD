# coding=utf-8
import sys
from datetime import datetime
import os
import random

import numpy as np
import torch
from ray import tune
from scipy.fftpack import fft, dct

from sklearn.metrics import classification_report
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertConfig, BertModel, get_linear_schedule_with_warmup
from torch.utils.data import RandomSampler, DataLoader
from torchvision import transforms
from tabulate import tabulate

from EarlyStopping import EarlyStopping
from fusion import NetShareFusion

random.seed(42)
environment = os.path.exists("../fakenews_experimental_environment")
if environment:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def process_dct_img(img):
    img = img.numpy()  # size = [1, 224, 224]
    height = img.shape[1]
    width = img.shape[2]
    N = 8
    step = int(height / N)  # 28

    dct_img = np.zeros((1, N * N, step * step, 1), dtype=np.float32)  # [1,64,784,1]
    fft_img = np.zeros((1, N * N, step * step, 1))
    # print('dct_img:{}'.format(dct_img.shape))

    i = 0
    for row in np.arange(0, height, step):
        for col in np.arange(0, width, step):
            block = np.array(img[:, row:(row + step), col:(col + step)], dtype=np.float32)
            # print('block:{}'.format(block.shape))
            block1 = block.reshape(-1, step * step, 1)  # [batch_size,784,1]
            dct_img[:, i, :, :] = dct(block1)  # [batch_size, 64, 784, 1]

            i += 1

    fft_img[:, :, :, :] = fft(dct_img[:, :, :, :]).real  # [batch_size,64, 784,1]

    fft_img = torch.from_numpy(fft_img).float()  # [batch_size, 64, 784, 1]
    new_img = F.interpolate(fft_img, size=[250, 1])  # [batch_size, 64, 250, 1]
    new_img = new_img.squeeze(0).squeeze(-1)  # torch.size = [64, 250]

    return new_img


# normalization
def normalize_to_sum(numbers):
    total = sum(numbers)
    normalized_numbers = [num / total for num in numbers]

    return normalized_numbers


class MyDataset:
    def __init__(self, data, VOCAB, max_sen_len, transform_vgg=None, transform_dct=None):
        super(MyDataset, self).__init__()

        self.transform_vgg = transform_vgg
        self.transform_dct = transform_dct
        self.tokenizer = BertTokenizer.from_pretrained(VOCAB)
        self.max_sen_len = max_sen_len

        self.post_id = torch.from_numpy(data['post_id'])
        self.tweet_content = data['post_content']
        self.image = list(data['image'])
        self.label = torch.from_numpy(data['label'])  # type:int
        self.llm_response = data['llm_response']
        self.llm_context = data['llm_context']
        self.llm_score = data['llm_score']

    def __getitem__(self, idx):
        content = str(self.tweet_content[idx])
        text_content = self.tokenizer.encode_plus(content, add_special_tokens=True, padding='max_length',
                                                  truncation=True, max_length=self.max_sen_len, return_tensors='pt')
        dct_img = self.transform_dct(self.image[idx].convert('L'))
        dct_img = process_dct_img(dct_img)

        # about llm response
        response = str(self.llm_response[idx]).replace('&nbsp', '')  # entire response
        response_content = self.tokenizer.encode_plus(response, add_special_tokens=True, padding='max_length',
                                                      truncation=True, max_length=self.max_sen_len, return_tensors='pt')

        llm_context_text_input_ids = []  # seperate response
        llm_context_attention_mask = []
        llm_context_token_type_ids = []
        llm_context_scores = []
        for i, item in enumerate(self.llm_context[idx]):
            if i > 2:
                continue
            llm_context_content = self.tokenizer.encode_plus(item, add_special_tokens=True, padding='max_length',
                                                             truncation=True, max_length=self.max_sen_len,
                                                             return_tensors='pt')
            llm_context_text_input_ids.append(
                llm_context_content["input_ids"].flatten().clone().detach().type(torch.LongTensor))
            llm_context_attention_mask.append(
                llm_context_content["attention_mask"].flatten().clone().detach().type(torch.LongTensor))
            llm_context_token_type_ids.append(
                llm_context_content["token_type_ids"].flatten().clone().detach().type(torch.LongTensor))
            if self.llm_score[idx][i] == 'null' or self.llm_score[idx][i] == 'undefined':
                llm_context_scores.append(
                    1)
            else:
                llm_context_scores.append(int(self.llm_score[idx][i]))

        llm_context_text_input_ids_tensor = torch.stack(llm_context_text_input_ids,
                                                        dim=0)
        llm_context_attention_mask_tensor = torch.stack(llm_context_attention_mask,
                                                        dim=0)
        llm_context_token_type_ids_tensor = torch.stack(llm_context_token_type_ids,
                                                        dim=0)
        llm_context_scores_normal = normalize_to_sum(llm_context_scores)
        llm_context_scores_tensor = torch.tensor(llm_context_scores_normal)

        return {
            "text_input_ids": text_content["input_ids"].flatten().clone().detach().type(torch.LongTensor),
            "attention_mask": text_content["attention_mask"].flatten().clone().detach().type(torch.LongTensor),
            "token_type_ids": text_content["token_type_ids"].flatten().clone().detach().type(torch.LongTensor),
            "llm_text_input_ids": response_content["input_ids"].flatten().clone().detach().type(torch.LongTensor),
            "llm_attention_mask": response_content["attention_mask"].flatten().clone().detach().type(torch.LongTensor),
            "llm_token_type_ids": response_content["token_type_ids"].flatten().clone().detach().type(torch.LongTensor),
            "llm_context_text_input_ids": llm_context_text_input_ids_tensor,
            "llm_context_attention_mask": llm_context_attention_mask_tensor,
            "llm_context_token_type_ids": llm_context_token_type_ids_tensor,
            "llm_context_scores": llm_context_scores_tensor,
            "image": self.transform_vgg(self.image[idx]),
            "dct_img": dct_img,
            "post_id": self.post_id[idx],
            "label": self.label[idx],
        }

    def __len__(self):
        return len(self.label)


#  setup() -> step() train() -> save_checkpoint() -> load_checkpoint()
class TrainALL(tune.Trainable):
    def get_dataloader(self):

        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)

        if self.dataset_name == 'weibo':
            from dataprocess import data_process_weibo as pro
        elif self.dataset_name == 'twitter':
            # import data_process_twitter as pro
            from dataprocess import data_process_weibo as pro
        image_list = pro.read_images(pro.image_file_list)


        train_data, train_data_num = pro.get_data('train', image_list)
        test_data, valid_data_num = pro.get_data('test', image_list)

        if self.dataset_name == 'twitter':
            transform_vgg = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.454, 0.440, 0.423], [0.282, 0.278, 0.278])
            ])
            transform_dct = transforms.Compose(
                [transforms.Resize((224, 224)),
                 transforms.ToTensor()
                 ])

        elif self.dataset_name == 'weibo':
            transform_vgg = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])
            transform_dct = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])
        else:
            raise 'Dataset Error'

        train_dataset = MyDataset(data=train_data,
                                  VOCAB=self.VOCAB,
                                  max_sen_len=self.max_sen_len,
                                  transform_vgg=transform_vgg,
                                  transform_dct=transform_dct)
        train_sampler = RandomSampler(train_dataset)
        train_loader = DataLoader(dataset=train_dataset,
                                  sampler=train_sampler,
                                  batch_size=self.train_bs,
                                  num_workers=0,
                                  drop_last=True)
        test_dataset = MyDataset(data=test_data,
                                 VOCAB=self.VOCAB,
                                 max_sen_len=self.max_sen_len,
                                 transform_vgg=transform_vgg,
                                 transform_dct=transform_dct)
        test_sampler = RandomSampler(test_dataset)
        test_loader = DataLoader(dataset=test_dataset,
                                 sampler=test_sampler,
                                 batch_size=self.test_bs,
                                 num_workers=0)
        return train_loader, test_loader

    def get_optimizer(self):
        no_decay = [
            "bias",
            "gamma",
            "beta",
            "LayerNorm.weight",
            "bn_text.weight",
            "bn_dct.weight",
            "bn_1.weight",
        ]

        bert_param_optimizer = list(self.model.bert.named_parameters())
        vgg_param_optimizer = list(self.model.vgg.named_parameters())
        dtcconv_param_optimizer = list(self.model.dct_img.named_parameters())
        fusion_param_optimizer = list(
            self.model.fusion_layers.named_parameters()
        )
        linear_param_optimizer = (
                list(self.model.linear_text.named_parameters())
                + list(self.model.linear_image.named_parameters())
                + list(self.model.linear_dct.named_parameters())
        )
        classifier_param_optimizer = list(self.model.linear1.named_parameters()) + list(
            self.model.linear2.named_parameters()
        )
        optimizer_grouped_parameters = [
            # bert_param_optimizer
            {"params": [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
             "weight_decay": self.weight_decay,
             "lr": self.bert_learning_rate, },
            {"params": [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0,
             "lr": self.bert_learning_rate, },
            # vgg_param_optimizer
            {"params": [p for n, p in vgg_param_optimizer if not any(nd in n for nd in no_decay)],
             "weight_decay": self.weight_decay,
             "lr": self.vgg_learning_rate, },
            {"params": [p for n, p in vgg_param_optimizer if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0,
             "lr": self.vgg_learning_rate, },
            # dtcconv_param_optimizer
            {"params": [p for n, p in dtcconv_param_optimizer if not any(nd in n for nd in no_decay)],
             "weight_decay": self.weight_decay,
             "lr": self.dtcconv_learning_rate, },
            {"params": [p for n, p in dtcconv_param_optimizer if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0,
             "lr": self.dtcconv_learning_rate, },
            # fusion_param_optimizer
            {"params": [p for n, p in fusion_param_optimizer if not any(nd in n for nd in no_decay)],
             "weight_decay": self.weight_decay,
             "lr": self.fusion_learning_rate, },
            {"params": [p for n, p in fusion_param_optimizer if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0,
             "lr": self.fusion_learning_rate, },
            # linear_param_optimizer
            {"params": [p for n, p in linear_param_optimizer if not any(nd in n for nd in no_decay)],
             "weight_decay": self.weight_decay,
             "lr": self.linear_learning_rate, },
            {"params": [p for n, p in linear_param_optimizer if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0,
             "lr": self.linear_learning_rate, },
            # classifier_param_optimizer
            {"params": [p for n, p in classifier_param_optimizer if not any(nd in n for nd in no_decay)],
             "weight_decay": self.weight_decay,
             "lr": self.classifier_learning_rate, },
            {"params": [p for n, p in classifier_param_optimizer if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0,
             "lr": self.classifier_learning_rate, },
        ]

        if self.optimizer_name == "SGD":
            optimizer = torch.optim.SGD(
                optimizer_grouped_parameters,
                lr=self.learning_rate,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
            )
        elif self.optimizer_name == "Adam":
            optimizer = torch.optim.Adam(
                optimizer_grouped_parameters,
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        elif self.optimizer_name == "AdamW":
            optimizer = torch.optim.AdamW(
                optimizer_grouped_parameters,
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        elif self.optimizer_name == "AdaBelief":
            from adabelief_pytorch import AdaBelief
            optimizer = AdaBelief(
                optimizer_grouped_parameters,
                lr=self.learning_rate,
                eps=1e-10,  # or 1e-16
                betas=(0.9, 0.999),
                weight_decouple=True,
                rectify=False)
        else:
            raise 'optimizer WRONG'
        return optimizer

    def get_scheduler(self):
        # Total number of training steps is number of batches * number of epochs.
        total_steps = len(self.train_loader) * self.epochs

        # Create the learning rate scheduler.
        scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=round(total_steps * self.warm_up_percentage),
            num_training_steps=total_steps
        )
        return scheduler

    def init_network(self, exclude_list=['bert', 'vgg']):
        if self.init_method != 'default':
            for name, w in self.model.named_parameters():
                cross = [val for val in exclude_list if val in name.split('.')]
                if cross == []:
                    if [val for val in ['bn_text', 'bn_vgg', 'bn_dct', 'bn_1', 'layer_norm'] if
                        val in name.split('.')] == []:
                        if 'weight' in name:

                            if self.init_method == 'xavier-normal':
                                nn.init.xavier_normal_(w)
                            elif self.init_method == 'xavier-uniform':
                                nn.init.xavier_uniform_(w)
                            elif self.init_method == 'kaiming-normal':
                                nn.init.kaiming_normal_(w)
                            elif self.init_method == 'kaiming-uniform':
                                nn.init.kaiming_uniform_(w)
                            else:
                                pass
                        elif 'bias' in name:
                            nn.init.constant_(w, 0)
                        else:
                            pass

    def get_model(self):
        model = NetShareFusion(CASED=self.CASED,  # text model
                               pthfile=self.pthfile,  # image model
                               kernel_sizes=self.kernel_sizes,
                               num_channels=self.num_channels,
                               num_layers=self.num_layers,
                               num_heads=self.num_heads,
                               model_dim=self.model_dim,
                               dropout=self.dropout,
                               confidence=self.confidence,
                               drop_and_BN=self.drop_and_BN)

        if self.FREEZE_BERT:
            for name, param in model.named_parameters():
                if "bert" in name:
                    param.requires_grad = False

        if self.FREEZE_VGG:
            for name, param in model.named_parameters():
                if "vgg" in name:
                    param.requires_grad = False

        return model

    def flat_accuracy(self, preds, labels):
        pred_flat = np.argmax(preds, axis=1)
        labels_flat = labels
        return np.sum(pred_flat == labels_flat) / len(labels)

    def config_check(self):
        if self.dataset_name == 'weibo' and 'multilingual' in self.CASED:
            raise ('Using weibo dataset with multilingual model!')
        if self.dataset_name == 'twitter' and 'chinese' in self.CASED:
            raise ('Using twitter dataset with chinese model!')

    def setup(self, config):
        self.config = config
        if environment:
            self.CASED = 'bert-base-uncased'  # multilingual-cased
            self.VOCAB = 'bert-base-uncased/vocab.txt'
            self.pthfile = 'vgg19_dcbb9e9d.pth'
            self.save_root = 'output'

        self.init_method = config.get("init_method")
        self.max_grad_norm = 1.0
        self.warm_up_percentage = 0.1
        self.early_stopping_patience = config.get("early_stopping_patience")
        self.early_stopping = EarlyStopping(patience=self.early_stopping_patience, verbose=True)

        self.bert_learning_rate = config.get("bert_learning_rate")
        self.vgg_learning_rate = config.get("vgg_learning_rate")
        self.dtcconv_learning_rate = config.get("dtcconv_learning_rate")
        self.fusion_learning_rate = config.get("fusion_learning_rate")
        self.linear_learning_rate = config.get("linear_learning_rate")
        self.classifier_learning_rate = config.get("classifier_learning_rate")

        self.FREEZE_BERT = config.get("FREEZE_BERT")
        self.FREEZE_VGG = config.get("FREEZE_VGG")

        self.seed = config.get("seed")
        self.kernel_sizes = config.get("kernel_sizes")  # [3, 3, 3]
        self.num_channels = config.get("num_channels")  # [32, 64, 128]
        self.drop_and_BN = config.get(
            "drop_and_BN"
        )  # 'BN-drop', 'drop-BN', 'BN-only', 'drop-only', 'none'
        self.num_layers = config.get("num_layers")  # int, e.g, 1
        self.num_heads = config.get("num_heads")  # int, e.g, 8
        self.model_dim = config.get("model_dim")
        self.dropout = config.get("dropout")  # number, e.g. 0.5

        self.train_bs = config.get("train_bs")
        self.test_bs = config.get("test_bs")
        self.confidence = config.get("confidence")
        self.momentum = config.get("momentum")
        self.epochs = config.get("epochs")

        self.ablation = config.get("ablation")
        self.dataset_name = config.get("dataset_name")  # weibo, twitter
        self.optimizer_name = config.get("optimizer_name")  # SGD, Adam, AdamW

        self.learning_rate = config.get("learning_rate")  # number
        self.weight_decay = config.get("weight_decay")  # number
        self.max_sen_len = config.get("max_sen_len")  # int
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        self.config_check()

        self.model = self.get_model().to(self.device)
        self.init_network()

        self.train_loader, self.test_loader = self.get_dataloader()

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = self.get_optimizer()
        self.scheduler = self.get_scheduler()

    def handle_batch_input(self, train_data):
        bert_paras = ["text_input_ids", "token_type_ids", "attention_mask"]
        response_bert_paras = ["llm_text_input_ids", "llm_token_type_ids", "llm_attention_mask"]
        llm_context_bert_paras = ["llm_context_text_input_ids", "llm_context_token_type_ids",
                                  "llm_context_attention_mask", "llm_context_scores"]
        vgg_paras = ["image"]
        dct_paras = ["dct_img"]
        share_paras = ['label', 'post_id']
        parameters = {}
        involve = bert_paras + response_bert_paras + llm_context_bert_paras + vgg_paras + dct_paras
        involve += share_paras

        for para in involve:
            parameters[para] = train_data[para].to(self.device)
        return parameters

    def handle_model_input(self, parameters):
        outputs = self.model(parameters['text_input_ids'],
                             parameters['token_type_ids'],
                             parameters['attention_mask'],
                             parameters['llm_text_input_ids'],
                             parameters['llm_token_type_ids'],
                             parameters['llm_attention_mask'],
                             parameters['llm_context_text_input_ids'],
                             parameters['llm_context_token_type_ids'],
                             parameters['llm_context_attention_mask'],
                             parameters['llm_context_scores'],
                             parameters['image'],
                             parameters['dct_img'],
                             attn_mask=None)

        return outputs

    def train_one_time(self):
        # training
        loss_values, test_loss_values = [], []
        acc_values, test_acc_values = [], []
        test_precision_values = []
        test_recall_values = []
        test_f1_values = []

        for epoch_index, epoch in enumerate(range(self.epochs)):
            print('epoch:{}{}'.format(epoch_index, '-' * 20))

            self.model.train()

            train_batch_loss = []
            train_batch_acc = []
            for i, train_data in enumerate(self.train_loader):
                parameters = self.handle_batch_input(train_data)
                train_label = parameters['label']

                # Forward + Backward + Optimize
                self.model.zero_grad()
                outputs = self.handle_model_input(parameters)

                loss_input = outputs[0]

                loss = self.criterion(loss_input, train_label.to(torch.long))
                loss.backward(retain_graph=True)

                # Gradient cropping
                torch.nn.utils.clip_grad_norm_(
                    parameters=self.model.parameters(),
                    max_norm=self.max_grad_norm)

                train_label = train_label.cpu().detach().numpy().tolist()
                pred_input = torch.sigmoid(
                    outputs[1]).cpu().detach().numpy().tolist()  # output[1]

                print('pred_input:{}', pred_input)

                acc = self.flat_accuracy(pred_input, train_label)
                self.optimizer.step()
                self.scheduler.step()

                train_batch_loss.append(loss.detach().item())
                train_batch_acc.append(acc)

            # Store the loss value for plotting the learning curve.
            train_epoch_loss = sum(train_batch_loss) / len(self.train_loader)
            loss_values.append(train_epoch_loss)

            # Store the acc value
            train_epoch_acc = sum(train_batch_acc) / len(self.train_loader)
            acc_values.append(train_epoch_acc)

            self.model.eval()

            test_batch_loss = []
            test_batch_acc = []
            report_label = []
            report_predict = []

            best_test_acc = 0

            for i, test_data in enumerate(self.test_loader):
                parameters = self.handle_batch_input(test_data)
                test_label = parameters['label']

                with torch.no_grad():
                    outputs = self.handle_model_input(parameters)

                test_loss_input = outputs[0]
                test_loss = self.criterion(test_loss_input, test_label.to(torch.long))

                predict = torch.max(outputs[1].cpu().detach(), 1)[1]

                test_pred_input = torch.sigmoid(
                    outputs[1]).cpu().detach().numpy().tolist()  # output[1]
                test_label = test_label.cpu().detach().numpy().tolist()

                test_acc = self.flat_accuracy(test_pred_input, test_label)

                test_batch_loss.append(test_loss.detach().item())
                test_batch_acc.append(test_acc)

                for j in range(len(test_label)):
                    report_label.append(test_label[j])
                    report_predict.append(predict[j])

            test_epoch_loss = sum(test_batch_loss) / len(self.test_loader)
            test_epoch_acc = sum(test_batch_acc) / len(self.test_loader)

            # generate report
            report = classification_report(report_label, report_predict, output_dict=True)

            if test_epoch_acc > best_test_acc:
                best_test_acc = test_epoch_acc
                self.condition_save(epoch_index, test_epoch_acc, report)

            test_loss_values.append(test_epoch_loss)
            test_acc_values.append(test_epoch_acc)
            test_precision_values.append(float(report["macro avg"]["precision"]))
            test_recall_values.append(float(report["macro avg"]["recall"]))
            test_f1_values.append(float(report["macro avg"]["f1-score"]))

            self.print_result_table_handler(loss_values, acc_values, test_loss_values, test_acc_values,
                                            test_precision_values, test_recall_values, test_f1_values, report,
                                            print_type='tabel', table_type='pretty')


            # early_stopping
            self.early_stopping(test_epoch_acc, test_recall_values)

            if self.early_stopping.early_stop:
                break

        return np.max(test_acc_values)

    def print_result_table_handler(self, loss_values, acc_values,
                                   test_loss_values, test_acc_values,
                                   test_precision_values, test_recall_values,
                                   test_f1_values, report, print_type='tabel',
                                   table_type='pretty'):

        def trend(values_list):
            if len(values_list) == 1:
                diff_value = values_list[-1]
                return '↑ ({:+.6f})'.format(diff_value)
            else:
                diff_value = values_list[-1] - values_list[-2]
                if values_list[-1] > values_list[-2]:
                    return '↑ ({:+.6f})'.format(diff_value)
                elif values_list[-1] == values_list[-2]:
                    return '~'
                else:
                    return '↓ ({:+.6f})'.format(diff_value)

        if print_type == 'tabel':
            avg_table = [["train loss", loss_values[-1], trend(loss_values)],
                         ["train acc", acc_values[-1], trend(acc_values)],
                         ["test loss", test_loss_values[-1], trend(test_loss_values)],
                         ["test acc", test_acc_values[-1], trend(test_acc_values)],
                         ["test pre", test_precision_values[-1], trend(test_precision_values)],
                         ['test rec', test_recall_values[-1], trend(test_recall_values)],
                         ['test F1', test_f1_values[-1], trend(test_f1_values)]]

            avg_header = ['metric', 'value', 'trend']
            print((tabulate(avg_table, avg_header, floatfmt=".6f", tablefmt=table_type)))

            class_table = [['0', report["0"]["precision"], report["0"]["recall"], report["0"]["f1-score"],
                            '{}/{}'.format(report["0"]["support"], report['macro avg']["support"])],
                           ['1', report["1"]["precision"], report["1"]["recall"], report["1"]["f1-score"],
                            '{}/{}'.format(report["1"]["support"], report['macro avg']["support"])]]

            class_header = ['class', 'precision', 'recall', 'f1', 'support']
            print((tabulate(class_table, class_header, floatfmt=".6f", tablefmt=table_type)))
        else:
            print(("Average train loss: {}".format(loss_values[-1])))
            print(("Average train acc: {}".format(acc_values[-1])))
            print(("Average test loss: {}".format(test_loss_values[-1])))
            print(("Average test acc: {}".format(test_acc_values[-1])))
            print(report)

    def step(self):
        test_acc = self.train_one_time()
        return {"best_test_accuracy": test_acc}

    def save_model(self, folder_path, epoch_index, test_acc, report):
        root = self.save_root
        now = datetime.now()
        dt_string = now.strftime("%Y_%m_%d_%H_%M_%S")

        path = os.path.join(root, folder_path)

        if not os.path.exists(path):
            os.makedirs(path)

        save_name = "task_{}-epoch_{}-model_{}-date-{}-acc_{}-precision_{}-recall_{}-f1_{}.pth".format(
            self.dataset_name, epoch_index, self.ablation, dt_string, test_acc, report["macro avg"]["precision"],
            report["macro avg"]["recall"], report["macro avg"]["f1-score"])
        print("Saving model to {}, as {}".format(path, save_name))

        state = {
            "net": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "config": self.config,
        }
        torch.save(
            state,
            os.path.join(
                path,
                save_name,
            ),
        )

    def condition_save(self, epoch_index, test_epoch_acc, report):
        twitter_threshold = 0.89
        weibo_threshold = 0.89
        if self.dataset_name == 'twitter':
            if test_epoch_acc >= twitter_threshold:
                folder_path = 'model_save'
                self.save_model(folder_path, epoch_index, test_epoch_acc, report)
        elif self.dataset_name == 'weibo':
            if test_epoch_acc >= weibo_threshold:
                folder_path = 'model_save'
                self.save_model(folder_path, epoch_index, test_epoch_acc, report)

    def save_checkpoint(self, checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir, "model.pth")
        torch.save(self.model.state_dict(), checkpoint_path)
        return checkpoint_path

    def load_checkpoint(self, checkpoint_path):
        self.model.load_state_dict(torch.load(checkpoint_path))
