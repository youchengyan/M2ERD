# coding=utf-8
import numpy as np
import torch
import torch.nn as nn
from transformers import BertConfig, BertModel
import torch.nn.functional as F
from torchvision import transforms
import torchvision

from DctCNN import DctStem, ConvBNRelu2d, DctInceptionBlock
from multimodal_attention import multimodal_fusion_layer
from self_attention import self_MultiHeadAttention


class vgg(nn.Module):
    """
    obtain visual feature
    """

    def __init__(self, model_dim, pthfile):
        super(vgg, self).__init__()
        self.model_dim = model_dim
        self.pthfile = pthfile

        # image
        vgg_19 = torchvision.models.vgg19(pretrained=False)
        vgg_19.load_state_dict(torch.load(self.pthfile))

        self.feature = vgg_19.features
        self.classifier = nn.Sequential(*list(vgg_19.classifier.children())[:-3])
        pretrained_dict = vgg_19.state_dict()
        model_dict = self.classifier.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}  # delect the last layer
        model_dict.update(pretrained_dict)  # update
        self.classifier.load_state_dict(model_dict)  # load the new parameter

    def forward(self, img):
        # image
        # image = self.vgg(img) #[batch, num_ftrs]
        img = self.feature(img)
        img = img.view(img.size(0), -1)
        image = self.classifier(img)

        return image


class DctCNN(nn.Module):
    def __init__(self,
                 model_dim,
                 dropout,
                 kernel_sizes,
                 num_channels,
                 in_channel=128,
                 branch1_channels=[64],
                 branch2_channels=[48, 64],
                 branch3_channels=[64, 96, 96],
                 branch4_channels=[32],
                 out_channels=64):
        super(DctCNN, self).__init__()

        self.stem = DctStem(kernel_sizes, num_channels)

        self.InceptionBlock = DctInceptionBlock(
            in_channel,
            branch1_channels,
            branch2_channels,
            branch3_channels,
            branch4_channels,
        )

        self.maxPool = nn.MaxPool2d((1, 122))

        self.dropout = nn.Dropout(dropout)

        self.conv = ConvBNRelu2d(branch1_channels[-1] + branch2_channels[-1] +
                                 branch3_channels[-1] + branch4_channels[-1],
                                 out_channels,
                                 kernel_size=1)

    def forward(self, dct_img):
        dct_f = self.stem(dct_img)
        x = self.InceptionBlock(dct_f)
        x = self.maxPool(x)
        x = x.permute(0, 2, 1, 3)
        x = self.conv(x)
        x = x.permute(0, 2, 1, 3)
        x = x.squeeze(-1)

        x = x.reshape(-1, 4096)

        return x


class NetShareFusion(nn.Module):
    def __init__(self,
                 CASED,
                 pthfile,
                 kernel_sizes,
                 num_channels,
                 model_dim,
                 drop_and_BN,
                 bert_dim=768,
                 img_size=250,
                 num_labels=2,
                 num_layers=1,
                 num_heads=8,
                 ffn_dim=2048,
                 dropout=0.5,
                 confidence=0.9):

        super(NetShareFusion, self).__init__()

        self.CASED = CASED
        self.model_dim = model_dim
        self.pthfile = pthfile
        self.drop_and_BN = drop_and_BN
        self.num_heads = num_heads
        self.dropout_num = dropout
        self.confidence = confidence
        # text
        self.config = BertConfig.from_pretrained(self.CASED)

        self.bert = BertModel.from_pretrained(self.CASED, config=self.config)
        self.linear_text = nn.Linear(bert_dim, model_dim)
        self.bn_text = nn.BatchNorm1d(model_dim)

        self.dropout = nn.Dropout(dropout)

        # image
        self.vgg = vgg(model_dim, pthfile)
        self.linear_image = nn.Linear(4096, model_dim)
        self.bn_vgg = nn.BatchNorm1d(model_dim)

        # dct_image
        self.dct_img = DctCNN(model_dim,
                              dropout,
                              kernel_sizes,
                              num_channels,
                              in_channel=128,
                              branch1_channels=[64],
                              branch2_channels=[48, 64],
                              branch3_channels=[64, 96, 96],
                              branch4_channels=[32],
                              out_channels=64)
        self.linear_dct = nn.Linear(4096, model_dim)
        self.bn_dct = nn.BatchNorm1d(model_dim)

        # multimodal fusion
        self.fusion_layers = nn.ModuleList([
            multimodal_fusion_layer(model_dim, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])
        self.text_self_attention = self_MultiHeadAttention(self.model_dim, self.num_heads, self.dropout_num)

        # classifier
        self.linear1 = nn.Linear(model_dim, 35)
        self.bn_1 = nn.BatchNorm1d(35)
        self.linear2 = nn.Linear(35, num_labels)
        self.softmax = nn.Softmax(dim=1)

    def drop_BN_layer(self, x, part='dct'):
        if part == 'dct':
            bn = self.bn_dct
        elif part == 'vgg':
            bn = self.bn_vgg
        elif part == 'bert':
            bn = self.bn_text

        if self.drop_and_BN == 'drop-BN':
            x = self.dropout(x)
            x = bn(x)
        elif self.drop_and_BN == 'BN-drop':
            x = bn(x)
            x = self.dropout(x)
        elif self.drop_and_BN == 'drop-only':
            x = self.dropout(x)
        elif self.drop_and_BN == 'BN-only':
            x = bn(x)
        elif self.drop_and_BN == 'none':
            pass

        return x

    def forward(self, text_input_ids, token_type_ids, attention_mask, llm_text_input_ids, llm_token_type_ids,
                llm_attention_mask, llm_context_text_input_ids, llm_context_token_type_ids, llm_context_attention_mask,
                llm_context_scores, image, dct_img, attn_mask):

        # textual feature
        bert_output = self.bert(input_ids=text_input_ids,
                                token_type_ids=token_type_ids,
                                attention_mask=attention_mask)
        text_output = bert_output[1]  # the representation of the whole sentence
        # print('bert_output:{}, shape:{}'.format(text_output, text_output.shape)) [batch_size, 768]
        text_output = F.relu(self.linear_text(text_output))
        text_output = self.drop_BN_layer(text_output, part='bert')

        # entire response
        llm_bert_output = self.bert(input_ids=llm_text_input_ids,
                                    token_type_ids=llm_token_type_ids,
                                    attention_mask=llm_attention_mask)
        llm_response_output = llm_bert_output[1]
        llm_response_output = F.relu(self.linear_text(llm_response_output))
        llm_response_output = self.drop_BN_layer(llm_response_output, part='bert')

        # seperate response
        seq_output_list = []
        output1_tensor = torch.zeros(llm_context_text_input_ids.shape[0], 2).cuda()
        seq_output_bert_list = []
        for i in range(llm_context_text_input_ids.shape[1]):
            # print('score:{}', llm_context_scores[:, i]) [batch_size]
            seq_input_ids = llm_context_text_input_ids[:, i, :]
            seq_token_type_ids = llm_context_token_type_ids[:, i, :]
            seq_attention_mask = llm_context_attention_mask[:, i, :]
            seq_bert_output = self.bert(input_ids=seq_input_ids,
                                        token_type_ids=seq_token_type_ids,
                                        attention_mask=seq_attention_mask)
            seq_output = seq_bert_output[1]
            # print('seq_output_shape:{}', seq_output.shape) # [2, bert_dim]
            seq_output = F.relu(self.linear_text(seq_output))
            seq_output = self.drop_BN_layer(seq_output, part='bert')
            seq_output_bert_list.append(seq_output)
            output1 = F.relu(self.linear1(seq_output))
            output1 = self.dropout(output1)
            output1 = self.linear2(output1)
            output1_tensor += output1
            # print('output1_shape:{}', output1.shape)  [batch_size, 2]
            y_pred_prob1 = self.softmax(output1)
            # print('y_pred_prob1:{}', y_pred_prob1) [batch_size, 2]
            llm_context_pred_results_list = []
            for j in range(llm_context_text_input_ids.shape[0]):
                llm_context_pred_results = llm_context_scores[:, i][j] * y_pred_prob1[j]
                llm_context_pred_results_list.append(llm_context_pred_results)

            llm_context_pred_tensor = torch.stack(llm_context_pred_results_list, dim=0)
            seq_output_list.append(llm_context_pred_tensor)

        sum_tensor = torch.zeros_like(seq_output_list[0])
        for tensor in seq_output_list:
            sum_tensor += tensor

        llm_context_pred = sum_tensor

        # visual feature
        img_output = self.vgg(image)
        img_output = F.relu(self.linear_image(img_output))
        img_output = self.drop_BN_layer(img_output, part='vgg')

        # dct_feature
        dct_out = self.dct_img(dct_img)
        dct_out = F.relu(self.linear_dct(dct_out))
        dct_out = self.drop_BN_layer(dct_out, part='dct')

        # fusion
        text_output_attention = self.text_self_attention(text_output, text_output)

        llm_all_response_attention = self.text_self_attention(llm_response_output, llm_response_output)
        dct_out_attention = self.text_self_attention(dct_out, dct_out)
        img_output_attention = self.text_self_attention(img_output, img_output)
        text_output_llm_response = self.text_self_attention(text_output_attention, llm_all_response_attention,
                                                            attn_mask=None, is_aux=True)
        img_llm_response = self.text_self_attention(img_output_attention, llm_all_response_attention, attn_mask=None,
                                                    is_aux=True)

        for fusion_layer in self.fusion_layers:
            img_dct_output = fusion_layer(img_output_attention, dct_out_attention, attn_mask)
        for fusion_layer in self.fusion_layers:
            f_v = fusion_layer(img_dct_output, img_llm_response, attn_mask)
        for fusion_layer in self.fusion_layers:
            f_t = fusion_layer(text_output_attention, text_output_llm_response, attn_mask)


        output = self.text_self_attention(f_t, f_v, attn_mask=None, is_aux=True)
        output = F.relu(self.linear1(output))
        output = self.dropout(output)
        output = self.linear2(output)  
        # print('output_size:{}'.format(output.shape)) [batch_size, 2]
        y_pred_prob = self.softmax(output)
        output += (1 - self.confidence) * output1_tensor
        y_pred_prob = self.confidence * y_pred_prob + (1 - self.confidence) * llm_context_pred

        return output, y_pred_prob
