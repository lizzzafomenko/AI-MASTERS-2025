import torch.nn as nn

# source - https://github.com/GitYCC/crnn-pytorch/blob/master/src/model.py
# pre-trained data is used from checkpoint from here, except last FC

import torch
import torch.nn as nn
import torch.nn.functional as F



CHARS = "0123456789ABEKMHOPCTYX"
num_classes = len(CHARS) + 1


class CRNN(nn.Module):
    def __init__(self, img_channel=1, img_height=32, img_width=160,
                 map_to_seq_hidden=64, rnn_hidden=256, num_class=num_classes, leaky_relu=False, dropout_rate=0.2):
        super(CRNN, self).__init__()
        
        self.cnn, (output_channel, output_height, output_width) = \
            self._cnn_backbone(img_channel, img_height, img_width, leaky_relu, dropout_rate)
        
        self.map_to_seq = nn.Linear(output_channel * output_height, map_to_seq_hidden)
        
        self.rnn1 = nn.LSTM(map_to_seq_hidden, rnn_hidden, bidirectional=True, dropout=0.3)
        self.rnn2 = nn.LSTM(2 * rnn_hidden, rnn_hidden, bidirectional=True, dropout=0.3)
        
        self.dense =nn.Linear(2 * rnn_hidden, num_class)
    
    def _cnn_backbone(self, img_channel, img_height, img_width, leaky_relu, dropout_rate):
        
        channels = [img_channel, 64, 128, 256, 256, 512, 512, 512]
        kernel_sizes = [3,3,3,3,3,3,2]
        strides = [1]*7
        paddings = [1,1,1,1,1,1,0]
        
        cnn = nn.Sequential()
        def conv_block(i, batch_norm=False, dropout=False):
            in_ch = channels[i]
            out_ch = channels[i+1]
            cnn.add_module(f'conv{i}', nn.Conv2d(in_ch, out_ch, kernel_sizes[i], strides[i], paddings[i]))
            if batch_norm:
                cnn.add_module(f'batchnorm{i}', nn.BatchNorm2d(out_ch))
            relu = nn.LeakyReLU(0.2, inplace=True) if leaky_relu else nn.ReLU(inplace=True)
            cnn.add_module(f'relu{i}', relu)
            if dropout:
                cnn.add_module(f'dropout{i}', nn.Dropout2d(dropout_rate))
        
        # Convolution + pooling layers
        conv_block(0)
        cnn.add_module('pool0', nn.MaxPool2d(2,2))       # H/2, W/2
        conv_block(1)
        cnn.add_module('pool1', nn.MaxPool2d(2,2))       # H/4, W/4
        conv_block(2)
        conv_block(3, batch_norm=True)
        cnn.add_module('pool2', nn.MaxPool2d((2,1)))     # H/8, W/4
        conv_block(4, batch_norm=True)
        conv_block(5, batch_norm=True, dropout=True)
        cnn.add_module('pool3', nn.MaxPool2d((2,1)))     # H/16, W/4
        conv_block(6, dropout=True)
        
        output_channel = channels[-1]
        output_height = img_height // 16 - 1
        output_width = img_width // 4 - 1
        
        return cnn, (output_channel, output_height, output_width)
    
    def forward(self, x):
        
        conv = self.cnn(x)
        batch, channel, height, width = conv.size()
        
        conv = conv.view(batch, channel * height, width)
        conv = conv.permute(2,0,1)  # (seq_len=width, batch, features)
        seq = self.map_to_seq(conv)
        
        recurrent, _ = self.rnn1(seq)
        recurrent, _ = self.rnn2(recurrent)
        
        output = self.dense(recurrent)
        return F.log_softmax(output, dim=2)  # log_probs for CTC




import torch.nn as nn
import torch.nn.functional as F

class CRNN_WithComplexityHead(nn.Module):
    def __init__(self, base_crnn, cnn_out_ch=512, cnn_out_h=1, complexity_hidden=128):
        super().__init__()
        self.crnn = base_crnn  # pre-trained CRNN
        feature_dim = cnn_out_ch * cnn_out_h
        
        # complexity head
        self.complexity_head = nn.Sequential(
            nn.Linear(feature_dim, complexity_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(complexity_hidden, 2)  # binary: simple vs complex
        )
    
    def forward(self, x):
    # CNN features
        conv = self.crnn.cnn(x)
        batch, channel, height, width = conv.size()
    
    # Global avg pooling for complexity head
        gap_feat = conv.mean(dim=[2,3])  # batch x channel
        complexity_logits = self.complexity_head(gap_feat)
    
    # Prepare sequence for CRNN
        conv_seq = conv.view(batch, channel * height, width)
        conv_seq = conv_seq.permute(2,0,1)  # seq_len, batch, features
        seq = self.crnn.map_to_seq(conv_seq)
        recurrent, _ = self.crnn.rnn1(seq)
        recurrent, _ = self.crnn.rnn2(recurrent)
        logits = self.crnn.dense(recurrent)  # CTC output
    
        return logits, complexity_logits
