# -*- coding: utf-8 -*-
"""
    Collection of basic neural net models used in the OTDD experiments
"""

import os
import torch
import logging
from functools import partial
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models

import pdb

from .. import MODELS_DIR

logger = logging.getLogger(__name__)


MNIST_FLAT_DIM = 28 * 28

def reset_parameters(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()

# import torchvision.models as models
# resnet18 = models.resnet18()
# alexnet = models.alexnet()
# vgg16 = models.vgg16()
# squeezenet = models.squeezenet1_0()
# densenet = models.densenet161()
# inception = models.inception_v3()
# googlenet = models.googlenet()
# shufflenet = models.shufflenet_v2_x1_0()
# mobilenet = models.mobilenet_v2()
# resnext50_32x4d = models.resnext50_32x4d()
# wide_resnet50_2 = models.wide_resnet50_2()
# mnasnet = models.mnasnet1_0()

def model_factory(nclasses=10, dim=28, model_name='lenet', pretrained=False, aspartial=True):
    if pretrained and (nclasses != 1000):
        assert not aspartial, "If pretrained and nclasses!=1000, cannot return as partial, need to modify fc layer."
        # Pretrained models are trained on Imagenet, have 1000 top layer.
        logger.warning(f'Using pretrained model with dataset with {nclasses} != 1000 classes. Will replace with randomly init fc layer')
        replace_last = True
        true_nclasses = nclasses
        nclasses = 1000
    else:
        replace_last = False

    if model_name == 'lenet':
        model_class = partial(LeNet, input_size=dim, num_classes = nclasses, pretrained=pretrained)
    elif hasattr(torchvision.models, model_name):
        model_class = partial(getattr(torchvision.models, model_name), num_classes = nclasses, pretrained=pretrained)
    # elif model_name == 'alexnet':
    #     model_class = partial(models.alexnet, num_classes = nclasses, pretrained=pretrained)
    # elif model_name == 'vgg11':
    #     model_class = partial(models.vgg11, num_classes = nclasses, pretrained=pretrained)
    # elif model_name == 'vgg16':
    #     model_class = partial(models.vgg16, num_classes = nclasses, pretrained=pretrained)
    # elif model_name == 'densenet':
    #     model_class = partial(models.densenet, num_classes = nclasses, pretrained=pretrained)
    # elif model_name == 'resnet18':
    #     model_class = partial(models.resnet18, num_classes = nclasses, pretrained=pretrained)
    # elif model_name == 'resnet34':
    #     model_class = partial(models.resnet34, num_classes = nclasses, pretrained=pretrained)
    # elif model_name == 'resnet50':
    #     model_class = partial(models.resnet50, num_classes = nclasses, pretrained=pretrained)
    elif model_name == 'resnet': # If no depth prodived, default to resnet 18
        model_class = partial(models.resnet18, num_classes = nclasses, pretrained=pretrained)
    else:
        raise ValueError('Unrecognized model type')

    if aspartial:
        return model_class
    else:
        model = model_class()
        if replace_last:
            if hasattr(model, 'fc'):
                model.fc = torch.nn.Linear(model.fc.in_features, true_nclasses)
            elif hasattr(model, 'classifier') and (type(model.classifier) is list):
                model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, true_nclasses)
            elif hasattr(model, 'classifier') and (type(model.classifier) is nn.modules.linear.Linear):
                model.classifier = torch.nn.Linear(model.classifier.in_features, true_nclasses)
        return model

class LeNet(nn.Module):
    def __init__(self, pretrained=False, num_classes = 10, input_size=28, modeldir=None, **kwargs):
        super(LeNet, self).__init__()
        #assert not pretrained, "Pretrained LeNet not implemented yet"
        suffix = f'dim{input_size}_nc{num_classes}'
        if modeldir is None:
            self.model_path = os.path.join(MODELS_DIR, f'lenet_mnist_{suffix}.pt')
        else:
            self.model_path = os.path.join(modeldir, f'lenet_mnist_{suffix}.pt')

        assert input_size in [28,32], "Can only do LeNet on 28x28 or 32x32 for now."

        feat_dim = 16*5*5 if input_size == 32 else 16*4*4
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        if input_size == 32:
            self.conv1 = nn.Conv2d(1, 6, 3)
            self.conv2 = nn.Conv2d(6, 16, 3)
        elif input_size == 28:
            self.conv1 = nn.Conv2d(1, 6, 5)
            self.conv2 = nn.Conv2d(6, 16, 5)
        else:
            raise ValueError()

        self._init_classifier()

        if pretrained:
            state_dict = torch.load(self.model_path)
            self.load_state_dict(state_dict)

    def _init_classifier(self, num_classes=None):
        """ Useful for fine-tuning """
        num_classes = self.num_classes if num_classes is None else num_classes
        self.classifier = nn.Sequential(
            nn.Linear(self.feat_dim, 120),  # 6*6 from image dimension
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(84, num_classes)
        )

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        #return F.log_softmax(self.classifier(x), dim=1)
        return self.classifier(x)

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def save(self):
        state_dict = self.state_dict()
        torch.save(state_dict, self.model_path)

class MNIST_MLP(nn.Module):
    def __init__(
            self,
            input_dim=MNIST_FLAT_DIM,
            hidden_dim=98,
            output_dim=10,
            dropout=0.5,
    ):
        super(ClassifierModule, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.hidden  = nn.Linear(input_dim, hidden_dim)
        self.output  = nn.Linear(hidden_dim, output_dim)

    def forward(self, X, **kwargs):
        X = X.reshape(-1, self.hidden.in_features)
        X = F.relu(self.hidden(X))
        X = self.dropout(X)
        X = F.softmax(self.output(X), dim=-1)
        return X

class MNIST_CNN(nn.Module):
    def __init__(self, input_size=28, dropout=0.3, nclasses=10, pretrained=False, modeldir=None):
        super(MNIST_CNN, self).__init__()
        self.nclasses = nclasses
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv2_drop = nn.Dropout2d(p=dropout)
        self.fc1 = nn.Linear(1600, 100) # 1600 = number channels * width * height
        self.logit = nn.Linear(100, self.nclasses)
        self.fc1_drop = nn.Dropout(p=dropout)
        suffix = f'dim{input_size}_nc{nclasses}'
        if modeldir is None:
            self.model_path = os.path.join(MODELS_DIR, f'cnn_mnist_{suffix}.pt')
        else:
            self.model_path = os.path.join(modeldir, f'cnn_mnist_{suffix}.pt')
        if pretrained:
            state_dict = torch.load(self.model_path)
            self.load_state_dict(state_dict)

    def forward(self, x):
        x = torch.relu(F.max_pool2d(self.conv1(x), 2))
        x = torch.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        # flatten over channel, height and width = 1600
        x = x.view(-1, x.size(1) * x.size(2) * x.size(3))
        x = torch.relu(self.fc1_drop(self.fc1(x)))
        x = self.logit(x)
        x = F.log_softmax(x, dim=-1)
        return x

    def save(self):
        state_dict = self.state_dict()
        torch.save(state_dict, self.model_path)


class MLPClassifier(nn.Module):
    def __init__(
            self,
            input_size=None,
            hidden_size=400,
            num_classes=2,
            dropout=0.2,
            pretrained=False,
    ):
        super(MLPClassifier, self).__init__()
        self.num_classes = num_classes
        self.hidden_sizes = [hidden_size, int(hidden_size/2), int(hidden_size/4)]

        self.dropout = nn.Dropout(dropout)
        self.fc1     = nn.Linear(input_size, self.hidden_sizes[0])
        self.fc2     = nn.Linear(self.hidden_sizes[0], self.hidden_sizes[1])
        self.fc3     = nn.Linear(self.hidden_sizes[1], self.hidden_sizes[2])

        self._init_classifier()

    def _init_classifier(self, num_classes=None):
        num_classes = self.num_classes if num_classes is None else num_classes
        #self.classifier  = nn.Linear(self.hidden_sizes[-1], num_classes)
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_sizes[-1], 20),
            nn.ReLU(),
            nn.Linear(20, num_classes)
        )

    def forward(self, x, **kwargs):
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        x = self.classifier(x)
        return x

class BoWSentenceEmbedding():
    def __init__(self, vocab_size, embedding_dim, pretrained_vec, padding_idx=None, method = 'naive'):
        self.method = method
        if method == 'bag':
            self.emb = nn.EmbeddingBag.from_pretrained(pretrained_vec, padding_idx=padding_idx)
        else:
            self.emb = nn.Embedding.from_pretrained(pretrained_vec)

    def __call__(self, x):
        if self.method == 'bag':
            return self.emb(x)
        else:
            return self.emb(x).mean(dim=1)

class MLPPushforward(nn.Module):
    def __init__(self, input_size=2, n_layers=4, hidden_dim=100, nonlinearity='relu', dropout=0.0, **kwargs):
        ''' n_layers: number of hidden layers '''
        super(MLPPushforward, self).__init__()
        d = input_size
        _nonlins = {'relu': nn.ReLU, 'sigmoid': nn.Sigmoid, 'elu': nn.ELU}

        _layers = []

        dims = [d]
        if hidden_dim is None:
            # TODO
            #>>> np.arange(2, 2*(nl-1), 2)
            #[2*d, 4*d, ..., 4*d, 2*d]
            pdb.set_trace()
        elif type(hidden_dim) is int:
            dims += n_layers*[hidden_dim]
        else:
            dims += hidden_dim
        dims += [d]

        for i, d_in in enumerate(dims[:-1]):
            d_out = dims[i+ 1]
            _layers.append(nn.Linear(d_in, d_out))
            if i < len(dims) - 2:
                _layers.append(_nonlins[nonlinearity]())
                _layers.append(nn.Dropout(dropout))


        # for i in range(n_layers):
        #     _d_out = hidden_dim if hidden_dim else 2*_d_in
        #
        #     _layers.append(_nonlins[nonlinearity]())
        #     _layers.append(nn.Dropout(dropout))
        #     _d_in = _d_out
        # for i in range(n_layers):
        #     if i < n_layers - 1:
        #         _d_out = hidden_dim if hidden_dim else int(0.5*_d)
        #         _layers.append(nn.Linear(_d_in, _d_out))
        #         _layers.append(_nonlins[nonlinearity]())
        #         _layers.append(nn.Dropout(dropout))
        #     else:
        #         _layers.append(nn.Linear(_d_in, d))
        #
        #     _d_in = _d_out

        self.mapping = nn.Sequential(*_layers)

    def forward(self, x):
        return self.mapping(x)

    def reset_parameters(self):
        self.mapping.apply(reset_parameters)


class ConvPushforward(nn.Module):
    def __init__(self, input_size=28, channels = 1, nlayers_conv = 2, nlayers_mlp = 3, **kwargs):
        super(ConvPushforward, self).__init__()
        self.input_size = input_size
        self.channels = channels
        if input_size == 32:
            self.upconv1 = nn.Conv2d(1, 6, 3)
            self.upconv2 = nn.Conv2d(6, 16, 3)
            feat_dim = 16*5*5
            ## decoder layers ##
            self.dnconv1 = nn.ConvTranspose2d(4, 16, 2, stride=2)
            self.dnconv2 = nn.ConvTranspose2d(16, 1, 2, stride=2)
        elif input_size == 28:
            self.upconv1 = nn.Conv2d(1, 6, 5)
            self.upconv2 = nn.Conv2d(6, 16, 5)
            feat_dim = 16*4*4
            self.dnconv1 = nn.ConvTranspose2d(16, 6, 5)
            self.dnconv2 = nn.ConvTranspose2d(6, 1, 5)
        else:
            raise NotImplemented("Can only do LeNet on 28x28 or 32x32 for now.")
        self.feat_dim = feat_dim

        self.mlp = MLPPushforward(input_size = feat_dim, layers = nlayers_mlp)

    def forward(self, x):
        _orig_shape = x.shape
        x = x.reshape(-1, self.channels, self.input_size, self.input_size)
        x, idx1 = F.max_pool2d(F.relu(self.upconv1(x)), 2, return_indices=True)
        x, idx2 = F.max_pool2d(F.relu(self.upconv2(x)), 2, return_indices=True)
        _nonflat_shape = x.shape
        x = x.view(-1, self.num_flat_features(x))
        x = self.mlp(x).reshape(_nonflat_shape)
        x = F.relu(self.dnconv1(F.max_unpool2d(x, idx2, kernel_size=2)))
        x = torch.tanh(self.dnconv2(F.max_unpool2d(x, idx1, kernel_size=2)))
        return x.reshape(_orig_shape)

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def reset_parameters(self):
        for name, module in self.named_children():
            #print('resetting ', name)
            module.reset_parameters()


class ConvPushforward2(nn.Module):
    def __init__(self, input_size=28, channels = 1, nlayers_conv = 2, nlayers_mlp = 3, **kwargs):
        super(ConvPushforward2, self).__init__()
        self.input_size = input_size
        self.channels = channels
        if input_size == 32:
            self.upconv1 = nn.Conv2d(1, 6, 3)
            self.upconv2 = nn.Conv2d(6, 16, 3)
            feat_dim = 16*5*5
            ## decoder layers ##
            self.dnconv1 = nn.ConvTranspose2d(4, 16, 2, stride=2)
            self.dnconv2 = nn.ConvTranspose2d(16, 1, 2, stride=2)
        elif input_size == 28:
            self.upconv1 = nn.Conv2d(1, 16, 3, stride=3, padding=1)  # b, 16, 10, 10
            self.upconv2 = nn.Conv2d(16, 8, 3, stride=2, padding=1)  # b, 8, 3, 3
            feat_dim = 8*2*2
            self.dnconv1 = nn.ConvTranspose2d(8, 16, 3, stride=2)  # b, 16, 5, 5
            self.dnconv2 = nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1)  # b, 8, 15, 15
            self.dnconv3 = nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1)  # b, 1, 28, 28
        else:
            raise NotImplemented("Can only do LeNet on 28x28 or 32x32 for now.")
        self.feat_dim = feat_dim

        self.mlp = MLPPushforward(input_size = feat_dim, layers = nlayers_mlp)

    def forward(self, x):
        x = x.reshape(-1, self.channels, self.input_size, self.input_size)
        #pdb.set_trace()
        x = F.max_pool2d(F.relu(self.upconv1(x)), 2, stride=2)
        x = F.max_pool2d(F.relu(self.upconv2(x)), 2, stride=1)
        _nonflat_shape = x.shape
        #pdb.set_trace()
        x = x.view(-1, self.num_flat_features(x))
        x = self.mlp(x).reshape(_nonflat_shape)
        x = F.relu(self.dnconv1(x))
        x = F.relu(self.dnconv2(x))
        x = torch.tanh(self.dnconv3(x))
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def reset_parameters(self):
        for name, module in T.named_children():
            print('resetting ', name)
            module.reset_parameters()


class ConvPushforward3(nn.Module):
    def __init__(self, input_size=28, channels = 1, nlayers_conv = 2, nlayers_mlp = 3, **kwargs):
        super(ConvPushforward3, self).__init__()
        self.input_size = input_size
        self.channels = channels

        self.upconv1 = nn.Conv2d(1, 128, 3, 1, 2, dilation=2)
        self.upconv2 = nn.Conv2d(128, 128, 3, 1, 2)
        self.upconv3 = nn.Conv2d(128, 256, 3, 1, 2)
        self.upconv4 = nn.Conv2d(256, 256, 3, 1, 2)
        self.upconv5 = nn.Conv2d(128, 128, 3, 1, 2)
        self.upconv6 = nn.Conv2d(128, 128, 3, 1, 2)
        self.upconv7 = nn.Conv2d(128, 128, 3, 1, 2)
        self.upconv8 = nn.Conv2d(128, 128, 3, 1, 2)

        # self.dnconv8 = nn.ConvTranspose2d(256, 256, 3, 1, 2)
        # self.dnconv7 = nn.ConvTranspose2d(256, 128, 3, 1, 2)
        # self.dnconv6 = nn.ConvTranspose2d(128, 128, 3, 1, 2)
        # self.dnconv5 = nn.ConvTranspose2d(128, 128, 3, 1, 2)
        self.dnconv4 = nn.ConvTranspose2d(256, 256, 3, 1, 2)
        self.dnconv3 = nn.ConvTranspose2d(256, 128, 3, 1, 2)
        self.dnconv2 = nn.ConvTranspose2d(128, 128, 3, 1, 2)
        self.dnconv1 = nn.ConvTranspose2d(128, 1, 3, 1, 2, dilation=2)

        self.maxpool1 = nn.MaxPool2d(2, return_indices=True)
        self.maxpool2 = nn.MaxPool2d(2, return_indices=True)
        self.maxpool3 = nn.MaxPool2d(2, return_indices=True)
        self.maxunpool1 = nn.MaxUnpool2d(2)
        self.maxunpool2 = nn.MaxUnpool2d(2)

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.relu4 = nn.ReLU()
        self.relu5 = nn.ReLU()
        self.relu6 = nn.ReLU()
        self.relu7 = nn.ReLU()
        self.relu8 = nn.ReLU()
        self.derelu1 = nn.ReLU()
        self.derelu2 = nn.ReLU()
        self.derelu3 = nn.ReLU()
        self.derelu4 = nn.ReLU()
        self.derelu5 = nn.ReLU()
        self.derelu6 = nn.ReLU()
        self.derelu7 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(16)
        self.bn4 = nn.BatchNorm2d(1)
        # self.feat_dim = feat_dim

        # self.mlp = MLPPushforward(input_size = feat_dim, layers = nlayers_mlp)

    def forward(self, x):
        # _orig_shape = x.shape
        # x = x.reshape(-1, self.channels, self.input_size, self.input_size)
        # import pdb; pdb.set_trace()
        x = self.upconv1(x)
        x = self.relu1(x)
        # x, idx1 = self.maxpool1(x)

        x = self.upconv2(x)
        x = self.relu2(x)
        # x, idx2 = self.maxpool2(x)

        x = self.upconv3(x)
        x = self.relu3(x)
        # x, idx3 = self.maxpool3(x)

        x = self.upconv4(x)
        x = self.relu4(x)

        x = self.derelu4(x)
        x = self.dnconv4(x)

        x = self.derelu3(x)
        x = self.dnconv3(x)

        # x = self.maxunpool2(x, idx2)
        x = self.derelu2(x)
        x = self.dnconv2(x)

        # x = self.maxunpool1(x, idx1)
        x = self.derelu1(x)
        x = self.dnconv1(x)

        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def reset_parameters(self):
        for name, module in self.named_children():
            #print('resetting ', name)
            try:
                module.reset_parameters()
            except:
                pass
