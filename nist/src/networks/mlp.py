import torch
from torch import nn


def get_activation(activation):
    if activation == "relu":
        return nn.ReLU(True)
    elif activation == "celu":
        return nn.CELU()
    elif activation == "tanh":
        return nn.Tanh()
    elif activation == "softsign":  # map to [-1,1]
        return nn.Softsign()
    elif activation == "Prelu":
        return nn.PReLU()
    elif activation == "sigmoid":
        return nn.Sigmoid()
    else:
        raise NotImplementedError("activation [%s] is not found" % activation)


# pylint: disable=too-many-arguments,too-many-instance-attributes


class FullyConnected(nn.Module):
    def __init__(
        self,
        input_dim=785,
        output_dim=1,
        hidden_dim=1024,
        num_layer=1,
        activation="Prelu",
        final_actv=None,
        dropout=0,
        batch_nml=False,
        res=0,
    ):
        super().__init__()
        self.full_activ = final_actv is not None
        self.dropout = dropout
        self.batch_nml = batch_nml
        self.res = res

        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer1_activ = get_activation(activation)
        self.linearblock = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layer)]
        )
        self.atvt_list = nn.ModuleList(
            [get_activation(activation) for _ in range(num_layer)]
        )

        if batch_nml:
            self.batchnormal = nn.ModuleList(
                [nn.BatchNorm1d(hidden_dim) for _ in range(num_layer)]
            )
        if dropout > 0:
            self.dropout_list = nn.ModuleList(
                [nn.Dropout(dropout) for _ in range(num_layer)]
            )

        self.last_layer = nn.Linear(hidden_dim, output_dim)
        if self.full_activ:
            self.last_layer_activ = get_activation(final_actv)

    def forward(self, input_tensor):
        x = self.layer1_activ(self.layer1(input_tensor))

        for i, layer in enumerate(self.linearblock):
            x = layer(x)
            if self.batch_nml:
                x = self.batchnormal[i](x)
            if self.dropout > 0:
                x = self.dropout_list[i](x)
            x = self.atvt_list[i](x)

        x = self.last_layer(x)
        if self.full_activ:
            x = self.last_layer_activ(x)
        if self.res:
            x = x + input_tensor
        return x


class ResFeatNet(nn.Module):
    def __init__(
        self,
        feat_dim=2,
        label_emb_dim=4,
        output_dim=1,
        hidden_dim=1024,
        num_layer=1,
        activation="Prelu",
    ):
        super().__init__()
        self.layer1 = nn.Linear(feat_dim, hidden_dim)
        self.layer1_activ = get_activation(activation)
        self.linearblock = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layer)]
        )
        self.weight_label = nn.ModuleList(
            [nn.Linear(label_emb_dim, hidden_dim) for _ in range(num_layer)]
        )
        self.resblock = nn.ModuleList(
            [nn.Linear(feat_dim, hidden_dim) for _ in range(num_layer)]
        )
        self.atvt_list = nn.ModuleList(
            [get_activation(activation) for _ in range(num_layer)]
        )
        self.last_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, feat, label):
        input_feat = feat
        x = self.layer1_activ(self.layer1(feat))
        for i, layer in enumerate(self.linearblock):
            x = self.atvt_list[i](
                layer(x) + self.weight_label[i](label) + self.resblock[i](input_feat)
            )

        x = self.last_layer(x)
        return x


class MLPClassifier(nn.Module):
    def __init__(
        self,
        feat_dim=2,
        num_class=4,
        hidden_dim=1024,
        num_layer=1,
    ):
        super().__init__()
        self.classifier = FullyConnected(feat_dim, num_class, hidden_dim, num_layer)
        self.num_class = num_class
        self.softmax = nn.Softmax(dim=1)

    def forward(self, feature, source_label):
        del source_label
        class_output = self.classifier(feature)
        label_probs = self.softmax(class_output)
        return label_probs


class ResClassifier(nn.Module):
    def __init__(
        self,
        feat_dim=2,
        num_class=4,
        hidden_dim=1024,
        num_layer=1,
        fourier=True,
        fourier_freq=1.0,
    ):
        super().__init__()
        self.fourier = fourier
        if fourier:
            ff_dim = 64
            w_gauss = torch.randn([feat_dim, ff_dim])
            self.register_buffer("weight_ff", w_gauss * fourier_freq)
            feat_dim = feat_dim + ff_dim * 2

        self.classifier = ResFeatNet(
            feat_dim, num_class, num_class, hidden_dim, num_layer
        )
        self.num_class = num_class
        # use embedding to map the label data
        self.emb = nn.Embedding(num_embeddings=num_class, embedding_dim=num_class)
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, feature, source_label):
        # source_label: (N,)
        if self.fourier:
            x_proj = feature @ self.weight_ff
            ff_feature = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], axis=-1)
            feature = torch.cat([feature, ff_feature], axis=-1)
        label_embedding = self.emb(source_label.long())
        # label_embedding = torch.zeros([source_label.shape[0], self.num_class]).to(feature.device)
        class_output = self.classifier(feature, label_embedding)
        # label_probs = self.softmax(class_output)
        return class_output


#! Feature generator use residual-typed embedding of labels


class ResFeatureGenerator(nn.Module):
    def __init__(
        self,
        feat_dim=2,
        num_classes=4,
        hidden_dim=1024,
        num_layer=1,
    ):
        super().__init__()
        self.feat_generator = ResFeatNet(
            feat_dim, hidden_dim, feat_dim, hidden_dim, num_layer
        )
        # use embedding to map the label data
        self.emb = nn.Embedding(num_embeddings=num_classes, embedding_dim=hidden_dim)

    def forward(self, feat, source_label):
        label_embedding = self.emb(source_label.long())
        output_feat = self.feat_generator(feat, label_embedding)
        return output_feat


#! Discriminator uses simple concatenation of labels


class SimpleDiscriminator(nn.Module):
    def __init__(self, feat_dim=2, num_class=4, hidden_dim=1024, num_layer=1):
        super().__init__()
        self.mlp = FullyConnected(feat_dim + num_class, 1, hidden_dim, num_layer)

    def forward(self, feat, label_probs):
        feat_logit = torch.cat([feat, label_probs], axis=1)
        return self.mlp(feat_logit)


#! Discriminator uses residual-typed label logits


class ResDiscriminator(nn.Module):
    def __init__(self, feat_dim=2, num_class=4, hidden_dim=1024, num_layer=1):
        super().__init__()
        self.disc = ResFeatNet(feat_dim, num_class, 1, hidden_dim, num_layer)
        # self.linearblock_feat = nn.Linear(num_class, hidden_dim)

    def forward(self, feat, label_probs):
        # feat_emb = self.linearblock_feat(label_probs.float())
        return self.disc(feat, label_probs)
