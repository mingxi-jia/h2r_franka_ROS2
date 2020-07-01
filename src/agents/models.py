import numpy as np
from scipy import ndimage
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


class FCN(nn.Module):
    def __init__(self, n_input_channel=1, n_primitives=1):
        super(FCN, self).__init__()

        self.conv_net = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(n_input_channel, 32, kernel_size=7, stride=2)),
            ('bn1', nn.BatchNorm2d(32)),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv2', nn.Conv2d(32, 64, kernel_size=5, stride=1)),
            ('bn2', nn.BatchNorm2d(64)),
            ('relu2', nn.ReLU(inplace=True)),
            ('conv3', nn.Conv2d(64, 128, kernel_size=3, stride=1)),
            ('bn3', nn.BatchNorm2d(128)),
            ('relu3', nn.ReLU(inplace=True)),
            ('conv4', nn.Conv2d(128, 64, kernel_size=1, stride=1, bias=False)),
            ('bn4', nn.BatchNorm2d(64)),
            ('relu4', nn.ReLU(inplace=True)),
        ]))

        self.q_values = nn.Conv2d(64, n_primitives, kernel_size=1, stride=1, bias=False)

        for m in self.named_modules():
            if isinstance(m[1], nn.Conv2d):
                # nn.init.kaiming_normal_(m[1].weight.data)
                nn.init.xavier_normal_(m[1].weight.data)
            elif isinstance(m[1], nn.BatchNorm2d):
                m[1].weight.data.fill_(1)
                m[1].bias.data.zero_()

        self.mask = None

    def setCandidatePos(self, candidate_pos, input_size=(250, 250)):
        if candidate_pos is None:
            self.mask = None
        else:
            self.mask = torch.zeros(input_size)
            index = np.array(np.meshgrid(candidate_pos[0], candidate_pos[1])).T.reshape(-1, 2)
            index = torch.tensor(index).long()
            for idx in index:
                self.mask[idx[0], idx[1]] = 1

    def forward(self, x):
        # x = x.permute(0, 3, 1, 2)
        feature_map = self.conv_net(x)
        q_values = self.q_values(feature_map)
        deconv_q_values = F.interpolate(q_values, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        if self.mask is not None:
            mask = self.mask.expand(deconv_q_values.size()).bool()
            mask = 1-mask
            deconv_q_values[mask] = -100
        return deconv_q_values


class FCNSmall(nn.Module):
    def __init__(self, n_input_channel=1, n_primitives=1):
        super(FCNSmall, self).__init__()

        self.conv_net = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(n_input_channel, 32, kernel_size=5, stride=1)),
            ('bn1', nn.BatchNorm2d(32)),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv2', nn.Conv2d(32, 64, kernel_size=3, stride=1)),
            ('bn2', nn.BatchNorm2d(64)),
            ('relu2', nn.ReLU(inplace=True)),
            ('conv3', nn.Conv2d(64, 128, kernel_size=3, stride=1)),
            ('bn3', nn.BatchNorm2d(128)),
            ('relu3', nn.ReLU(inplace=True)),
            ('conv4', nn.Conv2d(128, 64, kernel_size=1, stride=1, bias=False)),
            ('bn4', nn.BatchNorm2d(64)),
            ('relu4', nn.ReLU(inplace=True)),
        ]))

        self.q_values = nn.Conv2d(64, n_primitives, kernel_size=1, stride=1, bias=False)

        self.mask = None

        for m in self.named_modules():
            if isinstance(m[1], nn.Conv2d):
                # nn.init.kaiming_normal_(m[1].weight.data)
                nn.init.xavier_normal_(m[1].weight.data)
            elif isinstance(m[1], nn.BatchNorm2d):
                m[1].weight.data.fill_(1)
                m[1].bias.data.zero_()

    def setCandidatePos(self, candidate_pos, input_size=(100, 100)):
        if candidate_pos is None:
            self.mask = None
        else:
            self.mask = torch.zeros(input_size)
            index = np.array(np.meshgrid(candidate_pos[0], candidate_pos[1])).T.reshape(-1, 2)
            index = torch.tensor(index).long()
            for idx in index:
                self.mask[idx[0], idx[1]] = 1

    def forward(self, x):
        # x = x.permute(0, 3, 1, 2)
        feature_map = self.conv_net(x)
        q_values = self.q_values(feature_map)
        deconv_q_values = F.interpolate(q_values, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)

        if self.mask is not None:
            mask = self.mask.expand(deconv_q_values.size()).bool()
            mask = 1-mask
            deconv_q_values[mask] = -100
        return deconv_q_values

class FCNDown(FCN):
    def __init__(self):
        super(FCNDown, self).__init__()

    def forward(self, x):
        down_sampled = F.interpolate(x, size=(int(x.size(2)/2), int(x.size(3)/2)))
        feature_map = self.conv_net(down_sampled)
        q_values = self.q_values(feature_map)
        deconv_q_values = F.interpolate(q_values, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)

        return deconv_q_values


class CNN(nn.Module):
    def __init__(self, image_shape, n_outputs):
        super(CNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(image_shape[0], 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        conv_out_size = self._getConvOut(image_shape)
        self.fc1 = nn.Linear(conv_out_size, 512)
        self.fc2 = nn.Linear(512, n_outputs)

    def _getConvOut(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x)
        x = conv_out.view(x.size()[0], -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

class CNNSmall(nn.Module):
    def __init__(self, image_shape, n_outputs):
        super(CNNSmall, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(image_shape[0], 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        conv_out_size = self._getConvOut(image_shape)
        self.fc1 = nn.Linear(conv_out_size, 512)
        self.fc2 = nn.Linear(512, n_outputs)

    def _getConvOut(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x)
        x = conv_out.view(x.size()[0], -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

class CNNSigSmall(nn.Module):
    def __init__(self, image_shape, n_outputs):
        super(CNNSigSmall, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(image_shape[0], 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        conv_out_size = self._getConvOut(image_shape)
        self.fc1 = nn.Linear(conv_out_size, 512)
        self.fc2 = nn.Linear(512, n_outputs)

    def _getConvOut(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x)
        x = conv_out.view(x.size()[0], -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return torch.sigmoid(x)

class CNNDomainSmall(nn.Module):
    def __init__(self, image_shape, n_outputs):
        super(CNNDomainSmall, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(image_shape[0], 32, kernel_size=7, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        conv_out_size = self._getConvOut(image_shape)
        self.fc1 = nn.Linear(conv_out_size, 512)
        self.fc2 = nn.Linear(512, n_outputs)

    def _getConvOut(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x)
        x = conv_out.view(x.size()[0], -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

class CNNDBInputSmall(nn.Module):
    def __init__(self, input1_shape, input2_shape, n_outputs):
        super().__init__()
        self.input1_conv = nn.Sequential(OrderedDict([
            ('input1_conv1', nn.Conv2d(input1_shape[0], 32, kernel_size=7, stride=4)),
            ('input1_relu1', nn.ReLU(inplace=True)),
            ('input1_conv2', nn.Conv2d(32, 64, kernel_size=5, stride=2)),
            ('input1_relu2', nn.ReLU(inplace=True)),
            ('input1_conv3', nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            ('input1_relu3', nn.ReLU(inplace=True)),
        ]))
        self.input2_conv = nn.Sequential(OrderedDict([
            ('input2_conv1', nn.Conv2d(input2_shape[0], 32, kernel_size=4, stride=2)),
            ('input2_relu1', nn.ReLU(inplace=True)),
            ('input2_conv2', nn.Conv2d(32, 64, kernel_size=4, stride=1)),
            ('input2_relu2', nn.ReLU(inplace=True)),
            ('input2_conv3', nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            ('input2_relu3', nn.ReLU(inplace=True)),
        ]))

        conv1_out_size = self._getConv1Out(input1_shape)
        conv2_out_size = self._getConv2Out(input2_shape)

        self.fc1_1 = nn.Linear(conv1_out_size, 512)
        self.fc1_2 = nn.Linear(conv2_out_size, 512)
        self.fc2 = nn.Linear(1024, n_outputs)

        # conv_out_size = self._getConv1Out(input1_shape) + self._getConv2Out(input2_shape)
        # self.fc1 = nn.Linear(conv_out_size, 512)
        # self.fc2 = nn.Linear(512, n_outputs)

        for m in self.named_modules():
            if isinstance(m[1], nn.Conv2d):
                # nn.init.kaiming_normal_(m[1].weight.data)
                nn.init.xavier_normal_(m[1].weight.data)
            elif isinstance(m[1], nn.BatchNorm2d):
                m[1].weight.data.fill_(1)
                m[1].bias.data.zero_()

    def _getConv1Out(self, shape):
        o = self.input1_conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def _getConv2Out(self, shape):
        o = self.input2_conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        input1, input2 = x
        conv_out_1 = self.input1_conv(input1)
        conv_out_2 = self.input2_conv(input2)
        conv_out_1 = conv_out_1.view(input1.size(0), -1)
        conv_out_2 = conv_out_2.view(input2.size(0), -1)

        conv_out_1 = F.relu(self.fc1_1(conv_out_1))
        conv_out_2 = F.relu(self.fc1_2(conv_out_2))
        conv_out = torch.cat((conv_out_1, conv_out_2), dim=1)
        x = self.fc2(conv_out)
        return x

class CNNSig(nn.Module):
    def __init__(self, image_shape, n_outputs):
        super(CNNSig, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(image_shape[0], 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        conv_out_size = self._getConvOut(image_shape)
        self.fc1 = nn.Linear(conv_out_size, 512)
        self.fc2 = nn.Linear(512, n_outputs)

    def _getConvOut(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x)
        x = conv_out.view(x.size()[0], -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return torch.sigmoid(x)

class FCNInHandSmall(nn.Module):
    def __init__(self, n_input_channel=1, n_primitives=1, patch_shape=(1, 24, 24), n_in_hand_feature=10):
        super(FCNInHandSmall, self).__init__()

        self.fcn_conv = nn.Sequential(OrderedDict([
            ('fcn_conv1', nn.Conv2d(n_input_channel, 32, kernel_size=5, stride=1)),
            ('fcn_bn1', nn.BatchNorm2d(32)),
            ('fcn_relu1', nn.ReLU(inplace=True)),
            ('fcn_conv2', nn.Conv2d(32, 64, kernel_size=3, stride=1)),
            ('fcn_bn2', nn.BatchNorm2d(64)),
            ('fcn_relu2', nn.ReLU(inplace=True)),
            ('fcn_conv3', nn.Conv2d(64, 128, kernel_size=3, stride=1)),
            ('fcn_bn3', nn.BatchNorm2d(128)),
            ('fcn_relu3', nn.ReLU(inplace=True)),
            ('fcn_conv4', nn.Conv2d(128, 64, kernel_size=1, stride=1, bias=False)),
            ('fcn_bn4', nn.BatchNorm2d(64)),
            ('fcn_relu4', nn.ReLU(inplace=True)),
        ]))

        self.q_values = nn.Conv2d(64+n_in_hand_feature, n_primitives, kernel_size=1, stride=1, bias=False)

        self.cnn_conv = nn.Sequential(OrderedDict([
            ('cnn_conv1', nn.Conv2d(patch_shape[0], 32, kernel_size=4, stride=2)),
            ('cnn_relu1', nn.ReLU(inplace=True)),
            ('cnn_conv2', nn.Conv2d(32, 64, kernel_size=4, stride=1)),
            ('cnn_relu2', nn.ReLU(inplace=True)),
            ('cnn_conv3', nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            ('cnn_relu3', nn.ReLU(inplace=True)),
        ]))
        conv_out_size = self._getConvOut(patch_shape)
        self.fc1 = nn.Linear(conv_out_size, 512)
        self.fc2 = nn.Linear(512, n_in_hand_feature)

        for m in self.named_modules():
            if isinstance(m[1], nn.Conv2d):
                # nn.init.kaiming_normal_(m[1].weight.data)
                nn.init.xavier_normal_(m[1].weight.data)
            elif isinstance(m[1], nn.BatchNorm2d):
                m[1].weight.data.fill_(1)
                m[1].bias.data.zero_()

        self.mask = None

    def setCandidatePos(self, candidate_pos, input_size=(100, 100)):
        if candidate_pos is None:
            self.mask = None
        else:
            self.mask = torch.zeros(input_size)
            index = np.array(np.meshgrid(candidate_pos[0], candidate_pos[1])).T.reshape(-1, 2)
            index = torch.tensor(index).long()
            for idx in index:
                self.mask[idx[0], idx[1]] = 1

    def _getConvOut(self, shape):
        o = self.cnn_conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, obs, in_hand):
        feature_map = self.fcn_conv(obs)

        in_hand_out = self.cnn_conv(in_hand)
        in_hand_out = in_hand_out.view(in_hand.size()[0], -1)
        in_hand_out = self.fc1(in_hand_out)
        in_hand_out = F.relu(in_hand_out)
        in_hand_out = self.fc2(in_hand_out)
        in_hand_out = in_hand_out.reshape(in_hand_out.size(0), in_hand_out.size(1), 1, 1)
        in_hand_out = in_hand_out.expand(in_hand_out.size(0), in_hand_out.size(1), feature_map.size(2), feature_map.size(3))

        feature_map = torch.cat((feature_map, in_hand_out), dim=1)

        q_values = self.q_values(feature_map)
        deconv_q_values = F.interpolate(q_values, size=(obs.size(2), obs.size(3)), mode='bilinear', align_corners=False)

        if self.mask is not None:
            mask = self.mask.expand(deconv_q_values.size()).bool()
            mask = 1-mask
            deconv_q_values[mask] = -100
        return deconv_q_values

class FCNInHandDomainFCSmall(nn.Module):
    def __init__(self, n_input_channel=1, n_primitives=1, patch_shape=(1, 24, 24), domain_shape=(1, 100, 100),
                 n_in_hand_feature=10, n_domain_feature=10):
        super(FCNInHandDomainFCSmall, self).__init__()

        self.fcn_conv = nn.Sequential(OrderedDict([
            ('fcn_conv1', nn.Conv2d(n_input_channel, 32, kernel_size=5, stride=1)),
            ('fcn_relu1', nn.ReLU(inplace=True)),
            ('fcn_conv2', nn.Conv2d(32, 64, kernel_size=3, stride=1)),
            ('fcn_relu2', nn.ReLU(inplace=True)),
            ('fcn_conv3', nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            ('fcn_relu3', nn.ReLU(inplace=True)),
            ('fcn_conv4', nn.Conv2d(64, 16, kernel_size=1, stride=1, bias=False)),
            ('fcn_relu4', nn.ReLU(inplace=True)),
        ]))

        self.domain_features = nn.Sequential(OrderedDict([
            ('df_conv1', nn.Conv2d(16, 8, kernel_size=1, stride=1, bias=False)),
            ('df_pool1', nn.MaxPool2d(2, 2)),
            ('df_relu1', nn.ReLU(inplace=True))
        ]))

        self.cnn_conv = nn.Sequential(OrderedDict([
            ('cnn_conv1', nn.Conv2d(patch_shape[0], 32, kernel_size=4, stride=2)),
            ('cnn_relu1', nn.ReLU(inplace=True)),
            ('cnn_conv2', nn.Conv2d(32, 64, kernel_size=4, stride=1)),
            ('cnn_relu2', nn.ReLU(inplace=True)),
            ('cnn_conv3', nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            ('cnn_relu3', nn.ReLU(inplace=True)),
        ]))

        self.q_values = nn.Conv2d(16+n_domain_feature+n_in_hand_feature, n_primitives, kernel_size=1, stride=1, bias=False)

        conv_out_size = self._getConvOut(patch_shape)
        self.cnn_fc1 = nn.Linear(conv_out_size, 512)
        self.cnn_fc2 = nn.Linear(512, n_in_hand_feature)

        self.fcn_fc1 = nn.Linear(self._getFeatureOut(domain_shape), 512)
        self.fcn_fc2 = nn.Linear(512, n_domain_feature)

        for m in self.named_modules():
            if isinstance(m[1], nn.Conv2d):
                # nn.init.kaiming_normal_(m[1].weight.data)
                nn.init.xavier_normal_(m[1].weight.data)
            elif isinstance(m[1], nn.BatchNorm2d):
                m[1].weight.data.fill_(1)
                m[1].bias.data.zero_()

        self.mask = None

    def setCandidatePos(self, candidate_pos, input_size=(100, 100)):
        if candidate_pos is None:
            self.mask = None
        else:
            self.mask = torch.zeros(input_size)
            index = np.array(np.meshgrid(candidate_pos[0], candidate_pos[1])).T.reshape(-1, 2)
            index = torch.tensor(index).long()
            for idx in index:
                self.mask[idx[0], idx[1]] = 1

    def _getConvOut(self, shape):
        o = self.cnn_conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def _getFeatureOut(self, shape):
        o = self.domain_features(self.fcn_conv(torch.zeros(1, *shape)))
        return int(np.prod(o.size()))

    def forward(self, obs, in_hand):
        feature_map = self.fcn_conv(obs)

        domain_feature_map = self.domain_features(feature_map)
        feature_flatten = domain_feature_map.view(obs.size()[0], -1)
        feature_flatten = self.fcn_fc1(feature_flatten)
        feature_flatten = F.relu(feature_flatten)
        feature_flatten = self.fcn_fc2(feature_flatten)
        feature_flatten = feature_flatten.reshape(feature_flatten.size(0), feature_flatten.size(1), 1, 1)
        feature_flatten = feature_flatten.expand(feature_flatten.size(0), feature_flatten.size(1), feature_map.size(2),
                                                 feature_map.size(3))

        in_hand_out = self.cnn_conv(in_hand)
        in_hand_out = in_hand_out.view(in_hand.size()[0], -1)
        in_hand_out = self.cnn_fc1(in_hand_out)
        in_hand_out = F.relu(in_hand_out)
        in_hand_out = self.cnn_fc2(in_hand_out)
        in_hand_out = in_hand_out.reshape(in_hand_out.size(0), in_hand_out.size(1), 1, 1)
        in_hand_out = in_hand_out.expand(in_hand_out.size(0), in_hand_out.size(1), feature_map.size(2), feature_map.size(3))

        feature_map = torch.cat((feature_map, feature_flatten, in_hand_out), dim=1)

        q_values = self.q_values(feature_map)
        deconv_q_values = F.interpolate(q_values, size=(obs.size(2), obs.size(3)), mode='bilinear', align_corners=False)

        if self.mask is not None:
            mask = self.mask.expand(deconv_q_values.size()).bool()
            mask = 1-mask
            deconv_q_values[mask] = -100
        return deconv_q_values

class FCNInHandDynamicFilter(nn.Module):
    def __init__(self, n_input_channel=1, n_primitives=1, patch_shape=(1, 24, 24), domain_shape=(1, 100, 100)):
        super(FCNInHandDynamicFilter, self).__init__()
        self.domain_conv = nn.Sequential(OrderedDict([
            ('fcn_conv1', nn.Conv2d(n_input_channel, 32, kernel_size=5, stride=1, padding=2)),
            ('fcn_relu1', nn.ReLU(inplace=True)),
            ('fcn_conv2', nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)),
            ('fcn_relu2', nn.ReLU(inplace=True)),
            ('fcn_conv3', nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)),
            ('fcn_relu3', nn.ReLU(inplace=True)),
            ('fcn_conv4', nn.Conv2d(64, 16, kernel_size=1, stride=1, bias=False)),
            ('fcn_relu4', nn.ReLU(inplace=True)),
        ]))

        self.domain_down = nn.Sequential(OrderedDict([
            ('df_conv1', nn.Conv2d(16, 8, kernel_size=1, stride=1, bias=False)),
            ('df_pool1', nn.MaxPool2d(2, 2)),
            ('df_relu1', nn.ReLU(inplace=True))
        ]))

        self.in_hand_conv = nn.Sequential(OrderedDict([
            ('cnn_conv1', nn.Conv2d(patch_shape[0], 32, kernel_size=4, stride=2)),
            ('cnn_relu1', nn.ReLU(inplace=True)),
            ('cnn_conv2', nn.Conv2d(32, 64, kernel_size=4, stride=1)),
            ('cnn_relu2', nn.ReLU(inplace=True)),
            ('cnn_conv3', nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            ('cnn_relu3', nn.ReLU(inplace=True)),
        ]))

        self.q_values = nn.Conv2d(16, n_primitives, kernel_size=1, stride=1, bias=False)
        conv_out_size = self._getDomainConvOut(domain_shape) + self._getInHandConvOut(patch_shape)
        self.filter_fc = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(conv_out_size, 1024)),
            ('fc_relu1', nn.ReLU(inplace=True)),
            ('fc2', nn.Linear(1024, 16*16+16))
        ]))

        for m in self.named_modules():
            if isinstance(m[1], nn.Conv2d):
                # nn.init.kaiming_normal_(m[1].weight.data)
                nn.init.xavier_normal_(m[1].weight.data)
            elif isinstance(m[1], nn.BatchNorm2d):
                m[1].weight.data.fill_(1)
                m[1].bias.data.zero_()

        self.mask = None

    def setCandidatePos(self, candidate_pos, input_size=(100, 100)):
        if candidate_pos is None:
            self.mask = None
        else:
            self.mask = torch.zeros(input_size)
            index = np.array(np.meshgrid(candidate_pos[0], candidate_pos[1])).T.reshape(-1, 2)
            index = torch.tensor(index).long()
            for idx in index:
                self.mask[idx[0], idx[1]] = 1

    def _getInHandConvOut(self, shape):
        o = self.in_hand_conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def _getDomainConvOut(self, shape):
        o = self.domain_down(self.domain_conv(torch.zeros(1, *shape)))
        return int(np.prod(o.size()))

    def forward(self, obs, in_hand):
        feature_map = self.domain_conv(obs)
        domain_feature_map = self.domain_down(feature_map)
        feature_flatten = domain_feature_map.view(obs.size()[0], -1)
        in_hand_out = self.in_hand_conv(in_hand)
        in_hand_flatten = in_hand_out.view(in_hand.size()[0], -1)

        filter_weights = self.filter_fc(torch.cat((feature_flatten, in_hand_flatten), dim=1))
        filter_weight = filter_weights[:, :16*16].reshape(obs.size()[0], 16, 16, 1, 1)
        filter_bias = filter_weights[:, 16*16:].reshape(obs.size()[0], 16)

        filter_weight = filter_weight.reshape(obs.size()[0]*16, 16, 1, 1)
        filter_bias = filter_bias.reshape(obs.size()[0]*16)
        feature_map = feature_map.reshape(1, feature_map.size(0)*feature_map.size(1), feature_map.size(2), feature_map.size(3))
        feature_map = F.conv2d(feature_map, weight=filter_weight, bias=filter_bias, groups=obs.size(0))
        feature_map = feature_map.reshape(obs.size(0), -1, feature_map.size(2), feature_map.size(3))

        q_values = self.q_values(feature_map)
        # deconv_q_values = F.interpolate(q_values, size=(obs.size(2), obs.size(3)), mode='bilinear', align_corners=False)

        if self.mask is not None:
            mask = self.mask.expand(q_values.size()).bool()
            mask = 1-mask
            q_values[mask] = -100
        return q_values

class FCNInHandDynamicFilterSep(nn.Module):
    def __init__(self, n_input_channel=1, n_primitives=1, patch_shape=(1, 24, 24), domain_shape=(1, 100, 100)):
        super(FCNInHandDynamicFilterSep, self).__init__()
        self.pick_domain_conv = nn.Sequential(OrderedDict([
            ('pick_fcn_conv1', nn.Conv2d(n_input_channel, 32, kernel_size=5, stride=1, padding=2)),
            ('pick_fcn_relu1', nn.ReLU(inplace=True)),
            ('pick_fcn_conv2', nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)),
            ('pick_fcn_relu2', nn.ReLU(inplace=True)),
            ('pick_fcn_conv3', nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)),
            ('pick_fcn_relu3', nn.ReLU(inplace=True)),
            ('pick_fcn_conv4', nn.Conv2d(64, 16, kernel_size=1, stride=1, bias=False)),
            ('pick_fcn_relu4', nn.ReLU(inplace=True)),
        ]))
        self.pick_domain_down = nn.Sequential(OrderedDict([
            ('pick_df_conv1', nn.Conv2d(16, 8, kernel_size=1, stride=1, bias=False)),
            ('pick_df_pool1', nn.MaxPool2d(2, 2)),
            ('pick_df_relu1', nn.ReLU(inplace=True))
        ]))
        conv_out_size = self._getDomainConvOut(domain_shape)
        self.pick_filter_fc = nn.Sequential(OrderedDict([
            ('pick_fc1', nn.Linear(conv_out_size, 1024)),
            ('pick_fc_relu1', nn.ReLU(inplace=True)),
            ('pick_fc2', nn.Linear(1024, 16*16+16))
        ]))
        self.pick_q_values = nn.Conv2d(16, 1, kernel_size=1, stride=1, bias=False)

        self.place_domain_conv = nn.Sequential(OrderedDict([
            ('place_fcn_conv1', nn.Conv2d(n_input_channel, 32, kernel_size=5, stride=1, padding=2)),
            ('place_fcn_relu1', nn.ReLU(inplace=True)),
            ('place_fcn_conv2', nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)),
            ('place_fcn_relu2', nn.ReLU(inplace=True)),
            ('place_fcn_conv3', nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)),
            ('place_fcn_relu3', nn.ReLU(inplace=True)),
            ('place_fcn_conv4', nn.Conv2d(64, 16, kernel_size=1, stride=1, bias=False)),
            ('place_fcn_relu4', nn.ReLU(inplace=True)),
        ]))
        self.place_domain_down = nn.Sequential(OrderedDict([
            ('place_df_conv1', nn.Conv2d(16, 8, kernel_size=1, stride=1, bias=False)),
            ('place_df_pool1', nn.MaxPool2d(2, 2)),
            ('place_df_relu1', nn.ReLU(inplace=True))
        ]))
        self.in_hand_conv = nn.Sequential(OrderedDict([
            ('cnn_conv1', nn.Conv2d(patch_shape[0], 32, kernel_size=4, stride=2)),
            ('cnn_relu1', nn.ReLU(inplace=True)),
            ('cnn_conv2', nn.Conv2d(32, 64, kernel_size=4, stride=1)),
            ('cnn_relu2', nn.ReLU(inplace=True)),
            ('cnn_conv3', nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            ('cnn_relu3', nn.ReLU(inplace=True)),
        ]))
        place_conv_out_size = self._getDomainConvOut(domain_shape) + self._getInHandConvOut(patch_shape)
        self.place_filter_fc = nn.Sequential(OrderedDict([
            ('place_fc1', nn.Linear(place_conv_out_size, 1024)),
            ('place_fc_relu1', nn.ReLU(inplace=True)),
            ('place_fc2', nn.Linear(1024, 16 * 16 + 16))
        ]))
        self.place_q_values = nn.Conv2d(16, 1, kernel_size=1, stride=1, bias=False)


        for m in self.named_modules():
            if isinstance(m[1], nn.Conv2d):
                # nn.init.kaiming_normal_(m[1].weight.data)
                nn.init.xavier_normal_(m[1].weight.data)
            elif isinstance(m[1], nn.BatchNorm2d):
                m[1].weight.data.fill_(1)
                m[1].bias.data.zero_()

        self.mask = None

    def setCandidatePos(self, candidate_pos, input_size=(100, 100)):
        if candidate_pos is None:
            self.mask = None
        else:
            self.mask = torch.zeros(input_size)
            index = np.array(np.meshgrid(candidate_pos[0], candidate_pos[1])).T.reshape(-1, 2)
            index = torch.tensor(index).long()
            for idx in index:
                self.mask[idx[0], idx[1]] = 1

    def _getDomainConvOut(self, shape):
        o = self.pick_domain_down(self.pick_domain_conv(torch.zeros(1, *shape)))
        return int(np.prod(o.size()))

    def _getInHandConvOut(self, shape):
        o = self.in_hand_conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, obs, in_hand):
        pick_feature_map = self.pick_domain_conv(obs)
        pick_domain_feature_map = self.pick_domain_down(pick_feature_map)
        pick_feature_flatten = pick_domain_feature_map.view(obs.size()[0], -1)

        pick_filter_weights = self.pick_filter_fc(pick_feature_flatten)
        pick_filter_weight = pick_filter_weights[:, :16*16].reshape(obs.size()[0], 16, 16, 1, 1)
        pick_filter_bias = pick_filter_weights[:, 16*16:].reshape(obs.size()[0], 16)

        pick_filter_weight = pick_filter_weight.reshape(obs.size()[0]*16, 16, 1, 1)
        pick_filter_bias = pick_filter_bias.reshape(obs.size()[0]*16)
        pick_feature_map = pick_feature_map.reshape(1, pick_feature_map.size(0)*pick_feature_map.size(1), pick_feature_map.size(2), pick_feature_map.size(3))
        pick_feature_map = F.conv2d(pick_feature_map, weight=pick_filter_weight, bias=pick_filter_bias, groups=obs.size(0))
        pick_feature_map = pick_feature_map.reshape(obs.size(0), -1, pick_feature_map.size(2), pick_feature_map.size(3))

        pick_q_values = self.pick_q_values(pick_feature_map)

        place_feature_map = self.place_domain_conv(obs)
        place_domain_feature_map = self.place_domain_down(place_feature_map)
        place_feature_flatten = place_domain_feature_map.view(obs.size()[0], -1)
        in_hand_out = self.in_hand_conv(in_hand)
        in_hand_flatten = in_hand_out.view(in_hand.size()[0], -1)
        place_filter_weights = self.place_filter_fc(torch.cat((place_feature_flatten, in_hand_flatten), dim=1))
        place_filter_weight = place_filter_weights[:, :16 * 16].reshape(obs.size()[0], 16, 16, 1, 1)
        place_filter_bias = place_filter_weights[:, 16 * 16:].reshape(obs.size()[0], 16)

        place_filter_weight = place_filter_weight.reshape(obs.size()[0] * 16, 16, 1, 1)
        place_filter_bias = place_filter_bias.reshape(obs.size()[0] * 16)
        place_feature_map = place_feature_map.reshape(1, place_feature_map.size(0) * place_feature_map.size(1),
                                                    place_feature_map.size(2), place_feature_map.size(3))
        place_feature_map = F.conv2d(place_feature_map, weight=place_filter_weight, bias=place_filter_bias,
                                    groups=obs.size(0))
        place_feature_map = place_feature_map.reshape(obs.size(0), -1, place_feature_map.size(2), place_feature_map.size(3))

        place_q_values = self.place_q_values(place_feature_map)

        q_values = torch.cat((pick_q_values, place_q_values), dim=1)
        # deconv_q_values = F.interpolate(q_values, size=(obs.size(2), obs.size(3)), mode='bilinear', align_corners=False)

        if self.mask is not None:
            mask = self.mask.expand(q_values.size()).bool()
            mask = 1-mask
            q_values[mask] = -100
        return q_values

class FCNInHandDynamicFilterSepX(nn.Module):
    def __init__(self, n_input_channel=1, n_primitives=1, patch_shape=(1, 24, 24), domain_shape=(1, 100, 100)):
        super(FCNInHandDynamicFilterSepX, self).__init__()
        self.pick_domain_conv = nn.Sequential(OrderedDict([
            ('pick_fcn_conv1', nn.Conv2d(n_input_channel, 32, kernel_size=11, stride=1, padding=5)),
            ('pick_fcn_relu1', nn.ReLU(inplace=True)),
            ('pick_fcn_conv2', nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2)),
            ('pick_fcn_relu2', nn.ReLU(inplace=True)),
            ('pick_fcn_conv3', nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)),
            ('pick_fcn_relu3', nn.ReLU(inplace=True)),
            ('pick_fcn_conv4', nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)),
            ('pick_fcn_relu4', nn.ReLU(inplace=True)),
            ('pick_fcn_conv5', nn.Conv2d(32, 16, kernel_size=1, stride=1)),
            ('pick_fcn_relu5', nn.ReLU(inplace=True))
        ]))
        self.pick_domain_down = nn.Sequential(OrderedDict([
            ('pick_df_conv1', nn.Conv2d(16, 8, kernel_size=1, stride=1, bias=False)),
            ('pick_df_pool1', nn.MaxPool2d(2, 2)),
            ('pick_df_relu1', nn.ReLU(inplace=True))
        ]))
        conv_out_size = self._getDomainConvOut(domain_shape)
        self.pick_filter_fc = nn.Sequential(OrderedDict([
            ('pick_fc1', nn.Linear(conv_out_size, 1024)),
            ('pick_fc_relu1', nn.ReLU(inplace=True)),
            ('pick_fc2', nn.Linear(1024, 16*16+16))
        ]))
        self.pick_q_values = nn.Conv2d(16, 1, kernel_size=1, stride=1, bias=False)

        self.place_domain_conv = nn.Sequential(OrderedDict([
            ('place_fcn_conv1', nn.Conv2d(n_input_channel, 32, kernel_size=11, stride=1, padding=5)),
            ('place_fcn_relu1', nn.ReLU(inplace=True)),
            ('place_fcn_conv2', nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2)),
            ('place_fcn_relu2', nn.ReLU(inplace=True)),
            ('place_fcn_conv3', nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)),
            ('place_fcn_relu3', nn.ReLU(inplace=True)),
            ('place_fcn_conv4', nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)),
            ('place_fcn_relu4', nn.ReLU(inplace=True)),
            ('place_fcn_conv5', nn.Conv2d(32, 16, kernel_size=1, stride=1)),
            ('place_fcn_relu5', nn.ReLU(inplace=True)),
        ]))
        self.place_domain_down = nn.Sequential(OrderedDict([
            ('place_df_conv1', nn.Conv2d(16, 8, kernel_size=1, stride=1, bias=False)),
            ('place_df_pool1', nn.MaxPool2d(2, 2)),
            ('place_df_relu1', nn.ReLU(inplace=True))
        ]))
        self.in_hand_conv = nn.Sequential(OrderedDict([
            ('cnn_conv1', nn.Conv2d(patch_shape[0], 32, kernel_size=4, stride=2)),
            ('cnn_relu1', nn.ReLU(inplace=True)),
            ('cnn_conv2', nn.Conv2d(32, 64, kernel_size=4, stride=1)),
            ('cnn_relu2', nn.ReLU(inplace=True)),
            ('cnn_conv3', nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            ('cnn_relu3', nn.ReLU(inplace=True)),
        ]))
        place_conv_out_size = self._getDomainConvOut(domain_shape) + self._getInHandConvOut(patch_shape)
        self.place_filter_fc = nn.Sequential(OrderedDict([
            ('place_fc1', nn.Linear(place_conv_out_size, 1024)),
            ('place_fc_relu1', nn.ReLU(inplace=True)),
            ('place_fc2', nn.Linear(1024, 16 * 16 + 16))
        ]))
        self.place_q_values = nn.Conv2d(16, 1, kernel_size=1, stride=1, bias=False)


        for m in self.named_modules():
            if isinstance(m[1], nn.Conv2d):
                # nn.init.kaiming_normal_(m[1].weight.data)
                nn.init.xavier_normal_(m[1].weight.data)
            elif isinstance(m[1], nn.BatchNorm2d):
                m[1].weight.data.fill_(1)
                m[1].bias.data.zero_()

        self.mask = None

    def setCandidatePos(self, candidate_pos, input_size=(100, 100)):
        if candidate_pos is None:
            self.mask = None
        else:
            self.mask = torch.zeros(input_size)
            index = np.array(np.meshgrid(candidate_pos[0], candidate_pos[1])).T.reshape(-1, 2)
            index = torch.tensor(index).long()
            for idx in index:
                self.mask[idx[0], idx[1]] = 1

    def _getDomainConvOut(self, shape):
        o = self.pick_domain_down(self.pick_domain_conv(torch.zeros(1, *shape)))
        return int(np.prod(o.size()))

    def _getInHandConvOut(self, shape):
        o = self.in_hand_conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, obs, in_hand):
        pick_feature_map = self.pick_domain_conv(obs)
        pick_domain_feature_map = self.pick_domain_down(pick_feature_map)
        pick_feature_flatten = pick_domain_feature_map.view(obs.size()[0], -1)

        pick_filter_weights = self.pick_filter_fc(pick_feature_flatten)
        pick_filter_weight = pick_filter_weights[:, :16*16].reshape(obs.size()[0], 16, 16, 1, 1)
        pick_filter_bias = pick_filter_weights[:, 16*16:].reshape(obs.size()[0], 16)

        pick_filter_weight = pick_filter_weight.reshape(obs.size()[0]*16, 16, 1, 1)
        pick_filter_bias = pick_filter_bias.reshape(obs.size()[0]*16)
        pick_feature_map = pick_feature_map.reshape(1, pick_feature_map.size(0)*pick_feature_map.size(1), pick_feature_map.size(2), pick_feature_map.size(3))
        pick_feature_map = F.conv2d(pick_feature_map, weight=pick_filter_weight, bias=pick_filter_bias, groups=obs.size(0))
        pick_feature_map = pick_feature_map.reshape(obs.size(0), -1, pick_feature_map.size(2), pick_feature_map.size(3))

        pick_q_values = self.pick_q_values(pick_feature_map)

        place_feature_map = self.place_domain_conv(obs)
        place_domain_feature_map = self.place_domain_down(place_feature_map)
        place_feature_flatten = place_domain_feature_map.view(obs.size()[0], -1)
        in_hand_out = self.in_hand_conv(in_hand)
        in_hand_flatten = in_hand_out.view(in_hand.size()[0], -1)
        place_filter_weights = self.place_filter_fc(torch.cat((place_feature_flatten, in_hand_flatten), dim=1))
        place_filter_weight = place_filter_weights[:, :16 * 16].reshape(obs.size()[0], 16, 16, 1, 1)
        place_filter_bias = place_filter_weights[:, 16 * 16:].reshape(obs.size()[0], 16)

        place_filter_weight = place_filter_weight.reshape(obs.size()[0] * 16, 16, 1, 1)
        place_filter_bias = place_filter_bias.reshape(obs.size()[0] * 16)
        place_feature_map = place_feature_map.reshape(1, place_feature_map.size(0) * place_feature_map.size(1),
                                                    place_feature_map.size(2), place_feature_map.size(3))
        place_feature_map = F.conv2d(place_feature_map, weight=place_filter_weight, bias=place_filter_bias,
                                    groups=obs.size(0))
        place_feature_map = place_feature_map.reshape(obs.size(0), -1, place_feature_map.size(2), place_feature_map.size(3))

        place_q_values = self.place_q_values(place_feature_map)

        q_values = torch.cat((pick_q_values, place_q_values), dim=1)
        # deconv_q_values = F.interpolate(q_values, size=(obs.size(2), obs.size(3)), mode='bilinear', align_corners=False)

        if self.mask is not None:
            mask = self.mask.expand(q_values.size()).bool()
            mask = 1-mask
            q_values[mask] = -100
        return q_values

class FCNInHandPyramidSep(nn.Module):
    def __init__(self, n_input_channel=1, n_primitives=1, patch_shape=(1, 24, 24), domain_shape=(1, 100, 100)):
        super(FCNInHandPyramidSep, self).__init__()
        self.pick_conv1 = nn.Sequential(
            (nn.Conv2d(n_input_channel, 32, kernel_size=11, stride=1, padding=5)),
            (nn.ReLU(inplace=True)),
            (nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2)),
            (nn.ReLU(inplace=True)),
            (nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)),
            (nn.ReLU(inplace=True)),
            (nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)),
            (nn.ReLU(inplace=True)),
            (nn.Conv2d(32, 16, kernel_size=1, stride=1)),
            (nn.ReLU(inplace=True))
        )

        self.pick_conv2 = nn.Sequential(
            (nn.Conv2d(n_input_channel, 32, kernel_size=11, stride=1, padding=5)),
            (nn.ReLU(inplace=True)),
            (nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2)),
            (nn.ReLU(inplace=True)),
            (nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)),
            (nn.ReLU(inplace=True)),
            (nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)),
            (nn.ReLU(inplace=True)),
            (nn.Conv2d(32, 16, kernel_size=1, stride=1)),
            (nn.ReLU(inplace=True))
        )

        self.pick_conv3 = nn.Sequential(
            (nn.Conv2d(n_input_channel, 32, kernel_size=11, stride=1, padding=5)),
            (nn.ReLU(inplace=True)),
            (nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2)),
            (nn.ReLU(inplace=True)),
            (nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)),
            (nn.ReLU(inplace=True)),
            (nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)),
            (nn.ReLU(inplace=True)),
            (nn.Conv2d(32, 16, kernel_size=1, stride=1)),
            (nn.ReLU(inplace=True))
        )

        self.pick_conv4 = nn.Sequential(
            (nn.Conv2d(n_input_channel, 32, kernel_size=11, stride=1, padding=5)),
            (nn.ReLU(inplace=True)),
            (nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2)),
            (nn.ReLU(inplace=True)),
            (nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)),
            (nn.ReLU(inplace=True)),
            (nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)),
            (nn.ReLU(inplace=True)),
            (nn.Conv2d(32, 16, kernel_size=1, stride=1)),
            (nn.ReLU(inplace=True))
        )

        self.pick_q_values = nn.Conv2d(64, 1, kernel_size=1, stride=1, bias=False)

        self.place_conv1 = nn.Sequential(
            (nn.Conv2d(n_input_channel, 32, kernel_size=11, stride=1, padding=5)),
            (nn.ReLU(inplace=True)),
            (nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2)),
            (nn.ReLU(inplace=True)),
            (nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)),
            (nn.ReLU(inplace=True)),
            (nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)),
            (nn.ReLU(inplace=True)),
            (nn.Conv2d(32, 16, kernel_size=1, stride=1)),
            (nn.ReLU(inplace=True))
        )

        self.place_conv2 = nn.Sequential(
            (nn.Conv2d(n_input_channel, 32, kernel_size=11, stride=1, padding=5)),
            (nn.ReLU(inplace=True)),
            (nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2)),
            (nn.ReLU(inplace=True)),
            (nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)),
            (nn.ReLU(inplace=True)),
            (nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)),
            (nn.ReLU(inplace=True)),
            (nn.Conv2d(32, 16, kernel_size=1, stride=1)),
            (nn.ReLU(inplace=True))
        )

        self.place_conv3 = nn.Sequential(
            (nn.Conv2d(n_input_channel, 32, kernel_size=11, stride=1, padding=5)),
            (nn.ReLU(inplace=True)),
            (nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2)),
            (nn.ReLU(inplace=True)),
            (nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)),
            (nn.ReLU(inplace=True)),
            (nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)),
            (nn.ReLU(inplace=True)),
            (nn.Conv2d(32, 16, kernel_size=1, stride=1)),
            (nn.ReLU(inplace=True))
        )

        self.place_conv4 = nn.Sequential(
            (nn.Conv2d(n_input_channel, 32, kernel_size=11, stride=1, padding=5)),
            (nn.ReLU(inplace=True)),
            (nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2)),
            (nn.ReLU(inplace=True)),
            (nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)),
            (nn.ReLU(inplace=True)),
            (nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)),
            (nn.ReLU(inplace=True)),
            (nn.Conv2d(32, 16, kernel_size=1, stride=1)),
            (nn.ReLU(inplace=True))
        )

        self.in_hand_conv = nn.Sequential(OrderedDict([
            ('cnn_conv1', nn.Conv2d(patch_shape[0], 32, kernel_size=4, stride=2)),
            ('cnn_relu1', nn.ReLU(inplace=True)),
            ('cnn_conv2', nn.Conv2d(32, 64, kernel_size=4, stride=1)),
            ('cnn_relu2', nn.ReLU(inplace=True)),
            ('cnn_conv3', nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            ('cnn_relu3', nn.ReLU(inplace=True)),
        ]))
        place_conv_out_size = self._getInHandConvOut(patch_shape)
        self.place_filter_fc = nn.Sequential(OrderedDict([
            ('place_fc1', nn.Linear(place_conv_out_size, 1024)),
            ('place_fc_relu1', nn.ReLU(inplace=True)),
            ('place_fc2', nn.Linear(1024, 64 * 64 + 64))
        ]))
        self.place_q_values = nn.Conv2d(64, 1, kernel_size=1, stride=1, bias=False)


        for m in self.named_modules():
            if isinstance(m[1], nn.Conv2d):
                # nn.init.kaiming_normal_(m[1].weight.data)
                nn.init.xavier_normal_(m[1].weight.data)
            elif isinstance(m[1], nn.BatchNorm2d):
                m[1].weight.data.fill_(1)
                m[1].bias.data.zero_()

        self.mask = None

    def setCandidatePos(self, candidate_pos, input_size=(100, 100)):
        if candidate_pos is None:
            self.mask = None
        else:
            self.mask = torch.zeros(input_size)
            index = np.array(np.meshgrid(candidate_pos[0], candidate_pos[1])).T.reshape(-1, 2)
            index = torch.tensor(index).long()
            for idx in index:
                self.mask[idx[0], idx[1]] = 1

    def _getInHandConvOut(self, shape):
        o = self.in_hand_conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, obs, in_hand):
        obs_2 = F.interpolate(obs, scale_factor=0.5, mode='bilinear', align_corners=False)
        obs_4 = F.interpolate(obs_2, scale_factor=0.5, mode='bilinear', align_corners=False)
        obs_8 = F.interpolate(obs_4, scale_factor=0.5, mode='bilinear', align_corners=False)

        pick_feature_map_1 = self.pick_conv1(obs)
        pick_feature_map_2 = F.interpolate(self.pick_conv2(obs_2), size=(obs.shape[2], obs.shape[3]), mode='bilinear', align_corners=False)
        pick_feature_map_4 = F.interpolate(self.pick_conv3(obs_4), size=(obs.shape[2], obs.shape[3]), mode='bilinear', align_corners=False)
        pick_feature_map_8 = F.interpolate(self.pick_conv4(obs_8), size=(obs.shape[2], obs.shape[3]), mode='bilinear', align_corners=False)

        pick_feature_map = torch.cat((pick_feature_map_1, pick_feature_map_2, pick_feature_map_4, pick_feature_map_8), dim=1)
        pick_q_values = self.pick_q_values(pick_feature_map)

        place_feature_map_1 = self.place_conv1(obs)
        place_feature_map_2 = F.interpolate(self.place_conv2(obs_2), size=(obs.shape[2], obs.shape[3]), mode='bilinear', align_corners=False)
        place_feature_map_4 = F.interpolate(self.place_conv3(obs_4), size=(obs.shape[2], obs.shape[3]), mode='bilinear', align_corners=False)
        place_feature_map_8 = F.interpolate(self.place_conv4(obs_8), size=(obs.shape[2], obs.shape[3]), mode='bilinear', align_corners=False)

        place_feature_map = torch.cat((place_feature_map_1, place_feature_map_2, place_feature_map_4, place_feature_map_8), dim=1)

        in_hand_out = self.in_hand_conv(in_hand)
        in_hand_flatten = in_hand_out.view(in_hand.size()[0], -1)
        place_filter_weights = self.place_filter_fc(in_hand_flatten)
        place_filter_weight = place_filter_weights[:, :64 * 64].reshape(obs.size()[0], 64, 64, 1, 1)
        place_filter_bias = place_filter_weights[:, 64 * 64:].reshape(obs.size()[0], 64)
        place_filter_weight = place_filter_weight.reshape(obs.size()[0] * 64, 64, 1, 1)
        place_filter_bias = place_filter_bias.reshape(obs.size()[0] * 64)
        place_feature_map = place_feature_map.reshape(1, place_feature_map.size(0) * place_feature_map.size(1),
                                                      place_feature_map.size(2), place_feature_map.size(3))
        place_feature_map = F.conv2d(place_feature_map, weight=place_filter_weight, bias=place_filter_bias,
                                     groups=obs.size(0))
        place_feature_map = place_feature_map.reshape(obs.size(0), -1, place_feature_map.size(2),
                                                      place_feature_map.size(3))

        place_q_values = self.place_q_values(place_feature_map)

        q_values = torch.cat((pick_q_values, place_q_values), dim=1)

        if self.mask is not None:
            mask = self.mask.expand(q_values.size()).bool()
            mask = 1-mask
            q_values[mask] = -100
        return q_values

class FCNInHandDynamicFilterPyramidSep(nn.Module):
    def __init__(self, n_input_channel=1, n_primitives=1, patch_shape=(1, 24, 24), domain_shape=(1, 100, 100)):
        super(FCNInHandDynamicFilterPyramidSep, self).__init__()
        self.pick_conv1 = nn.Sequential(
            (nn.Conv2d(n_input_channel, 32, kernel_size=11, stride=1, padding=5)),
            (nn.ReLU(inplace=True)),
            (nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2)),
            (nn.ReLU(inplace=True)),
            (nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)),
            (nn.ReLU(inplace=True)),
            (nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)),
            (nn.ReLU(inplace=True)),
            (nn.Conv2d(32, 16, kernel_size=1, stride=1)),
            (nn.ReLU(inplace=True))
        )
        self.pick_conv2 = nn.Sequential(
            (nn.Conv2d(n_input_channel, 32, kernel_size=11, stride=1, padding=5)),
            (nn.ReLU(inplace=True)),
            (nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2)),
            (nn.ReLU(inplace=True)),
            (nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)),
            (nn.ReLU(inplace=True)),
            (nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)),
            (nn.ReLU(inplace=True)),
            (nn.Conv2d(32, 16, kernel_size=1, stride=1)),
            (nn.ReLU(inplace=True))
        )
        self.pick_domain_down = nn.Sequential(
            (nn.Conv2d(32, 8, kernel_size=1, stride=1, bias=False)),
            (nn.MaxPool2d(2, 2)),
            (nn.ReLU(inplace=True))
        )
        conv_out_size = self._getDomainConvOut(domain_shape)
        self.pick_filter_fc = nn.Sequential(
            (nn.Linear(conv_out_size, 1024)),
            (nn.ReLU(inplace=True)),
            (nn.Linear(1024, 32 * 32 + 32))
        )
        self.pick_q_values = nn.Conv2d(32, 1, kernel_size=1, stride=1, bias=False)

        self.place_conv1 = nn.Sequential(
            (nn.Conv2d(n_input_channel, 32, kernel_size=11, stride=1, padding=5)),
            (nn.ReLU(inplace=True)),
            (nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2)),
            (nn.ReLU(inplace=True)),
            (nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)),
            (nn.ReLU(inplace=True)),
            (nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)),
            (nn.ReLU(inplace=True)),
            (nn.Conv2d(32, 16, kernel_size=1, stride=1)),
            (nn.ReLU(inplace=True))
        )
        self.place_conv2 = nn.Sequential(
            (nn.Conv2d(n_input_channel, 32, kernel_size=11, stride=1, padding=5)),
            (nn.ReLU(inplace=True)),
            (nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2)),
            (nn.ReLU(inplace=True)),
            (nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)),
            (nn.ReLU(inplace=True)),
            (nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)),
            (nn.ReLU(inplace=True)),
            (nn.Conv2d(32, 16, kernel_size=1, stride=1)),
            (nn.ReLU(inplace=True))
        )
        self.place_domain_down = nn.Sequential(
            (nn.Conv2d(32, 8, kernel_size=1, stride=1, bias=False)),
            (nn.MaxPool2d(2, 2)),
            (nn.ReLU(inplace=True))
        )
        self.in_hand_conv = nn.Sequential(
            (nn.Conv2d(patch_shape[0], 32, kernel_size=4, stride=2)),
            (nn.ReLU(inplace=True)),
            (nn.Conv2d(32, 64, kernel_size=4, stride=1)),
            (nn.ReLU(inplace=True)),
            (nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            (nn.ReLU(inplace=True)),
        )
        place_conv_out_size = self._getDomainConvOut(domain_shape) + self._getInHandConvOut(patch_shape)
        self.place_filter_fc = nn.Sequential(
            (nn.Linear(place_conv_out_size, 1024)),
            (nn.ReLU(inplace=True)),
            (nn.Linear(1024, 32 * 32 + 32))
        )
        self.place_q_values = nn.Conv2d(32, 1, kernel_size=1, stride=1, bias=False)

        for m in self.named_modules():
            if isinstance(m[1], nn.Conv2d):
                # nn.init.kaiming_normal_(m[1].weight.data)
                nn.init.xavier_normal_(m[1].weight.data)
            elif isinstance(m[1], nn.BatchNorm2d):
                m[1].weight.data.fill_(1)
                m[1].bias.data.zero_()

        self.mask = None

    def setCandidatePos(self, candidate_pos, input_size=(100, 100)):
        if candidate_pos is None:
            self.mask = None
        else:
            self.mask = torch.zeros(input_size)
            index = np.array(np.meshgrid(candidate_pos[0], candidate_pos[1])).T.reshape(-1, 2)
            index = torch.tensor(index).long()
            for idx in index:
                self.mask[idx[0], idx[1]] = 1

    def _getDomainConvOut(self, shape):
        o = self.pick_domain_down(torch.cat((self.pick_conv1(torch.zeros(1, *shape)),
                                            self.pick_conv2(torch.zeros(1, *shape))),
                                            dim=1))
        return int(np.prod(o.size()))

    def _getInHandConvOut(self, shape):
        o = self.in_hand_conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, obs, in_hand):
        df_channel = 32
        obs_2 = F.interpolate(obs, scale_factor=0.5, mode='bilinear', align_corners=False)

        pick_feature_map_1 = self.pick_conv1(obs)
        pick_feature_map_2 = F.interpolate(self.pick_conv2(obs_2), size=(obs.shape[2], obs.shape[3]), mode='bilinear', align_corners=False)
        pick_feature_map = torch.cat((pick_feature_map_1, pick_feature_map_2), dim=1)

        pick_domain_feature_map = self.pick_domain_down(pick_feature_map)
        pick_feature_flatten = pick_domain_feature_map.view(obs.size()[0], -1)

        pick_filter_weights = self.pick_filter_fc(pick_feature_flatten)
        pick_filter_weight = pick_filter_weights[:, :df_channel * df_channel].reshape(obs.size()[0], df_channel, df_channel, 1, 1)
        pick_filter_bias = pick_filter_weights[:, df_channel * df_channel:].reshape(obs.size()[0], df_channel)
        pick_filter_weight = pick_filter_weight.reshape(obs.size()[0] * df_channel, df_channel, 1, 1)
        pick_filter_bias = pick_filter_bias.reshape(obs.size()[0] * df_channel)
        pick_feature_map = pick_feature_map.reshape(1, pick_feature_map.size(0) * pick_feature_map.size(1),
                                                    pick_feature_map.size(2), pick_feature_map.size(3))
        pick_feature_map = F.conv2d(pick_feature_map, weight=pick_filter_weight, bias=pick_filter_bias,
                                    groups=obs.size(0))
        pick_feature_map = pick_feature_map.reshape(obs.size(0), -1, pick_feature_map.size(2), pick_feature_map.size(3))

        pick_q_values = self.pick_q_values(pick_feature_map)

        place_feature_map_1 = self.place_conv1(obs)
        place_feature_map_2 = F.interpolate(self.place_conv2(obs_2), size=(obs.shape[2], obs.shape[3]), mode='bilinear', align_corners=False)
        place_feature_map = torch.cat((place_feature_map_1, place_feature_map_2), dim=1)

        place_domain_feature_map = self.place_domain_down(place_feature_map)
        place_feature_flatten = place_domain_feature_map.view(obs.size()[0], -1)
        in_hand_out = self.in_hand_conv(in_hand)
        in_hand_flatten = in_hand_out.view(in_hand.size()[0], -1)
        place_filter_weights = self.place_filter_fc(torch.cat((place_feature_flatten, in_hand_flatten), dim=1))
        place_filter_weight = place_filter_weights[:, :df_channel * df_channel].reshape(obs.size()[0], df_channel, df_channel, 1, 1)
        place_filter_bias = place_filter_weights[:, df_channel * df_channel:].reshape(obs.size()[0], df_channel)
        place_filter_weight = place_filter_weight.reshape(obs.size()[0] * df_channel, df_channel, 1, 1)
        place_filter_bias = place_filter_bias.reshape(obs.size()[0] * df_channel)
        place_feature_map = place_feature_map.reshape(1, place_feature_map.size(0) * place_feature_map.size(1),
                                                      place_feature_map.size(2), place_feature_map.size(3))
        place_feature_map = F.conv2d(place_feature_map, weight=place_filter_weight, bias=place_filter_bias,
                                     groups=obs.size(0))
        place_feature_map = place_feature_map.reshape(obs.size(0), -1, place_feature_map.size(2),
                                                      place_feature_map.size(3))

        place_q_values = self.place_q_values(place_feature_map)

        q_values = torch.cat((pick_q_values, place_q_values), dim=1)

        if self.mask is not None:
            mask = self.mask.expand(q_values.size()).bool()
            mask = 1-mask
            q_values[mask] = -100
        return q_values

class FCNInHandDynamicFilterPyramidSepDown4(nn.Module):
    def __init__(self, n_input_channel=1, n_primitives=1, patch_shape=(1, 24, 24), domain_shape=(1, 100, 100)):
        super(FCNInHandDynamicFilterPyramidSepDown4, self).__init__()
        self.pick_conv1 = nn.Sequential(
            (nn.Conv2d(n_input_channel, 32, kernel_size=11, stride=1, padding=5)),
            (nn.ReLU(inplace=True)),
            (nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2)),
            (nn.ReLU(inplace=True)),
            (nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)),
            (nn.ReLU(inplace=True)),
            (nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)),
            (nn.ReLU(inplace=True)),
            (nn.Conv2d(32, 16, kernel_size=1, stride=1)),
            (nn.ReLU(inplace=True))
        )
        self.pick_conv2 = nn.Sequential(
            (nn.Conv2d(n_input_channel, 32, kernel_size=11, stride=1, padding=5)),
            (nn.ReLU(inplace=True)),
            (nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2)),
            (nn.ReLU(inplace=True)),
            (nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)),
            (nn.ReLU(inplace=True)),
            (nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)),
            (nn.ReLU(inplace=True)),
            (nn.Conv2d(32, 16, kernel_size=1, stride=1)),
            (nn.ReLU(inplace=True))
        )
        self.pick_domain_down = nn.Sequential(
            (nn.Conv2d(32, 4, kernel_size=1, stride=1, bias=False)),
            (nn.MaxPool2d(2, 2)),
            (nn.ReLU(inplace=True))
        )
        conv_out_size = self._getDomainConvOut(domain_shape)
        self.pick_filter_fc = nn.Sequential(
            (nn.Linear(conv_out_size, 1024)),
            (nn.ReLU(inplace=True)),
            (nn.Linear(1024, 32 * 32 + 32))
        )
        self.pick_q_values = nn.Conv2d(32, 1, kernel_size=1, stride=1, bias=False)

        self.place_conv1 = nn.Sequential(
            (nn.Conv2d(n_input_channel, 32, kernel_size=11, stride=1, padding=5)),
            (nn.ReLU(inplace=True)),
            (nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2)),
            (nn.ReLU(inplace=True)),
            (nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)),
            (nn.ReLU(inplace=True)),
            (nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)),
            (nn.ReLU(inplace=True)),
            (nn.Conv2d(32, 16, kernel_size=1, stride=1)),
            (nn.ReLU(inplace=True))
        )
        self.place_conv2 = nn.Sequential(
            (nn.Conv2d(n_input_channel, 32, kernel_size=11, stride=1, padding=5)),
            (nn.ReLU(inplace=True)),
            (nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2)),
            (nn.ReLU(inplace=True)),
            (nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)),
            (nn.ReLU(inplace=True)),
            (nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)),
            (nn.ReLU(inplace=True)),
            (nn.Conv2d(32, 16, kernel_size=1, stride=1)),
            (nn.ReLU(inplace=True))
        )
        self.place_domain_down = nn.Sequential(
            (nn.Conv2d(32, 4, kernel_size=1, stride=1, bias=False)),
            (nn.MaxPool2d(2, 2)),
            (nn.ReLU(inplace=True))
        )
        self.in_hand_conv = nn.Sequential(
            (nn.Conv2d(patch_shape[0], 32, kernel_size=4, stride=2)),
            (nn.ReLU(inplace=True)),
            (nn.Conv2d(32, 64, kernel_size=4, stride=1)),
            (nn.ReLU(inplace=True)),
            (nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            (nn.ReLU(inplace=True)),
        )
        place_conv_out_size = self._getDomainConvOut(domain_shape) + self._getInHandConvOut(patch_shape)
        self.place_filter_fc = nn.Sequential(
            (nn.Linear(place_conv_out_size, 1024)),
            (nn.ReLU(inplace=True)),
            (nn.Linear(1024, 32 * 32 + 32))
        )
        self.place_q_values = nn.Conv2d(32, 1, kernel_size=1, stride=1, bias=False)

        for m in self.named_modules():
            if isinstance(m[1], nn.Conv2d):
                # nn.init.kaiming_normal_(m[1].weight.data)
                nn.init.xavier_normal_(m[1].weight.data)
            elif isinstance(m[1], nn.BatchNorm2d):
                m[1].weight.data.fill_(1)
                m[1].bias.data.zero_()

        self.mask = None

    def setCandidatePos(self, candidate_pos, input_size=(100, 100)):
        if candidate_pos is None:
            self.mask = None
        else:
            self.mask = torch.zeros(input_size)
            index = np.array(np.meshgrid(candidate_pos[0], candidate_pos[1])).T.reshape(-1, 2)
            index = torch.tensor(index).long()
            for idx in index:
                self.mask[idx[0], idx[1]] = 1

    def _getDomainConvOut(self, shape):
        o = self.pick_domain_down(torch.cat((self.pick_conv1(torch.zeros(1, *shape)),
                                            self.pick_conv2(torch.zeros(1, *shape))),
                                            dim=1))
        return int(np.prod(o.size()))

    def _getInHandConvOut(self, shape):
        o = self.in_hand_conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, obs, in_hand):
        df_channel = 32
        obs_2 = F.interpolate(obs, scale_factor=0.5, mode='bilinear', align_corners=False)

        pick_feature_map_1 = self.pick_conv1(obs)
        pick_feature_map_2 = F.interpolate(self.pick_conv2(obs_2), size=(obs.shape[2], obs.shape[3]), mode='bilinear', align_corners=False)
        pick_feature_map = torch.cat((pick_feature_map_1, pick_feature_map_2), dim=1)

        pick_domain_feature_map = self.pick_domain_down(pick_feature_map)
        pick_feature_flatten = pick_domain_feature_map.view(obs.size()[0], -1)

        pick_filter_weights = self.pick_filter_fc(pick_feature_flatten)
        pick_filter_weight = pick_filter_weights[:, :df_channel * df_channel].reshape(obs.size()[0], df_channel, df_channel, 1, 1)
        pick_filter_bias = pick_filter_weights[:, df_channel * df_channel:].reshape(obs.size()[0], df_channel)
        pick_filter_weight = pick_filter_weight.reshape(obs.size()[0] * df_channel, df_channel, 1, 1)
        pick_filter_bias = pick_filter_bias.reshape(obs.size()[0] * df_channel)
        pick_feature_map = pick_feature_map.reshape(1, pick_feature_map.size(0) * pick_feature_map.size(1),
                                                    pick_feature_map.size(2), pick_feature_map.size(3))
        pick_feature_map = F.conv2d(pick_feature_map, weight=pick_filter_weight, bias=pick_filter_bias,
                                    groups=obs.size(0))
        pick_feature_map = pick_feature_map.reshape(obs.size(0), -1, pick_feature_map.size(2), pick_feature_map.size(3))

        pick_q_values = self.pick_q_values(pick_feature_map)

        place_feature_map_1 = self.place_conv1(obs)
        place_feature_map_2 = F.interpolate(self.place_conv2(obs_2), size=(obs.shape[2], obs.shape[3]), mode='bilinear', align_corners=False)
        place_feature_map = torch.cat((place_feature_map_1, place_feature_map_2), dim=1)

        place_domain_feature_map = self.place_domain_down(place_feature_map)
        place_feature_flatten = place_domain_feature_map.view(obs.size()[0], -1)
        in_hand_out = self.in_hand_conv(in_hand)
        in_hand_flatten = in_hand_out.view(in_hand.size()[0], -1)
        place_filter_weights = self.place_filter_fc(torch.cat((place_feature_flatten, in_hand_flatten), dim=1))
        place_filter_weight = place_filter_weights[:, :df_channel * df_channel].reshape(obs.size()[0], df_channel, df_channel, 1, 1)
        place_filter_bias = place_filter_weights[:, df_channel * df_channel:].reshape(obs.size()[0], df_channel)
        place_filter_weight = place_filter_weight.reshape(obs.size()[0] * df_channel, df_channel, 1, 1)
        place_filter_bias = place_filter_bias.reshape(obs.size()[0] * df_channel)
        place_feature_map = place_feature_map.reshape(1, place_feature_map.size(0) * place_feature_map.size(1),
                                                      place_feature_map.size(2), place_feature_map.size(3))
        place_feature_map = F.conv2d(place_feature_map, weight=place_filter_weight, bias=place_filter_bias,
                                     groups=obs.size(0))
        place_feature_map = place_feature_map.reshape(obs.size(0), -1, place_feature_map.size(2),
                                                      place_feature_map.size(3))

        place_q_values = self.place_q_values(place_feature_map)

        q_values = torch.cat((pick_q_values, place_q_values), dim=1)

        if self.mask is not None:
            mask = self.mask.expand(q_values.size()).bool()
            mask = 1-mask
            q_values[mask] = -100
        return q_values


class FCNInHandDynamicFilterPyramidSepCross(nn.Module):
    def __init__(self, n_input_channel=1, n_primitives=1, patch_shape=(1, 24, 24), domain_shape=(1, 100, 100)):
        super(FCNInHandDynamicFilterPyramidSepCross, self).__init__()
        self.pick_conv1 = nn.Sequential(
            (nn.Conv2d(n_input_channel, 32, kernel_size=11, stride=1, padding=5)),
            (nn.ReLU(inplace=True)),
            (nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2)),
            (nn.ReLU(inplace=True)),
            (nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)),
            (nn.ReLU(inplace=True)),
            (nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)),
            (nn.ReLU(inplace=True)),
            (nn.Conv2d(32, 16, kernel_size=1, stride=1)),
            (nn.ReLU(inplace=True))
        )
        self.pick_conv2 = nn.Sequential(
            (nn.Conv2d(n_input_channel, 32, kernel_size=11, stride=1, padding=5)),
            (nn.ReLU(inplace=True)),
            (nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2)),
            (nn.ReLU(inplace=True)),
            (nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)),
            (nn.ReLU(inplace=True)),
            (nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)),
            (nn.ReLU(inplace=True)),
            (nn.Conv2d(32, 16, kernel_size=1, stride=1)),
            (nn.ReLU(inplace=True))
        )
        self.pick_domain_down = nn.Sequential(
            (nn.Conv2d(32, 8, kernel_size=1, stride=1, bias=False)),
            (nn.MaxPool2d(2, 2)),
            (nn.ReLU(inplace=True))
        )
        conv_out_size = self._getDomainConvOut(domain_shape)
        self.pick_filter_fc = nn.Sequential(
            (nn.Linear(conv_out_size, 1024)),
            (nn.ReLU(inplace=True)),
            (nn.Linear(1024, 2 * (16 * 16 + 16)))
        )
        self.pick_q_values = nn.Conv2d(32, 1, kernel_size=1, stride=1, bias=False)

        self.place_conv1 = nn.Sequential(
            (nn.Conv2d(n_input_channel, 32, kernel_size=11, stride=1, padding=5)),
            (nn.ReLU(inplace=True)),
            (nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2)),
            (nn.ReLU(inplace=True)),
            (nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)),
            (nn.ReLU(inplace=True)),
            (nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)),
            (nn.ReLU(inplace=True)),
            (nn.Conv2d(32, 16, kernel_size=1, stride=1)),
            (nn.ReLU(inplace=True))
        )
        self.place_conv2 = nn.Sequential(
            (nn.Conv2d(n_input_channel, 32, kernel_size=11, stride=1, padding=5)),
            (nn.ReLU(inplace=True)),
            (nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2)),
            (nn.ReLU(inplace=True)),
            (nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)),
            (nn.ReLU(inplace=True)),
            (nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)),
            (nn.ReLU(inplace=True)),
            (nn.Conv2d(32, 16, kernel_size=1, stride=1)),
            (nn.ReLU(inplace=True))
        )
        self.place_domain_down = nn.Sequential(
            (nn.Conv2d(32, 8, kernel_size=1, stride=1, bias=False)),
            (nn.MaxPool2d(2, 2)),
            (nn.ReLU(inplace=True))
        )
        self.in_hand_conv = nn.Sequential(
            (nn.Conv2d(patch_shape[0], 32, kernel_size=4, stride=2)),
            (nn.ReLU(inplace=True)),
            (nn.Conv2d(32, 64, kernel_size=4, stride=1)),
            (nn.ReLU(inplace=True)),
            (nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            (nn.ReLU(inplace=True)),
        )
        place_conv_out_size = self._getDomainConvOut(domain_shape) + self._getInHandConvOut(patch_shape)
        self.place_filter_fc = nn.Sequential(
            (nn.Linear(place_conv_out_size, 1024)),
            (nn.ReLU(inplace=True)),
            (nn.Linear(1024, 2 * (16 * 16 + 16)))
        )
        self.place_q_values = nn.Conv2d(32, 1, kernel_size=1, stride=1, bias=False)

        for m in self.named_modules():
            if isinstance(m[1], nn.Conv2d):
                # nn.init.kaiming_normal_(m[1].weight.data)
                nn.init.xavier_normal_(m[1].weight.data)
            elif isinstance(m[1], nn.BatchNorm2d):
                m[1].weight.data.fill_(1)
                m[1].bias.data.zero_()

        self.mask = None

    def setCandidatePos(self, candidate_pos, input_size=(100, 100)):
        if candidate_pos is None:
            self.mask = None
        else:
            self.mask = torch.zeros(input_size)
            index = np.array(np.meshgrid(candidate_pos[0], candidate_pos[1])).T.reshape(-1, 2)
            index = torch.tensor(index).long()
            for idx in index:
                self.mask[idx[0], idx[1]] = 1

    def _getDomainConvOut(self, shape):
        o = self.pick_domain_down(torch.cat((self.pick_conv1(torch.zeros(1, *shape)),
                                            self.pick_conv2(torch.zeros(1, *shape))),
                                            dim=1))
        return int(np.prod(o.size()))

    def _getInHandConvOut(self, shape):
        o = self.in_hand_conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, obs, in_hand):
        df_channel = 16
        cross = 2

        obs_2 = F.interpolate(obs, scale_factor=0.5, mode='bilinear', align_corners=False)

        pick_feature_map_1 = self.pick_conv1(obs)
        pick_feature_map_2 = F.interpolate(self.pick_conv2(obs_2), size=(obs.shape[2], obs.shape[3]), mode='bilinear', align_corners=False)
        pick_feature_map = torch.cat((pick_feature_map_1, pick_feature_map_2), dim=1)

        pick_domain_feature_map = self.pick_domain_down(pick_feature_map)
        pick_feature_flatten = pick_domain_feature_map.view(obs.size()[0], -1)

        pick_filter_weights = self.pick_filter_fc(pick_feature_flatten)
        pick_filter_weight = pick_filter_weights[:, :cross * df_channel * df_channel]
        pick_filter_bias = pick_filter_weights[:, cross * df_channel * df_channel:]
        pick_filter_weight = pick_filter_weight.reshape(obs.size()[0] * cross * df_channel, df_channel, 1, 1)
        pick_filter_bias = pick_filter_bias.reshape(obs.size()[0] * cross * df_channel)
        pick_feature_map = pick_feature_map.reshape(1, pick_feature_map.size(0) * pick_feature_map.size(1),
                                                    pick_feature_map.size(2), pick_feature_map.size(3))
        pick_feature_map = F.conv2d(pick_feature_map, weight=pick_filter_weight, bias=pick_filter_bias,
                                    groups=cross * obs.size(0))
        pick_feature_map = pick_feature_map.reshape(obs.size(0), -1, pick_feature_map.size(2), pick_feature_map.size(3))

        pick_q_values = self.pick_q_values(pick_feature_map)

        place_feature_map_1 = self.place_conv1(obs)
        place_feature_map_2 = F.interpolate(self.place_conv2(obs_2), size=(obs.shape[2], obs.shape[3]), mode='bilinear', align_corners=False)
        place_feature_map = torch.cat((place_feature_map_1, place_feature_map_2), dim=1)

        place_domain_feature_map = self.place_domain_down(place_feature_map)
        place_feature_flatten = place_domain_feature_map.view(obs.size()[0], -1)
        in_hand_out = self.in_hand_conv(in_hand)
        in_hand_flatten = in_hand_out.view(in_hand.size()[0], -1)
        place_filter_weights = self.place_filter_fc(torch.cat((place_feature_flatten, in_hand_flatten), dim=1))
        place_filter_weight = place_filter_weights[:, :cross * df_channel * df_channel]
        place_filter_bias = place_filter_weights[:, cross * df_channel * df_channel:]
        place_filter_weight = place_filter_weight.reshape(obs.size()[0] * cross * df_channel, df_channel, 1, 1)
        place_filter_bias = place_filter_bias.reshape(obs.size()[0] * cross * df_channel)
        place_feature_map = place_feature_map.reshape(1, place_feature_map.size(0) * place_feature_map.size(1),
                                                      place_feature_map.size(2), place_feature_map.size(3))
        place_feature_map = F.conv2d(place_feature_map, weight=place_filter_weight, bias=place_filter_bias,
                                     groups=cross * obs.size(0))
        place_feature_map = place_feature_map.reshape(obs.size(0), -1, place_feature_map.size(2),
                                                      place_feature_map.size(3))

        place_q_values = self.place_q_values(place_feature_map)

        q_values = torch.cat((pick_q_values, place_q_values), dim=1)

        if self.mask is not None:
            mask = self.mask.expand(q_values.size()).bool()
            mask = 1-mask
            q_values[mask] = -100
        return q_values

class FCNInHandDynamicFilterPyramid(nn.Module):
    def __init__(self, n_input_channel=1, n_primitives=1, patch_shape=(1, 24, 24), domain_shape=(1, 100, 100)):
        super(FCNInHandDynamicFilterPyramid, self).__init__()
        self.conv1 = nn.Sequential(
            (nn.Conv2d(n_input_channel, 32, kernel_size=11, stride=1, padding=5)),
            (nn.ReLU(inplace=True)),
            (nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2)),
            (nn.ReLU(inplace=True)),
            (nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)),
            (nn.ReLU(inplace=True)),
            (nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)),
            (nn.ReLU(inplace=True)),
            (nn.Conv2d(32, 16, kernel_size=1, stride=1)),
            (nn.ReLU(inplace=True))
        )
        self.conv2 = nn.Sequential(
            (nn.Conv2d(n_input_channel, 32, kernel_size=11, stride=1, padding=5)),
            (nn.ReLU(inplace=True)),
            (nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2)),
            (nn.ReLU(inplace=True)),
            (nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)),
            (nn.ReLU(inplace=True)),
            (nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)),
            (nn.ReLU(inplace=True)),
            (nn.Conv2d(32, 16, kernel_size=1, stride=1)),
            (nn.ReLU(inplace=True))
        )
        self.domain_down = nn.Sequential(
            (nn.Conv2d(32, 8, kernel_size=1, stride=1, bias=False)),
            (nn.MaxPool2d(2, 2)),
            (nn.ReLU(inplace=True))
        )
        self.in_hand_conv = nn.Sequential(
            (nn.Conv2d(patch_shape[0], 32, kernel_size=4, stride=2)),
            (nn.ReLU(inplace=True)),
            (nn.Conv2d(32, 64, kernel_size=4, stride=1)),
            (nn.ReLU(inplace=True)),
            (nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            (nn.ReLU(inplace=True)),
        )
        conv_out_size = self._getDomainConvOut(domain_shape) + self._getInHandConvOut(patch_shape)
        self.filter_fc = nn.Sequential(
            (nn.Linear(conv_out_size, 1024)),
            (nn.ReLU(inplace=True)),
            (nn.Linear(1024, 32 * 32 + 32))
        )
        self.q_values = nn.Conv2d(32, n_primitives, kernel_size=1, stride=1, bias=False)

        for m in self.named_modules():
            if isinstance(m[1], nn.Conv2d):
                # nn.init.kaiming_normal_(m[1].weight.data)
                nn.init.xavier_normal_(m[1].weight.data)
            elif isinstance(m[1], nn.BatchNorm2d):
                m[1].weight.data.fill_(1)
                m[1].bias.data.zero_()

        self.mask = None

    def setCandidatePos(self, candidate_pos, input_size=(100, 100)):
        if candidate_pos is None:
            self.mask = None
        else:
            self.mask = torch.zeros(input_size)
            index = np.array(np.meshgrid(candidate_pos[0], candidate_pos[1])).T.reshape(-1, 2)
            index = torch.tensor(index).long()
            for idx in index:
                self.mask[idx[0], idx[1]] = 1

    def _getDomainConvOut(self, shape):
        o = self.domain_down(torch.cat((self.conv1(torch.zeros(1, *shape)),
                                        self.conv2(torch.zeros(1, *shape))),
                                       dim=1))
        return int(np.prod(o.size()))

    def _getInHandConvOut(self, shape):
        o = self.in_hand_conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, obs, in_hand):
        df_channel = 32
        obs_2 = F.interpolate(obs, scale_factor=0.5, mode='bilinear', align_corners=False)

        feature_map_1 = self.conv1(obs)
        feature_map_2 = F.interpolate(self.conv2(obs_2), size=(obs.shape[2], obs.shape[3]), mode='bilinear', align_corners=False)
        feature_map = torch.cat((feature_map_1, feature_map_2), dim=1)

        domain_feature_map = self.domain_down(feature_map)
        feature_flatten = domain_feature_map.view(obs.size()[0], -1)
        in_hand_out = self.in_hand_conv(in_hand)
        in_hand_flatten = in_hand_out.view(in_hand.size()[0], -1)
        filter_weights = self.filter_fc(torch.cat((feature_flatten, in_hand_flatten), dim=1))
        filter_weight = filter_weights[:, :df_channel * df_channel].reshape(obs.size()[0], df_channel, df_channel, 1, 1)
        filter_bias = filter_weights[:, df_channel * df_channel:].reshape(obs.size()[0], df_channel)
        filter_weight = filter_weight.reshape(obs.size()[0] * df_channel, df_channel, 1, 1)
        filter_bias = filter_bias.reshape(obs.size()[0] * df_channel)
        feature_map = feature_map.reshape(1, feature_map.size(0) * feature_map.size(1),
                                          feature_map.size(2), feature_map.size(3))
        feature_map = F.conv2d(feature_map, weight=filter_weight, bias=filter_bias,
                               groups=obs.size(0))
        feature_map = feature_map.reshape(obs.size(0), -1, feature_map.size(2),
                                          feature_map.size(3))
        q_values = self.q_values(feature_map)

        if self.mask is not None:
            mask = self.mask.expand(q_values.size()).bool()
            mask = 1-mask
            q_values[mask] = -100
        return q_values

class FCNInHandDynamicFilterPyramidCross(nn.Module):
    def __init__(self, n_input_channel=1, n_primitives=1, patch_shape=(1, 24, 24), domain_shape=(1, 100, 100)):
        super(FCNInHandDynamicFilterPyramidCross, self).__init__()
        self.conv1 = nn.Sequential(
            (nn.Conv2d(n_input_channel, 32, kernel_size=11, stride=1, padding=5)),
            (nn.ReLU(inplace=True)),
            (nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2)),
            (nn.ReLU(inplace=True)),
            (nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)),
            (nn.ReLU(inplace=True)),
            (nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)),
            (nn.ReLU(inplace=True)),
            (nn.Conv2d(32, 16, kernel_size=1, stride=1)),
            (nn.ReLU(inplace=True))
        )
        self.conv2 = nn.Sequential(
            (nn.Conv2d(n_input_channel, 32, kernel_size=11, stride=1, padding=5)),
            (nn.ReLU(inplace=True)),
            (nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2)),
            (nn.ReLU(inplace=True)),
            (nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)),
            (nn.ReLU(inplace=True)),
            (nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)),
            (nn.ReLU(inplace=True)),
            (nn.Conv2d(32, 16, kernel_size=1, stride=1)),
            (nn.ReLU(inplace=True))
        )
        self.domain_down = nn.Sequential(
            (nn.Conv2d(32, 8, kernel_size=1, stride=1, bias=False)),
            (nn.MaxPool2d(2, 2)),
            (nn.ReLU(inplace=True))
        )
        self.in_hand_conv = nn.Sequential(
            (nn.Conv2d(patch_shape[0], 32, kernel_size=4, stride=2)),
            (nn.ReLU(inplace=True)),
            (nn.Conv2d(32, 64, kernel_size=4, stride=1)),
            (nn.ReLU(inplace=True)),
            (nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            (nn.ReLU(inplace=True)),
        )
        conv_out_size = self._getDomainConvOut(domain_shape) + self._getInHandConvOut(patch_shape)
        self.filter_fc = nn.Sequential(
            (nn.Linear(conv_out_size, 1024)),
            (nn.ReLU(inplace=True)),
            (nn.Linear(1024, 2 * (16 * 16 + 16)))
        )
        self.q_values = nn.Conv2d(32, n_primitives, kernel_size=1, stride=1, bias=False)

        for m in self.named_modules():
            if isinstance(m[1], nn.Conv2d):
                # nn.init.kaiming_normal_(m[1].weight.data)
                nn.init.xavier_normal_(m[1].weight.data)
            elif isinstance(m[1], nn.BatchNorm2d):
                m[1].weight.data.fill_(1)
                m[1].bias.data.zero_()

        self.mask = None

    def setCandidatePos(self, candidate_pos, input_size=(100, 100)):
        if candidate_pos is None:
            self.mask = None
        else:
            self.mask = torch.zeros(input_size)
            index = np.array(np.meshgrid(candidate_pos[0], candidate_pos[1])).T.reshape(-1, 2)
            index = torch.tensor(index).long()
            for idx in index:
                self.mask[idx[0], idx[1]] = 1

    def _getDomainConvOut(self, shape):
        o = self.domain_down(torch.cat((self.conv1(torch.zeros(1, *shape)),
                                        self.conv2(torch.zeros(1, *shape))),
                                       dim=1))
        return int(np.prod(o.size()))

    def _getInHandConvOut(self, shape):
        o = self.in_hand_conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, obs, in_hand):
        df_channel = 16
        cross = 2
        obs_2 = F.interpolate(obs, scale_factor=0.5, mode='bilinear', align_corners=False)

        feature_map_1 = self.conv1(obs)
        feature_map_2 = F.interpolate(self.conv2(obs_2), size=(obs.shape[2], obs.shape[3]), mode='bilinear', align_corners=False)
        feature_map = torch.cat((feature_map_1, feature_map_2), dim=1)

        domain_feature_map = self.domain_down(feature_map)
        feature_flatten = domain_feature_map.view(obs.size()[0], -1)
        in_hand_out = self.in_hand_conv(in_hand)
        in_hand_flatten = in_hand_out.view(in_hand.size()[0], -1)
        filter_weights = self.filter_fc(torch.cat((feature_flatten, in_hand_flatten), dim=1))
        filter_weight = filter_weights[:, :cross * df_channel * df_channel]
        filter_bias = filter_weights[:, cross * df_channel * df_channel:]
        filter_weight = filter_weight.reshape(obs.size()[0] * cross * df_channel, df_channel, 1, 1)
        filter_bias = filter_bias.reshape(obs.size()[0] * cross * df_channel)
        feature_map = feature_map.reshape(1, feature_map.size(0) * feature_map.size(1),
                                          feature_map.size(2), feature_map.size(3))
        feature_map = F.conv2d(feature_map, weight=filter_weight, bias=filter_bias,
                               groups=cross * obs.size(0))
        feature_map = feature_map.reshape(obs.size(0), -1, feature_map.size(2),
                                          feature_map.size(3))
        q_values = self.q_values(feature_map)

        if self.mask is not None:
            mask = self.mask.expand(q_values.size()).bool()
            mask = 1-mask
            q_values[mask] = -100
        return q_values

class FCNInHandDynamicFilterPyramidCrossSepHead(nn.Module):
    def __init__(self, n_input_channel=1, n_primitives=1, patch_shape=(1, 24, 24), domain_shape=(1, 100, 100)):
        super(FCNInHandDynamicFilterPyramidCrossSepHead, self).__init__()
        self.conv1 = nn.Sequential(
            (nn.Conv2d(n_input_channel, 32, kernel_size=11, stride=1, padding=5)),
            (nn.ReLU(inplace=True)),
            (nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2)),
            (nn.ReLU(inplace=True)),
            (nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)),
            (nn.ReLU(inplace=True)),
            (nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)),
            (nn.ReLU(inplace=True)),
            (nn.Conv2d(32, 16, kernel_size=1, stride=1)),
            (nn.ReLU(inplace=True))
        )
        self.conv2 = nn.Sequential(
            (nn.Conv2d(n_input_channel, 32, kernel_size=11, stride=1, padding=5)),
            (nn.ReLU(inplace=True)),
            (nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2)),
            (nn.ReLU(inplace=True)),
            (nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)),
            (nn.ReLU(inplace=True)),
            (nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)),
            (nn.ReLU(inplace=True)),
            (nn.Conv2d(32, 16, kernel_size=1, stride=1)),
            (nn.ReLU(inplace=True))
        )
        self.domain_down = nn.Sequential(
            (nn.Conv2d(32, 8, kernel_size=1, stride=1, bias=False)),
            (nn.MaxPool2d(2, 2)),
            (nn.ReLU(inplace=True))
        )
        self.in_hand_conv = nn.Sequential(
            (nn.Conv2d(patch_shape[0], 32, kernel_size=4, stride=2)),
            (nn.ReLU(inplace=True)),
            (nn.Conv2d(32, 64, kernel_size=4, stride=1)),
            (nn.ReLU(inplace=True)),
            (nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            (nn.ReLU(inplace=True)),
        )

        out_size = self._getDomainConvOut(domain_shape) + self._getInHandConvOut(patch_shape)
        self.filter_fc = nn.Sequential(
            (nn.Linear(out_size, 1024)),
            (nn.ReLU(inplace=True)),
            (nn.Linear(1024, 2 * 2 * (16 * 16 + 16)))
        )

        self.pick_q_values = nn.Conv2d(32, 1, kernel_size=1, stride=1, bias=False)
        self.place_q_values = nn.Conv2d(32, 1, kernel_size=1, stride=1, bias=False)

        for m in self.named_modules():
            if isinstance(m[1], nn.Conv2d):
                # nn.init.kaiming_normal_(m[1].weight.data)
                nn.init.xavier_normal_(m[1].weight.data)
            elif isinstance(m[1], nn.BatchNorm2d):
                m[1].weight.data.fill_(1)
                m[1].bias.data.zero_()

        self.mask = None

    def setCandidatePos(self, candidate_pos, input_size=(100, 100)):
        if candidate_pos is None:
            self.mask = None
        else:
            self.mask = torch.zeros(input_size)
            index = np.array(np.meshgrid(candidate_pos[0], candidate_pos[1])).T.reshape(-1, 2)
            index = torch.tensor(index).long()
            for idx in index:
                self.mask[idx[0], idx[1]] = 1

    def _getDomainConvOut(self, shape):
        o = self.domain_down(torch.cat((self.conv1(torch.zeros(1, *shape)),
                                        self.conv2(torch.zeros(1, *shape))),
                                       dim=1))
        return int(np.prod(o.size()))

    def _getInHandConvOut(self, shape):
        o = self.in_hand_conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, obs, in_hand):
        df_channel = 16
        cross = 2
        obs_2 = F.interpolate(obs, scale_factor=0.5, mode='bilinear', align_corners=False)

        feature_map_1 = self.conv1(obs)
        feature_map_2 = F.interpolate(self.conv2(obs_2), size=(obs.shape[2], obs.shape[3]), mode='bilinear', align_corners=False)
        feature_map = torch.cat((feature_map_1, feature_map_2), dim=1)

        domain_feature_map = self.domain_down(feature_map)
        feature_flatten = domain_feature_map.view(obs.size()[0], -1)
        in_hand_out = self.in_hand_conv(in_hand)
        in_hand_flatten = in_hand_out.view(in_hand.size()[0], -1)

        feature_map = feature_map.reshape(1, feature_map.size(0) * feature_map.size(1),
                                          feature_map.size(2), feature_map.size(3))
        filter_weights = self.filter_fc(torch.cat((feature_flatten, in_hand_flatten), dim=1))

        pick_filter_weights = filter_weights[:, :cross * (df_channel * df_channel + df_channel)]
        pick_filter_weight = pick_filter_weights[:, :cross * df_channel * df_channel]
        pick_filter_bias = pick_filter_weights[:, cross * df_channel * df_channel:]
        pick_filter_weight = pick_filter_weight.reshape(obs.size()[0] * cross * df_channel, df_channel, 1, 1)
        pick_filter_bias = pick_filter_bias.reshape(obs.size()[0] * cross * df_channel)

        place_filter_weights = filter_weights[:, cross * (df_channel * df_channel + df_channel):]
        place_filter_weight = place_filter_weights[:, :cross * df_channel * df_channel]
        place_filter_bias = place_filter_weights[:, cross * df_channel * df_channel:]
        place_filter_weight = place_filter_weight.reshape(obs.size()[0] * cross * df_channel, df_channel, 1, 1)
        place_filter_bias = place_filter_bias.reshape(obs.size()[0] * cross * df_channel)

        pick_feature_map = F.conv2d(feature_map, weight=pick_filter_weight, bias=pick_filter_bias,
                                    groups=cross * obs.size(0))
        pick_feature_map = pick_feature_map.reshape(obs.size(0), -1, feature_map.size(2),
                                                    feature_map.size(3))
        pick_q_values = self.pick_q_values(pick_feature_map)
        place_feature_map = F.conv2d(feature_map, weight=place_filter_weight, bias=place_filter_bias,
                                     groups=cross * obs.size(0))
        place_feature_map = place_feature_map.reshape(obs.size(0), -1, feature_map.size(2),
                                                      feature_map.size(3))
        place_q_values = self.place_q_values(place_feature_map)

        q_values = torch.cat((pick_q_values, place_q_values), dim=1)

        if self.mask is not None:
            mask = self.mask.expand(q_values.size()).bool()
            mask = 1-mask
            q_values[mask] = -100
        return q_values

class FCNInHandDynamicFilterPyramidCrossSepHead1(nn.Module):
    def __init__(self, n_input_channel=1, n_primitives=1, patch_shape=(1, 24, 24), domain_shape=(1, 100, 100)):
        super(FCNInHandDynamicFilterPyramidCrossSepHead1, self).__init__()
        self.conv1 = nn.Sequential(
            (nn.Conv2d(n_input_channel, 32, kernel_size=11, stride=1, padding=5)),
            (nn.ReLU(inplace=True)),
            (nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2)),
            (nn.ReLU(inplace=True)),
            (nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)),
            (nn.ReLU(inplace=True)),
            (nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)),
            (nn.ReLU(inplace=True)),
            (nn.Conv2d(32, 16, kernel_size=1, stride=1)),
            (nn.ReLU(inplace=True))
        )
        self.conv2 = nn.Sequential(
            (nn.Conv2d(n_input_channel, 32, kernel_size=11, stride=1, padding=5)),
            (nn.ReLU(inplace=True)),
            (nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2)),
            (nn.ReLU(inplace=True)),
            (nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)),
            (nn.ReLU(inplace=True)),
            (nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)),
            (nn.ReLU(inplace=True)),
            (nn.Conv2d(32, 16, kernel_size=1, stride=1)),
            (nn.ReLU(inplace=True))
        )
        self.domain_down = nn.Sequential(
            (nn.Conv2d(32, 8, kernel_size=1, stride=1, bias=False)),
            (nn.MaxPool2d(2, 2)),
            (nn.ReLU(inplace=True))
        )
        self.in_hand_conv = nn.Sequential(
            (nn.Conv2d(patch_shape[0], 32, kernel_size=4, stride=2)),
            (nn.ReLU(inplace=True)),
            (nn.Conv2d(32, 64, kernel_size=4, stride=1)),
            (nn.ReLU(inplace=True)),
            (nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            (nn.ReLU(inplace=True)),
        )

        out_size = self._getDomainConvOut(domain_shape) + self._getInHandConvOut(patch_shape)
        self.filter_fc = nn.Sequential(
            (nn.Linear(out_size, 1024)),
            (nn.ReLU(inplace=True)),
        )

        self.pick_filter_head = nn.Linear(1024, 2 * (16 * 16 + 16))
        self.place_filter_head = nn.Linear(1024, 2 * (16 * 16 + 16))

        self.pick_q_values = nn.Conv2d(32, 1, kernel_size=1, stride=1, bias=False)
        self.place_q_values = nn.Conv2d(32, 1, kernel_size=1, stride=1, bias=False)

        for m in self.named_modules():
            if isinstance(m[1], nn.Conv2d):
                # nn.init.kaiming_normal_(m[1].weight.data)
                nn.init.xavier_normal_(m[1].weight.data)
            elif isinstance(m[1], nn.BatchNorm2d):
                m[1].weight.data.fill_(1)
                m[1].bias.data.zero_()

        self.mask = None

    def setCandidatePos(self, candidate_pos, input_size=(100, 100)):
        if candidate_pos is None:
            self.mask = None
        else:
            self.mask = torch.zeros(input_size)
            index = np.array(np.meshgrid(candidate_pos[0], candidate_pos[1])).T.reshape(-1, 2)
            index = torch.tensor(index).long()
            for idx in index:
                self.mask[idx[0], idx[1]] = 1

    def _getDomainConvOut(self, shape):
        o = self.domain_down(torch.cat((self.conv1(torch.zeros(1, *shape)),
                                        self.conv2(torch.zeros(1, *shape))),
                                       dim=1))
        return int(np.prod(o.size()))

    def _getInHandConvOut(self, shape):
        o = self.in_hand_conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, obs, in_hand):
        df_channel = 16
        cross = 2
        obs_2 = F.interpolate(obs, scale_factor=0.5, mode='bilinear', align_corners=False)

        feature_map_1 = self.conv1(obs)
        feature_map_2 = F.interpolate(self.conv2(obs_2), size=(obs.shape[2], obs.shape[3]), mode='bilinear', align_corners=False)
        feature_map = torch.cat((feature_map_1, feature_map_2), dim=1)

        domain_feature_map = self.domain_down(feature_map)
        feature_flatten = domain_feature_map.view(obs.size()[0], -1)
        in_hand_out = self.in_hand_conv(in_hand)
        in_hand_flatten = in_hand_out.view(in_hand.size()[0], -1)

        feature_map = feature_map.reshape(1, feature_map.size(0) * feature_map.size(1),
                                          feature_map.size(2), feature_map.size(3))
        feature_down = self.filter_fc(torch.cat((feature_flatten, in_hand_flatten), dim=1))
        pick_filter_weights = self.pick_filter_head(feature_down)
        place_filter_weights = self.place_filter_head(feature_down)

        pick_filter_weight = pick_filter_weights[:, :cross * df_channel * df_channel]
        pick_filter_bias = pick_filter_weights[:, cross * df_channel * df_channel:]
        pick_filter_weight = pick_filter_weight.reshape(obs.size()[0] * cross * df_channel, df_channel, 1, 1)
        pick_filter_bias = pick_filter_bias.reshape(obs.size()[0] * cross * df_channel)

        place_filter_weight = place_filter_weights[:, :cross * df_channel * df_channel]
        place_filter_bias = place_filter_weights[:, cross * df_channel * df_channel:]
        place_filter_weight = place_filter_weight.reshape(obs.size()[0] * cross * df_channel, df_channel, 1, 1)
        place_filter_bias = place_filter_bias.reshape(obs.size()[0] * cross * df_channel)

        pick_feature_map = F.conv2d(feature_map, weight=pick_filter_weight, bias=pick_filter_bias,
                                    groups=cross * obs.size(0))
        pick_feature_map = pick_feature_map.reshape(obs.size(0), -1, feature_map.size(2),
                                                    feature_map.size(3))
        pick_q_values = self.pick_q_values(pick_feature_map)
        place_feature_map = F.conv2d(feature_map, weight=place_filter_weight, bias=place_filter_bias,
                                     groups=cross * obs.size(0))
        place_feature_map = place_feature_map.reshape(obs.size(0), -1, feature_map.size(2),
                                                      feature_map.size(3))
        place_q_values = self.place_q_values(place_feature_map)

        q_values = torch.cat((pick_q_values, place_q_values), dim=1)

        if self.mask is not None:
            mask = self.mask.expand(q_values.size()).bool()
            mask = 1-mask
            q_values[mask] = -100
        return q_values

class FCNDynamicFilter(nn.Module):
    def __init__(self, n_input_channel=1, n_primitives=1, domain_shape=(1, 100, 100)):
        super(FCNDynamicFilter, self).__init__()
        self.domain_conv = nn.Sequential(OrderedDict([
            ('fcn_conv1', nn.Conv2d(n_input_channel, 32, kernel_size=5, stride=1, padding=2)),
            ('fcn_relu1', nn.ReLU(inplace=True)),
            ('fcn_conv2', nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)),
            ('fcn_relu2', nn.ReLU(inplace=True)),
            ('fcn_conv3', nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)),
            ('fcn_relu3', nn.ReLU(inplace=True)),
            ('fcn_conv4', nn.Conv2d(64, 16, kernel_size=1, stride=1, bias=False)),
            ('fcn_relu4', nn.ReLU(inplace=True)),
        ]))

        self.domain_down = nn.Sequential(OrderedDict([
            ('df_conv1', nn.Conv2d(16, 8, kernel_size=1, stride=1, bias=False)),
            ('df_pool1', nn.MaxPool2d(2, 2)),
            ('df_relu1', nn.ReLU(inplace=True))
        ]))

        self.q_values = nn.Conv2d(16, n_primitives, kernel_size=1, stride=1, bias=False)
        conv_out_size = self._getDomainConvOut(domain_shape)
        self.filter_fc = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(conv_out_size, 1024)),
            ('fc_relu1', nn.ReLU(inplace=True)),
            ('fc2', nn.Linear(1024, 16*16+16))
        ]))

        for m in self.named_modules():
            if isinstance(m[1], nn.Conv2d):
                # nn.init.kaiming_normal_(m[1].weight.data)
                nn.init.xavier_normal_(m[1].weight.data)
            elif isinstance(m[1], nn.BatchNorm2d):
                m[1].weight.data.fill_(1)
                m[1].bias.data.zero_()

        self.mask = None

    def setCandidatePos(self, candidate_pos, input_size=(100, 100)):
        if candidate_pos is None:
            self.mask = None
        else:
            self.mask = torch.zeros(input_size)
            index = np.array(np.meshgrid(candidate_pos[0], candidate_pos[1])).T.reshape(-1, 2)
            index = torch.tensor(index).long()
            for idx in index:
                self.mask[idx[0], idx[1]] = 1

    def _getDomainConvOut(self, shape):
        o = self.domain_down(self.domain_conv(torch.zeros(1, *shape)))
        return int(np.prod(o.size()))

    def forward(self, obs):
        feature_map = self.domain_conv(obs)
        domain_feature_map = self.domain_down(feature_map)
        feature_flatten = domain_feature_map.view(obs.size()[0], -1)

        filter_weights = self.filter_fc(feature_flatten)
        filter_weight = filter_weights[:, :16*16].reshape(obs.size()[0], 16, 16, 1, 1)
        filter_bias = filter_weights[:, 16*16:].reshape(obs.size()[0], 16)

        filter_weight = filter_weight.reshape(obs.size()[0]*16, 16, 1, 1)
        filter_bias = filter_bias.reshape(obs.size()[0]*16)
        feature_map = feature_map.reshape(1, feature_map.size(0)*feature_map.size(1), feature_map.size(2), feature_map.size(3))
        feature_map = F.conv2d(feature_map, weight=filter_weight, bias=filter_bias, groups=obs.size(0))
        feature_map = feature_map.reshape(obs.size(0), -1, feature_map.size(2), feature_map.size(3))

        q_values = self.q_values(feature_map)
        # deconv_q_values = F.interpolate(q_values, size=(obs.size(2), obs.size(3)), mode='bilinear', align_corners=False)

        if self.mask is not None:
            mask = self.mask.expand(q_values.size()).bool()
            mask = 1-mask
            q_values[mask] = -100
        return q_values

class FCNDynamicFilterSep(nn.Module):
    def __init__(self, n_input_channel=1, n_primitives=1, domain_shape=(1, 100, 100)):
        super(FCNDynamicFilterSep, self).__init__()
        self.pick_domain_conv = nn.Sequential(OrderedDict([
            ('pick_fcn_conv1', nn.Conv2d(n_input_channel, 32, kernel_size=5, stride=1, padding=2)),
            ('pick_fcn_relu1', nn.ReLU(inplace=True)),
            ('pick_fcn_conv2', nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)),
            ('pick_fcn_relu2', nn.ReLU(inplace=True)),
            ('pick_fcn_conv3', nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)),
            ('pick_fcn_relu3', nn.ReLU(inplace=True)),
            ('pick_fcn_conv4', nn.Conv2d(64, 16, kernel_size=1, stride=1, bias=False)),
            ('pick_fcn_relu4', nn.ReLU(inplace=True)),
        ]))
        self.pick_domain_down = nn.Sequential(OrderedDict([
            ('pick_df_conv1', nn.Conv2d(16, 8, kernel_size=1, stride=1, bias=False)),
            ('pick_df_pool1', nn.MaxPool2d(2, 2)),
            ('pick_df_relu1', nn.ReLU(inplace=True))
        ]))
        conv_out_size = self._getDomainConvOut(domain_shape)
        self.pick_filter_fc = nn.Sequential(OrderedDict([
            ('pick_fc1', nn.Linear(conv_out_size, 1024)),
            ('pick_fc_relu1', nn.ReLU(inplace=True)),
            ('pick_fc2', nn.Linear(1024, 16*16+16))
        ]))
        self.pick_q_values = nn.Conv2d(16, 1, kernel_size=1, stride=1, bias=False)

        self.place_domain_conv = nn.Sequential(OrderedDict([
            ('place_fcn_conv1', nn.Conv2d(n_input_channel, 32, kernel_size=5, stride=1, padding=2)),
            ('place_fcn_relu1', nn.ReLU(inplace=True)),
            ('place_fcn_conv2', nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)),
            ('place_fcn_relu2', nn.ReLU(inplace=True)),
            ('place_fcn_conv3', nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)),
            ('place_fcn_relu3', nn.ReLU(inplace=True)),
            ('place_fcn_conv4', nn.Conv2d(64, 16, kernel_size=1, stride=1, bias=False)),
            ('place_fcn_relu4', nn.ReLU(inplace=True)),
        ]))
        self.place_domain_down = nn.Sequential(OrderedDict([
            ('place_df_conv1', nn.Conv2d(16, 8, kernel_size=1, stride=1, bias=False)),
            ('place_df_pool1', nn.MaxPool2d(2, 2)),
            ('place_df_relu1', nn.ReLU(inplace=True))
        ]))
        conv_out_size = self._getDomainConvOut(domain_shape)
        self.place_filter_fc = nn.Sequential(OrderedDict([
            ('place_fc1', nn.Linear(conv_out_size, 1024)),
            ('place_fc_relu1', nn.ReLU(inplace=True)),
            ('place_fc2', nn.Linear(1024, 16 * 16 + 16))
        ]))
        self.place_q_values = nn.Conv2d(16, 1, kernel_size=1, stride=1, bias=False)


        for m in self.named_modules():
            if isinstance(m[1], nn.Conv2d):
                # nn.init.kaiming_normal_(m[1].weight.data)
                nn.init.xavier_normal_(m[1].weight.data)
            elif isinstance(m[1], nn.BatchNorm2d):
                m[1].weight.data.fill_(1)
                m[1].bias.data.zero_()

        self.mask = None

    def setCandidatePos(self, candidate_pos, input_size=(100, 100)):
        if candidate_pos is None:
            self.mask = None
        else:
            self.mask = torch.zeros(input_size)
            index = np.array(np.meshgrid(candidate_pos[0], candidate_pos[1])).T.reshape(-1, 2)
            index = torch.tensor(index).long()
            for idx in index:
                self.mask[idx[0], idx[1]] = 1

    def _getDomainConvOut(self, shape):
        o = self.pick_domain_down(self.pick_domain_conv(torch.zeros(1, *shape)))
        return int(np.prod(o.size()))

    def forward(self, obs):
        pick_feature_map = self.pick_domain_conv(obs)
        pick_domain_feature_map = self.pick_domain_down(pick_feature_map)
        pick_feature_flatten = pick_domain_feature_map.view(obs.size()[0], -1)

        pick_filter_weights = self.pick_filter_fc(pick_feature_flatten)
        pick_filter_weight = pick_filter_weights[:, :16*16].reshape(obs.size()[0], 16, 16, 1, 1)
        pick_filter_bias = pick_filter_weights[:, 16*16:].reshape(obs.size()[0], 16)

        pick_filter_weight = pick_filter_weight.reshape(obs.size()[0]*16, 16, 1, 1)
        pick_filter_bias = pick_filter_bias.reshape(obs.size()[0]*16)
        pick_feature_map = pick_feature_map.reshape(1, pick_feature_map.size(0)*pick_feature_map.size(1), pick_feature_map.size(2), pick_feature_map.size(3))
        pick_feature_map = F.conv2d(pick_feature_map, weight=pick_filter_weight, bias=pick_filter_bias, groups=obs.size(0))
        pick_feature_map = pick_feature_map.reshape(obs.size(0), -1, pick_feature_map.size(2), pick_feature_map.size(3))

        pick_q_values = self.pick_q_values(pick_feature_map)

        place_feature_map = self.place_domain_conv(obs)
        place_domain_feature_map = self.place_domain_down(place_feature_map)
        place_feature_flatten = place_domain_feature_map.view(obs.size()[0], -1)

        place_filter_weights = self.place_filter_fc(place_feature_flatten)
        place_filter_weight = place_filter_weights[:, :16 * 16].reshape(obs.size()[0], 16, 16, 1, 1)
        place_filter_bias = place_filter_weights[:, 16 * 16:].reshape(obs.size()[0], 16)

        place_filter_weight = place_filter_weight.reshape(obs.size()[0] * 16, 16, 1, 1)
        place_filter_bias = place_filter_bias.reshape(obs.size()[0] * 16)
        place_feature_map = place_feature_map.reshape(1, place_feature_map.size(0) * place_feature_map.size(1),
                                                    place_feature_map.size(2), place_feature_map.size(3))
        place_feature_map = F.conv2d(place_feature_map, weight=place_filter_weight, bias=place_filter_bias,
                                    groups=obs.size(0))
        place_feature_map = place_feature_map.reshape(obs.size(0), -1, place_feature_map.size(2), place_feature_map.size(3))

        place_q_values = self.place_q_values(place_feature_map)

        q_values = torch.cat((pick_q_values, place_q_values), dim=1)
        # deconv_q_values = F.interpolate(q_values, size=(obs.size(2), obs.size(3)), mode='bilinear', align_corners=False)

        if self.mask is not None:
            mask = self.mask.expand(q_values.size()).bool()
            mask = 1-mask
            q_values[mask] = -100
        return q_values

class FCNDynamicFilterSepX(nn.Module):
    def __init__(self, n_input_channel=1, n_primitives=1, domain_shape=(1, 100, 100)):
        super(FCNDynamicFilterSepX, self).__init__()
        self.pick_domain_conv = nn.Sequential(OrderedDict([
            ('pick_fcn_conv1', nn.Conv2d(n_input_channel, 32, kernel_size=11, stride=1, padding=5)),
            ('pick_fcn_relu1', nn.ReLU(inplace=True)),
            ('pick_fcn_conv2', nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2)),
            ('pick_fcn_relu2', nn.ReLU(inplace=True)),
            ('pick_fcn_conv3', nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)),
            ('pick_fcn_relu3', nn.ReLU(inplace=True)),
            ('pick_fcn_conv4', nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)),
            ('pick_fcn_relu4', nn.ReLU(inplace=True)),
            ('pick_fcn_conv5', nn.Conv2d(32, 16, kernel_size=1, stride=1)),
            ('pick_fcn_relu5', nn.ReLU(inplace=True))
        ]))
        self.pick_domain_down = nn.Sequential(OrderedDict([
            ('pick_df_conv1', nn.Conv2d(16, 8, kernel_size=1, stride=1, bias=False)),
            ('pick_df_pool1', nn.MaxPool2d(2, 2)),
            ('pick_df_relu1', nn.ReLU(inplace=True))
        ]))
        conv_out_size = self._getDomainConvOut(domain_shape)
        self.pick_filter_fc = nn.Sequential(OrderedDict([
            ('pick_fc1', nn.Linear(conv_out_size, 1024)),
            ('pick_fc_relu1', nn.ReLU(inplace=True)),
            ('pick_fc2', nn.Linear(1024, 16*16+16))
        ]))
        self.pick_q_values = nn.Conv2d(16, 1, kernel_size=1, stride=1, bias=False)

        self.place_domain_conv = nn.Sequential(OrderedDict([
            ('place_fcn_conv1', nn.Conv2d(n_input_channel, 32, kernel_size=11, stride=1, padding=5)),
            ('place_fcn_relu1', nn.ReLU(inplace=True)),
            ('place_fcn_conv2', nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2)),
            ('place_fcn_relu2', nn.ReLU(inplace=True)),
            ('place_fcn_conv3', nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)),
            ('place_fcn_relu3', nn.ReLU(inplace=True)),
            ('place_fcn_conv4', nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)),
            ('place_fcn_relu4', nn.ReLU(inplace=True)),
            ('place_fcn_conv5', nn.Conv2d(32, 16, kernel_size=1, stride=1)),
            ('place_fcn_relu5', nn.ReLU(inplace=True)),
        ]))
        self.place_domain_down = nn.Sequential(OrderedDict([
            ('place_df_conv1', nn.Conv2d(16, 8, kernel_size=1, stride=1, bias=False)),
            ('place_df_pool1', nn.MaxPool2d(2, 2)),
            ('place_df_relu1', nn.ReLU(inplace=True))
        ]))
        conv_out_size = self._getDomainConvOut(domain_shape)
        self.place_filter_fc = nn.Sequential(OrderedDict([
            ('place_fc1', nn.Linear(conv_out_size, 1024)),
            ('place_fc_relu1', nn.ReLU(inplace=True)),
            ('place_fc2', nn.Linear(1024, 16 * 16 + 16))
        ]))
        self.place_q_values = nn.Conv2d(16, 1, kernel_size=1, stride=1, bias=False)


        for m in self.named_modules():
            if isinstance(m[1], nn.Conv2d):
                # nn.init.kaiming_normal_(m[1].weight.data)
                nn.init.xavier_normal_(m[1].weight.data)
            elif isinstance(m[1], nn.BatchNorm2d):
                m[1].weight.data.fill_(1)
                m[1].bias.data.zero_()

        self.mask = None

    def setCandidatePos(self, candidate_pos, input_size=(100, 100)):
        if candidate_pos is None:
            self.mask = None
        else:
            self.mask = torch.zeros(input_size)
            index = np.array(np.meshgrid(candidate_pos[0], candidate_pos[1])).T.reshape(-1, 2)
            index = torch.tensor(index).long()
            for idx in index:
                self.mask[idx[0], idx[1]] = 1

    def _getDomainConvOut(self, shape):
        o = self.pick_domain_down(self.pick_domain_conv(torch.zeros(1, *shape)))
        return int(np.prod(o.size()))

    def forward(self, obs):
        pick_feature_map = self.pick_domain_conv(obs)
        pick_domain_feature_map = self.pick_domain_down(pick_feature_map)
        pick_feature_flatten = pick_domain_feature_map.view(obs.size()[0], -1)

        pick_filter_weights = self.pick_filter_fc(pick_feature_flatten)
        pick_filter_weight = pick_filter_weights[:, :16*16].reshape(obs.size()[0], 16, 16, 1, 1)
        pick_filter_bias = pick_filter_weights[:, 16*16:].reshape(obs.size()[0], 16)

        pick_filter_weight = pick_filter_weight.reshape(obs.size()[0]*16, 16, 1, 1)
        pick_filter_bias = pick_filter_bias.reshape(obs.size()[0]*16)
        pick_feature_map = pick_feature_map.reshape(1, pick_feature_map.size(0)*pick_feature_map.size(1), pick_feature_map.size(2), pick_feature_map.size(3))
        pick_feature_map = F.conv2d(pick_feature_map, weight=pick_filter_weight, bias=pick_filter_bias, groups=obs.size(0))
        pick_feature_map = pick_feature_map.reshape(obs.size(0), -1, pick_feature_map.size(2), pick_feature_map.size(3))

        pick_q_values = self.pick_q_values(pick_feature_map)

        place_feature_map = self.place_domain_conv(obs)
        place_domain_feature_map = self.place_domain_down(place_feature_map)
        place_feature_flatten = place_domain_feature_map.view(obs.size()[0], -1)

        place_filter_weights = self.place_filter_fc(place_feature_flatten)
        place_filter_weight = place_filter_weights[:, :16 * 16].reshape(obs.size()[0], 16, 16, 1, 1)
        place_filter_bias = place_filter_weights[:, 16 * 16:].reshape(obs.size()[0], 16)

        place_filter_weight = place_filter_weight.reshape(obs.size()[0] * 16, 16, 1, 1)
        place_filter_bias = place_filter_bias.reshape(obs.size()[0] * 16)
        place_feature_map = place_feature_map.reshape(1, place_feature_map.size(0) * place_feature_map.size(1),
                                                    place_feature_map.size(2), place_feature_map.size(3))
        place_feature_map = F.conv2d(place_feature_map, weight=place_filter_weight, bias=place_filter_bias,
                                    groups=obs.size(0))
        place_feature_map = place_feature_map.reshape(obs.size(0), -1, place_feature_map.size(2), place_feature_map.size(3))

        place_q_values = self.place_q_values(place_feature_map)

        q_values = torch.cat((pick_q_values, place_q_values), dim=1)
        # deconv_q_values = F.interpolate(q_values, size=(obs.size(2), obs.size(3)), mode='bilinear', align_corners=False)

        if self.mask is not None:
            mask = self.mask.expand(q_values.size()).bool()
            mask = 1-mask
            q_values[mask] = -100
        return q_values


# class FCNSmallTest(nn.Module):
#     def __init__(self, n_input_channel=1, n_primitives=1, patch_shape=(1, 24, 24), n_in_hand_feature=10):
#         super(FCNSmallTest, self).__init__()
#
#         self.fcn_conv = nn.Sequential(OrderedDict([
#             ('fcn_conv1', nn.Conv2d(n_input_channel, 32, kernel_size=5, stride=1)),
#             ('fcn_bn1', nn.BatchNorm2d(32)),
#             ('fcn_relu1', nn.ReLU(inplace=True)),
#             ('fcn_conv2', nn.Conv2d(32, 64, kernel_size=3, stride=1)),
#             ('fcn_bn2', nn.BatchNorm2d(64)),
#             ('fcn_relu2', nn.ReLU(inplace=True)),
#             ('fcn_conv3', nn.Conv2d(64, 128, kernel_size=3, stride=1)),
#             ('fcn_bn3', nn.BatchNorm2d(128)),
#             ('fcn_relu3', nn.ReLU(inplace=True)),
#             ('fcn_conv4', nn.Conv2d(128, 64, kernel_size=1, stride=1, bias=False)),
#             ('fcn_bn4', nn.BatchNorm2d(64)),
#             ('fcn_relu4', nn.ReLU(inplace=True)),
#         ]))
#
#         self.q_values = nn.Conv2d(64+n_in_hand_feature, n_primitives, kernel_size=1, stride=1, bias=False)
#
#         self.cnn_conv = nn.Sequential(OrderedDict([
#             ('cnn_conv1', nn.Conv2d(patch_shape[0], 32, kernel_size=7, stride=4)),
#             ('cnn_relu1', nn.ReLU(inplace=True)),
#             ('cnn_conv2', nn.Conv2d(32, 64, kernel_size=5, stride=2)),
#             ('cnn_relu2', nn.ReLU(inplace=True)),
#             ('cnn_conv3', nn.Conv2d(64, 64, kernel_size=3, stride=1)),
#             ('cnn_relu3', nn.ReLU(inplace=True)),
#         ]))
#         conv_out_size = self._getConvOut(patch_shape)
#         self.fc1 = nn.Linear(conv_out_size, 512)
#         self.fc2 = nn.Linear(512, n_in_hand_feature)
#
#         for m in self.named_modules():
#             if isinstance(m[1], nn.Conv2d):
#                 # nn.init.kaiming_normal_(m[1].weight.data)
#                 nn.init.xavier_normal_(m[1].weight.data)
#             elif isinstance(m[1], nn.BatchNorm2d):
#                 m[1].weight.data.fill_(1)
#                 m[1].bias.data.zero_()
#
#         self.mask = None
#
#     def setCandidatePos(self, candidate_pos, input_size=(100, 100)):
#         if candidate_pos is None:
#             self.mask = None
#         else:
#             self.mask = torch.zeros(input_size)
#             index = np.array(np.meshgrid(candidate_pos[0], candidate_pos[1])).T.reshape(-1, 2)
#             index = torch.tensor(index).long()
#             for idx in index:
#                 self.mask[idx[0], idx[1]] = 1
#
#     def _getConvOut(self, shape):
#         o = self.cnn_conv(torch.zeros(1, *shape))
#         return int(np.prod(o.size()))
#
#     def forward(self, obs):
#         feature_map = self.fcn_conv(obs)
#
#         feature_out = self.cnn_conv(obs)
#         feature_out = feature_out.view(obs.size()[0], -1)
#         feature_out = self.fc1(feature_out)
#         feature_out = F.relu(feature_out)
#         feature_out = self.fc2(feature_out)
#         feature_out = feature_out.reshape(feature_out.size(0), feature_out.size(1), 1, 1)
#         feature_out = feature_out.expand(feature_out.size(0), feature_out.size(1), feature_map.size(2), feature_map.size(3))
#
#         feature_map = torch.cat((feature_map, feature_out), dim=1)
#
#         q_values = self.q_values(feature_map)
#         deconv_q_values = F.interpolate(q_values, size=(obs.size(2), obs.size(3)), mode='bilinear', align_corners=False)
#
#         if self.mask is not None:
#             mask = self.mask.expand(deconv_q_values.size()).bool()
#             mask = 1-mask
#             deconv_q_values[mask] = -100
#         return deconv_q_values

class FCNDomainFCSmall(nn.Module):
    def __init__(self, n_input_channel=1, n_primitives=1, domain_shape=(1, 100, 100), n_domain_feature=10):
        super(FCNDomainFCSmall, self).__init__()

        self.conv_net = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(n_input_channel, 32, kernel_size=5, stride=1)),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv2', nn.Conv2d(32, 64, kernel_size=3, stride=1)),
            ('relu2', nn.ReLU(inplace=True)),
            ('conv3', nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            ('relu3', nn.ReLU(inplace=True)),
            ('conv4', nn.Conv2d(64, 16, kernel_size=1, stride=1, bias=False)),
            ('relu4', nn.ReLU(inplace=True)),
        ]))

        self.domain_features = nn.Sequential(OrderedDict([
            ('df_conv1', nn.Conv2d(16, 8, kernel_size=1, stride=1, bias=False)),
            ('df_pool1', nn.MaxPool2d(2, 2)),
            ('df_relu1', nn.ReLU(inplace=True))
        ]))

        conv_out_size = self._getConvOut(domain_shape)
        self.fc1 = nn.Linear(conv_out_size, 512)
        self.fc2 = nn.Linear(512, n_domain_feature)

        self.q_values = nn.Conv2d(16 + n_domain_feature, n_primitives, kernel_size=1, stride=1, bias=False)

        for m in self.named_modules():
            if isinstance(m[1], nn.Conv2d):
                # nn.init.kaiming_normal_(m[1].weight.data)
                nn.init.xavier_normal_(m[1].weight.data)
            elif isinstance(m[1], nn.BatchNorm2d):
                m[1].weight.data.fill_(1)
                m[1].bias.data.zero_()

        self.mask = None

    def setCandidatePos(self, candidate_pos, input_size=(100, 100)):
        if candidate_pos is None:
            self.mask = None
        else:
            self.mask = torch.zeros(input_size)
            index = np.array(np.meshgrid(candidate_pos[0], candidate_pos[1])).T.reshape(-1, 2)
            index = torch.tensor(index).long()
            for idx in index:
                self.mask[idx[0], idx[1]] = 1

    def _getConvOut(self, shape):
        o = self.conv_net(torch.zeros(1, *shape))
        o = self.domain_features(o)
        return int(np.prod(o.size()))

    def forward(self, obs):
        feature_map = self.conv_net(obs)

        domain_feature_map = self.domain_features(feature_map)
        feature_flatten = domain_feature_map.view(obs.size()[0], -1)
        feature_flatten = self.fc1(feature_flatten)
        feature_flatten = F.relu(feature_flatten)
        feature_flatten = self.fc2(feature_flatten)
        feature_flatten = feature_flatten.reshape(feature_flatten.size(0), feature_flatten.size(1), 1, 1)
        feature_flatten = feature_flatten.expand(feature_flatten.size(0), feature_flatten.size(1), feature_map.size(2), feature_map.size(3))

        feature_map = torch.cat((feature_map, feature_flatten), dim=1)

        q_values = self.q_values(feature_map)
        deconv_q_values = F.interpolate(q_values, size=(obs.size(2), obs.size(3)), mode='bilinear', align_corners=False)

        if self.mask is not None:
            mask = self.mask.expand(deconv_q_values.size()).bool()
            mask = 1-mask
            deconv_q_values[mask] = -100
        return deconv_q_values

class FCNInHand(nn.Module):
    def __init__(self, n_input_channel=1, n_primitives=1, patch_shape=(1, 64, 64), n_in_hand_feature=10):
        super(FCNInHand, self).__init__()

        self.fcn_conv = nn.Sequential(OrderedDict([
            ('fcn_conv1', nn.Conv2d(n_input_channel, 32, kernel_size=7, stride=2)),
            ('fcn_bn1', nn.BatchNorm2d(32)),
            ('fcn_relu1', nn.ReLU(inplace=True)),
            ('fcn_conv2', nn.Conv2d(32, 64, kernel_size=5, stride=1)),
            ('fcn_bn2', nn.BatchNorm2d(64)),
            ('fcn_relu2', nn.ReLU(inplace=True)),
            ('fcn_conv3', nn.Conv2d(64, 128, kernel_size=3, stride=1)),
            ('fcn_bn3', nn.BatchNorm2d(128)),
            ('fcn_relu3', nn.ReLU(inplace=True)),
            ('fcn_conv4', nn.Conv2d(128, 64, kernel_size=1, stride=1, bias=False)),
            ('fcn_bn4', nn.BatchNorm2d(64)),
            ('fcn_relu4', nn.ReLU(inplace=True)),
        ]))

        self.q_values = nn.Conv2d(64+n_in_hand_feature, n_primitives, kernel_size=1, stride=1, bias=False)

        self.cnn_conv = nn.Sequential(OrderedDict([
            ('cnn_conv1', nn.Conv2d(patch_shape[0], 32, kernel_size=4, stride=2)),
            ('cnn_relu1', nn.ReLU(inplace=True)),
            ('cnn_conv2', nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            ('cnn_relu2', nn.ReLU(inplace=True)),
            ('cnn_conv3', nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            ('cnn_relu3', nn.ReLU(inplace=True)),
        ]))
        conv_out_size = self._getConvOut(patch_shape)
        self.fc1 = nn.Linear(conv_out_size, 512)
        self.fc2 = nn.Linear(512, n_in_hand_feature)

        for m in self.named_modules():
            if isinstance(m[1], nn.Conv2d):
                # nn.init.kaiming_normal_(m[1].weight.data)
                nn.init.xavier_normal_(m[1].weight.data)
            elif isinstance(m[1], nn.BatchNorm2d):
                m[1].weight.data.fill_(1)
                m[1].bias.data.zero_()

        self.mask = None

    def setCandidatePos(self, candidate_pos, input_size=(250, 250)):
        if candidate_pos is None:
            self.mask = None
        else:
            self.mask = torch.zeros(input_size)
            index = np.array(np.meshgrid(candidate_pos[0], candidate_pos[1])).T.reshape(-1, 2)
            index = torch.tensor(index).long()
            for idx in index:
                self.mask[idx[0], idx[1]] = 1

    def _getConvOut(self, shape):
        o = self.cnn_conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, obs, in_hand):
        feature_map = self.fcn_conv(obs)

        in_hand_out = self.cnn_conv(in_hand)
        in_hand_out = in_hand_out.view(in_hand.size()[0], -1)
        in_hand_out = self.fc1(in_hand_out)
        in_hand_out = F.relu(in_hand_out)
        in_hand_out = self.fc2(in_hand_out)
        in_hand_out = in_hand_out.reshape(in_hand_out.size(0), in_hand_out.size(1), 1, 1)
        in_hand_out = in_hand_out.expand(in_hand_out.size(0), in_hand_out.size(1), feature_map.size(2), feature_map.size(3))

        feature_map = torch.cat((feature_map, in_hand_out), dim=1)

        q_values = self.q_values(feature_map)
        deconv_q_values = F.interpolate(q_values, size=(obs.size(2), obs.size(3)), mode='bilinear', align_corners=False)

        if self.mask is not None:
            mask = self.mask.expand(deconv_q_values.size()).bool()
            mask = 1-mask
            deconv_q_values[mask] = -100
        return deconv_q_values

class FCNRotationOnly(nn.Module):
    def __init__(self, device, num_rotations=8, half_rotation=False):
        """
        fully convolutional net for rotation only agent
        Args:
            device: torch device
            num_rotations: number of rotations
            half_rotation: if True, max rotation is 180 degree. otherwise 360
        """
        super(FCNRotationOnly, self).__init__()

        self.num_rotations = num_rotations
        self.half_rotation = half_rotation

        self.device = device

        self.rotation_net = nn.Sequential(OrderedDict([
            ('rotation_conv1', nn.Conv2d(1, 32, kernel_size=7, stride=2)),
            ('rotation_bn1', nn.BatchNorm2d(32)),
            ('rotation_relu1', nn.ReLU(inplace=True)),
            ('rotation_conv2', nn.Conv2d(32, 64, kernel_size=7, stride=2)),
            ('rotation_bn2', nn.BatchNorm2d(64)),
            ('rotation_relu2', nn.ReLU(inplace=True)),
            ('rotation_conv3', nn.Conv2d(64, 128, kernel_size=3, stride=1)),
            ('rotation_bn3', nn.BatchNorm2d(128)),
            ('rotation_relu3', nn.ReLU(inplace=True)),
            ('rotation_conv4', nn.Conv2d(128, 64, kernel_size=1, stride=1, bias=False)),
            ('rotation_bn4', nn.BatchNorm2d(64)),
            ('rotation_relu4', nn.ReLU(inplace=True)),
            ('rotation_output', nn.Conv2d(64, 1, kernel_size=1, stride=1, bias=False)),
        ]))

        for m in self.named_modules():
            if isinstance(m[1], nn.Conv2d):
                nn.init.kaiming_normal_(m[1].weight.data)
            elif isinstance(m[1], nn.BatchNorm2d):
                m[1].weight.data.fill_(1)
                m[1].bias.data.zero_()

    def forwardRotationFCN(self, x, specific_rotation=None):
        """
        forward pass rotation fcn
        Args:
            x: input tensor with shape b x c x d x d
            specific_rotation: int. specify the rotation. if None, get outputs for all possible rotations

        Returns: list of output tensors for each rotation. Each tensor has shape b x 1 x d x d

        """
        outputs = []
        if specific_rotation is None:
            rotations = range(self.num_rotations)
        else:
            rotations = [specific_rotation]
        for rotate_idx in rotations:
            if self.half_rotation:
                rotate_theta = np.radians(rotate_idx * (180 / self.num_rotations))
            else:
                rotate_theta = np.radians(rotate_idx * (360 / self.num_rotations))

            # Compute sample grid for rotation BEFORE neural network
            affine_mat_before = np.asarray([[np.cos(-rotate_theta), np.sin(-rotate_theta), 0],
                                            [-np.sin(-rotate_theta), np.cos(-rotate_theta), 0]])
            affine_mat_before.shape = (2, 3, 1)
            affine_mat_before = torch.from_numpy(affine_mat_before).permute(2, 0, 1).float().to(self.device)
            flow_grid_before = F.affine_grid(affine_mat_before, x.size())

            # Rotate images clockwise
            rotate_depth = F.grid_sample(x, flow_grid_before, mode='nearest')

            # forward pass conv net
            conv_output = self.rotation_net(rotate_depth)

            # Compute sample grid for rotation AFTER neural network
            affine_mat_after = np.asarray([[np.cos(rotate_theta), np.sin(rotate_theta), 0],
                                           [-np.sin(rotate_theta), np.cos(rotate_theta), 0]])
            affine_mat_after.shape = (2, 3, 1)
            affine_mat_after = torch.from_numpy(affine_mat_after).permute(2, 0, 1).float().to(self.device)
            flow_grid_after = F.affine_grid(affine_mat_after, x.size())

            unrotate_output = F.grid_sample(conv_output, flow_grid_after, mode='nearest')

            deconv = F.interpolate(unrotate_output, size=(x.size(2), x.size(3)),
                                   mode='bilinear', align_corners=False)
            outputs.append(deconv)
        return outputs

    def forward(self, obs, specific_rotations=None):
        """
        forward pass whole model and get predictions
        Args:
            obs: observation tensor in shape b x d x d x c
            specific_rotations: list of ints. specify rotations for each observations

        Returns:  prediction tensor in shape b x num_rotations x d x d

        """
        obs = obs.cpu()
        predictions = []
        for i in range(obs.shape[0]):
            # Add padding
            depth_heightmap = np.squeeze(obs[i])
            depth_heightmap_2x = ndimage.zoom(depth_heightmap, zoom=2, order=0)
            diag_length = float(depth_heightmap_2x.shape[0]) * np.sqrt(2)
            diag_length = np.ceil(diag_length / 32) * 32
            padding_width = int((diag_length - depth_heightmap_2x.shape[0]) / 2)
            depth_heightmap_2x = np.pad(depth_heightmap_2x, padding_width, 'constant', constant_values=0)
            depth_heightmap_2x.shape = (1, 1, depth_heightmap_2x.shape[0], depth_heightmap_2x.shape[1])
            rotation_input_tensor = torch.tensor(depth_heightmap_2x).to(self.device)

            # forward pass rotation net
            if specific_rotations is None:
                rotation_output = self.forwardRotationFCN(rotation_input_tensor)
            else:
                rotation_output = self.forwardRotationFCN(rotation_input_tensor, specific_rotations[i])

            # remove padding
            rotation_prediction = rotation_output[0][:, 0,
                                  padding_width: -padding_width:2,
                                  padding_width: -padding_width:2]
            for rotate_idx in range(1, len(rotation_output)):
                rotation_prediction = torch.cat((rotation_prediction,
                                                 rotation_output[rotate_idx][:, 0,
                                                 padding_width: -padding_width:2,
                                                 padding_width: -padding_width:2]))

            predictions.append(rotation_prediction)
        return torch.stack(predictions)


class FCNAdding(nn.Module):
    def __init__(self, device, num_rotations=8, half_rotation=False, num_heights=10, height_range=(0, 0.1)):
        """
        model for agent that add results from rotation fcn and height fcn
        Args:
            device: torch device
            num_rotations: number of rotations
            half_rotation: if True, max rotation is 180 degree. otherwise 360
            num_heights: number of heights
            height_range: range of heights
        """
        super(FCNAdding, self).__init__()

        self.num_rotations = num_rotations
        self.half_rotation = half_rotation

        self.num_heights = num_heights
        self.height_range = height_range
        self.height_space = np.linspace(height_range[0], height_range[1], self.num_heights)

        self.device = device

        self.rotation_net = nn.Sequential(OrderedDict([
            ('rotation_conv1', nn.Conv2d(1, 32, kernel_size=7, stride=2)),
            ('rotation_bn1', nn.BatchNorm2d(32)),
            ('rotation_relu1', nn.ReLU(inplace=True)),
            ('rotation_conv2', nn.Conv2d(32, 64, kernel_size=7, stride=2)),
            ('rotation_bn2', nn.BatchNorm2d(64)),
            ('rotation_relu2', nn.ReLU(inplace=True)),
            ('rotation_conv3', nn.Conv2d(64, 128, kernel_size=3, stride=1)),
            ('rotation_bn3', nn.BatchNorm2d(128)),
            ('rotation_relu3', nn.ReLU(inplace=True)),
            ('rotation_conv4', nn.Conv2d(128, 64, kernel_size=1, stride=1, bias=False)),
            ('rotation_bn4', nn.BatchNorm2d(64)),
            ('rotation_relu4', nn.ReLU(inplace=True)),
            ('rotation_output', nn.Conv2d(64, 1, kernel_size=1, stride=1, bias=False)),
        ]))

        self.height_net = nn.Sequential(OrderedDict([
            ('height_conv1', nn.Conv2d(1, 32, kernel_size=7, stride=2)),
            ('height_bn1', nn.BatchNorm2d(32)),
            ('height_relu1', nn.ReLU(inplace=True)),
            ('height_conv2', nn.Conv2d(32, 64, kernel_size=7, stride=2)),
            ('height_bn2', nn.BatchNorm2d(64)),
            ('height_relu2', nn.ReLU(inplace=True)),
            ('height_conv3', nn.Conv2d(64, 128, kernel_size=3, stride=1)),
            ('height_bn3', nn.BatchNorm2d(128)),
            ('height_relu3', nn.ReLU(inplace=True)),
            ('height_conv4', nn.Conv2d(128, 64, kernel_size=1, stride=1, bias=False)),
            ('height_bn4', nn.BatchNorm2d(64)),
            ('height_relu4', nn.ReLU(inplace=True)),
            ('height_output', nn.Conv2d(64, 1, kernel_size=1, stride=1, bias=False)),
        ]))

        for m in self.named_modules():
            if isinstance(m[1], nn.Conv2d):
                nn.init.kaiming_normal_(m[1].weight.data)
            elif isinstance(m[1], nn.BatchNorm2d):
                m[1].weight.data.fill_(1)
                m[1].bias.data.zero_()

    def forwardRotationFCN(self, x, specific_rotation=-1):
        """
        forward pass rotation fcn
        Args:
            x: input tensor with shape b x c x d x d
            specific_rotation: int. specify the rotation. if None, get outputs for all possible rotations

        Returns: list of output tensors for each rotation. Each tensor has shape b x 1 x d x d

        """
        outputs = []
        if specific_rotation == -1:
            rotations = range(self.num_rotations)
        else:
            rotations = [specific_rotation]
        for rotate_idx in rotations:
            if self.half_rotation:
                rotate_theta = np.radians(rotate_idx * (180 / self.num_rotations))
            else:
                rotate_theta = np.radians(rotate_idx * (360 / self.num_rotations))

            # Compute sample grid for rotation BEFORE neural network
            affine_mat_before = np.asarray([[np.cos(-rotate_theta), np.sin(-rotate_theta), 0],
                                            [-np.sin(-rotate_theta), np.cos(-rotate_theta), 0]])
            affine_mat_before.shape = (2, 3, 1)
            affine_mat_before = torch.from_numpy(affine_mat_before).permute(2, 0, 1).float().to(self.device)
            flow_grid_before = F.affine_grid(affine_mat_before, x.size())

            # Rotate images clockwise
            rotate_depth = F.grid_sample(x, flow_grid_before, mode='nearest')

            # forward pass conv net
            conv_output = self.rotation_net(rotate_depth)

            # Compute sample grid for rotation AFTER neural network
            affine_mat_after = np.asarray([[np.cos(rotate_theta), np.sin(rotate_theta), 0],
                                           [-np.sin(rotate_theta), np.cos(rotate_theta), 0]])
            affine_mat_after.shape = (2, 3, 1)
            affine_mat_after = torch.from_numpy(affine_mat_after).permute(2, 0, 1).float().to(self.device)
            flow_grid_after = F.affine_grid(affine_mat_after, x.size())

            unrotate_output = F.grid_sample(conv_output, flow_grid_after, mode='nearest')

            deconv = F.interpolate(unrotate_output, size=(x.size(2), x.size(3)),
                                   mode='bilinear', align_corners=False)
            outputs.append(deconv)
        return outputs

    def forwardHeightFCN(self, x, specific_height=None):
        """
        forward pass height fcn
        Args:
            x: input tensor in shape b x c x d x d
            specific_height: int. specify the height. if None, get outputs for all possible heights

        Returns: list of output tensors for each heights. Each tensor has shape b x 1 x d x d

        """
        outputs = []
        if specific_height is None:
            heights = range(self.num_heights)
        else:
            heights = [specific_height]
        for height_idx in heights:
            height_offset = self.height_space[height_idx]

            # compute heightmap with offset
            offset_heightmap = x - height_offset

            # forward pass conv net
            conv_output = self.height_net(offset_heightmap)

            deconv = F.interpolate(conv_output, size=(x.size(2), x.size(3)),
                                   mode='bilinear', align_corners=False)
            outputs.append(deconv)
        return outputs

    def forward(self, obs, specific_rotations=None, specific_heights=None):
        """
        forward pass whole model and get predictions
        Args:
            obs: observation tensor in shape b x d x d x c
            specific_rotations: list of ints. specify rotations for each observations
            specific_heights: list of ints. specify heights for each observations

        Returns: prediction tensor in shape b x num_rotations x num_heights x d x d

        """
        obs = obs.cpu()
        predictions = []
        for i in range(obs.shape[0]):
            # Add padding
            depth_heightmap = np.squeeze(obs[i])
            depth_heightmap_2x = ndimage.zoom(depth_heightmap, zoom=2, order=0)
            diag_length = float(depth_heightmap_2x.shape[0]) * np.sqrt(2)
            diag_length = np.ceil(diag_length / 32) * 32
            padding_width = int((diag_length - depth_heightmap_2x.shape[0]) / 2)
            depth_heightmap_2x = np.pad(depth_heightmap_2x, padding_width, 'constant', constant_values=0)
            depth_heightmap_2x.shape = (1, 1, depth_heightmap_2x.shape[0], depth_heightmap_2x.shape[1])
            rotation_input_tensor = torch.tensor(depth_heightmap_2x).to(self.device)

            # forward pass rotation net
            if specific_rotations is None:
                rotation_output = self.forwardRotationFCN(rotation_input_tensor)
            else:
                rotation_output = self.forwardRotationFCN(rotation_input_tensor, specific_rotations[i])

            # remove padding
            rotation_prediction = rotation_output[0][:, 0,
                                  padding_width: -padding_width:2,
                                  padding_width: -padding_width:2]
            for rotate_idx in range(1, len(rotation_output)):
                rotation_prediction = torch.cat((rotation_prediction,
                                                 rotation_output[rotate_idx][:, 0,
                                                 padding_width: -padding_width:2,
                                                 padding_width: -padding_width:2]))

            rotation_prediction = rotation_prediction.expand((self.num_heights, -1, -1, -1))
            rotation_prediction = rotation_prediction.permute((1, 0, 2, 3))

            height_input_tensor = obs[i].unsqueeze(0).permute(0, 3, 1, 2).to(self.device)
            if specific_heights is None:
                height_output = self.forwardHeightFCN(height_input_tensor)
            else:
                height_output = self.forwardHeightFCN(height_input_tensor, specific_heights[i])

            height_prediction = torch.cat(height_output).squeeze(1)
            height_prediction = height_prediction.expand((self.num_rotations, -1, -1, -1))

            prediction = 0.5 * rotation_prediction + 0.5 * height_prediction

            predictions.append(prediction)
        return torch.stack(predictions)
