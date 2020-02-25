import numpy as np
from scipy import ndimage
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

class FCNInHand(nn.Module):
    def __init__(self, n_input_channel=1, n_primitives=1, patch_shape=(1, 24, 24), domain_shape=(1, 100, 100)):
        super(FCNInHand, self).__init__()
        self.pick_domain_conv = nn.Sequential(
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

        self.pick_q_values = nn.Conv2d(16, 1, kernel_size=1, stride=1, bias=False)

        self.place_domain_conv = nn.Sequential(
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

        self.in_hand_conv = nn.Sequential(
            (nn.Conv2d(patch_shape[0], 32, kernel_size=4, stride=2)),
            (nn.ReLU(inplace=True)),
            (nn.Conv2d(32, 64, kernel_size=4, stride=1)),
            (nn.ReLU(inplace=True)),
            (nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            (nn.ReLU(inplace=True)),
        )
        in_hand_out_size = self._getInHandConvOut(patch_shape)
        self.in_hand_fc = nn.Sequential(
            nn.Linear(in_hand_out_size, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 16)
        )
        self.place_q_values = nn.Conv2d(32, 1, kernel_size=1, stride=1, bias=False)


        for m in self.named_modules():
            if isinstance(m[1], nn.Conv2d):
                # nn.init.kaiming_normal_(m[1].weight.data)
                nn.init.xavier_normal_(m[1].weight.data)
            elif isinstance(m[1], nn.BatchNorm2d):
                m[1].weight.data.fill_(1)
                m[1].bias.data.zero_()

    def _getDomainConvOut(self, shape):
        o = self.pick_domain_down(self.pick_domain_conv(torch.zeros(1, *shape)))
        return int(np.prod(o.size()))

    def _getInHandConvOut(self, shape):
        o = self.in_hand_conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, obs, in_hand):
        pick_feature_map = self.pick_domain_conv(obs)

        pick_q_values = self.pick_q_values(pick_feature_map)

        place_feature_map = self.place_domain_conv(obs)

        in_hand_out = self.in_hand_conv(in_hand)
        in_hand_out = in_hand_out.view(in_hand.size()[0], -1)
        in_hand_out = self.in_hand_fc(in_hand_out)
        in_hand_out = in_hand_out.reshape(in_hand_out.size(0), in_hand_out.size(1), 1, 1)
        in_hand_out = in_hand_out.expand(in_hand_out.size(0), in_hand_out.size(1), place_feature_map.size(2), place_feature_map.size(3))

        feature_map = torch.cat((place_feature_map, in_hand_out), dim=1)
        place_q_values = self.place_q_values(feature_map)

        q_values = torch.cat((pick_q_values, place_q_values), dim=1)

        return q_values


class FCNInHandDomainFC(nn.Module):
    def __init__(self, n_input_channel=1, n_primitives=1, patch_shape=(1, 24, 24), domain_shape=(1, 100, 100)):
        super(FCNInHandDomainFC, self).__init__()
        self.pick_domain_conv = nn.Sequential(
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
            (nn.Conv2d(16, 8, kernel_size=1, stride=1, bias=False)),
            (nn.MaxPool2d(2, 2)),
            (nn.ReLU(inplace=True))
        )
        conv_out_size = self._getDomainConvOut(domain_shape)
        self.pick_fc = nn.Sequential(
            nn.Linear(conv_out_size, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 16)
        )
        self.pick_q_values = nn.Conv2d(32, 1, kernel_size=1, stride=1, bias=False)

        self.place_domain_conv = nn.Sequential(
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
            (nn.Conv2d(16, 8, kernel_size=1, stride=1, bias=False)),
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

        self.place_fc = nn.Sequential(
            nn.Linear(conv_out_size, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 16)
        )

        in_hand_out_size = self._getInHandConvOut(patch_shape)
        self.in_hand_fc = nn.Sequential(
            nn.Linear(in_hand_out_size, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 16)
        )
        self.place_q_values = nn.Conv2d(48, 1, kernel_size=1, stride=1, bias=False)

        for m in self.named_modules():
            if isinstance(m[1], nn.Conv2d):
                # nn.init.kaiming_normal_(m[1].weight.data)
                nn.init.xavier_normal_(m[1].weight.data)
            elif isinstance(m[1], nn.BatchNorm2d):
                m[1].weight.data.fill_(1)
                m[1].bias.data.zero_()

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
        pick_feature_flatten = self.pick_fc(pick_feature_flatten)
        pick_feature_flatten = pick_feature_flatten.reshape(pick_feature_flatten.size(0), pick_feature_flatten.size(1), 1, 1)
        pick_feature_flatten = pick_feature_flatten.expand(pick_feature_flatten.size(0), pick_feature_flatten.size(1), pick_feature_map.size(2),
                                                      pick_feature_map.size(3))

        pick_q_values = self.pick_q_values(torch.cat((pick_feature_map, pick_feature_flatten), dim=1))

        place_feature_map = self.place_domain_conv(obs)
        place_domain_feature_map = self.place_domain_down(place_feature_map)
        place_feature_flatten = place_domain_feature_map.view(obs.size()[0], -1)
        place_feature_flatten = self.place_fc(place_feature_flatten)
        place_feature_flatten = place_feature_flatten.reshape(place_feature_flatten.size(0), place_feature_flatten.size(1), 1, 1)
        place_feature_flatten = place_feature_flatten.expand(place_feature_flatten.size(0), place_feature_flatten.size(1), place_feature_map.size(2),
                                                      place_feature_map.size(3))
        in_hand_out = self.in_hand_conv(in_hand)
        in_hand_out = in_hand_out.view(in_hand.size()[0], -1)
        in_hand_out = self.in_hand_fc(in_hand_out)
        in_hand_out = in_hand_out.reshape(in_hand_out.size(0), in_hand_out.size(1), 1, 1)
        in_hand_out = in_hand_out.expand(in_hand_out.size(0), in_hand_out.size(1), place_feature_map.size(2),
                                         place_feature_map.size(3))

        place_q_values = self.place_q_values(torch.cat((place_feature_map, place_feature_flatten, in_hand_out), dim=1))

        q_values = torch.cat((pick_q_values, place_q_values), dim=1)
        # deconv_q_values = F.interpolate(q_values, size=(obs.size(2), obs.size(3)), mode='bilinear', align_corners=False)

        return q_values

class FCNInHandDynamicFilter(nn.Module):
    def __init__(self, n_input_channel=1, n_primitives=1, patch_shape=(1, 24, 24), domain_shape=(1, 100, 100)):
        super(FCNInHandDynamicFilter, self).__init__()
        self.pick_domain_conv = nn.Sequential(
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
            (nn.Conv2d(16, 8, kernel_size=1, stride=1, bias=False)),
            (nn.MaxPool2d(2, 2)),
            (nn.ReLU(inplace=True))
        )
        conv_out_size = self._getDomainConvOut(domain_shape)
        self.pick_filter_fc = nn.Sequential(
            nn.Linear(conv_out_size, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 16*16+16)
        )
        self.pick_q_values = nn.Conv2d(16, 1, kernel_size=1, stride=1, bias=False)

        self.place_domain_conv = nn.Sequential(
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
            (nn.Conv2d(16, 8, kernel_size=1, stride=1, bias=False)),
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
            nn.Linear(place_conv_out_size, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 16*16+16)
        )
        self.place_q_values = nn.Conv2d(16, 1, kernel_size=1, stride=1, bias=False)


        for m in self.named_modules():
            if isinstance(m[1], nn.Conv2d):
                # nn.init.kaiming_normal_(m[1].weight.data)
                nn.init.xavier_normal_(m[1].weight.data)
            elif isinstance(m[1], nn.BatchNorm2d):
                m[1].weight.data.fill_(1)
                m[1].bias.data.zero_()

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

        self.pick_conv5 = nn.Sequential(
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

        self.pick_q_values = nn.Conv2d(80, 1, kernel_size=1, stride=1, bias=False)

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

        self.place_conv5 = nn.Sequential(
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
            ('place_fc2', nn.Linear(1024, 80 * 80 + 80))
        ]))
        self.place_q_values = nn.Conv2d(80, 1, kernel_size=1, stride=1, bias=False)


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
        df_channel = 80
        obs_2 = F.interpolate(obs, scale_factor=0.5, mode='bilinear', align_corners=False)
        obs_4 = F.interpolate(obs_2, scale_factor=0.5, mode='bilinear', align_corners=False)
        obs_8 = F.interpolate(obs_4, scale_factor=0.5, mode='bilinear', align_corners=False)
        obs_16 = F.interpolate(obs_8, scale_factor=0.5, mode='bilinear', align_corners=False)

        pick_feature_map_1 = self.pick_conv1(obs)
        pick_feature_map_2 = F.interpolate(self.pick_conv2(obs_2), size=(obs.shape[2], obs.shape[3]), mode='bilinear', align_corners=False)
        pick_feature_map_4 = F.interpolate(self.pick_conv3(obs_4), size=(obs.shape[2], obs.shape[3]), mode='bilinear', align_corners=False)
        pick_feature_map_8 = F.interpolate(self.pick_conv4(obs_8), size=(obs.shape[2], obs.shape[3]), mode='bilinear', align_corners=False)
        pick_feature_map_16 = F.interpolate(self.pick_conv5(obs_16), size=(obs.shape[2], obs.shape[3]), mode='bilinear', align_corners=False)

        pick_feature_map = torch.cat((pick_feature_map_1, pick_feature_map_2, pick_feature_map_4, pick_feature_map_8, pick_feature_map_16), dim=1)
        pick_q_values = self.pick_q_values(pick_feature_map)

        place_feature_map_1 = self.place_conv1(obs)
        place_feature_map_2 = F.interpolate(self.place_conv2(obs_2), size=(obs.shape[2], obs.shape[3]), mode='bilinear', align_corners=False)
        place_feature_map_4 = F.interpolate(self.place_conv3(obs_4), size=(obs.shape[2], obs.shape[3]), mode='bilinear', align_corners=False)
        place_feature_map_8 = F.interpolate(self.place_conv4(obs_8), size=(obs.shape[2], obs.shape[3]), mode='bilinear', align_corners=False)
        place_feature_map_16 = F.interpolate(self.place_conv5(obs_16), size=(obs.shape[2], obs.shape[3]), mode='bilinear', align_corners=False)

        place_feature_map = torch.cat((place_feature_map_1, place_feature_map_2, place_feature_map_4, place_feature_map_8, place_feature_map_16), dim=1)

        in_hand_out = self.in_hand_conv(in_hand)
        in_hand_flatten = in_hand_out.view(in_hand.size()[0], -1)
        place_filter_weights = self.place_filter_fc(in_hand_flatten)

        place_filter_weight = place_filter_weights[:, :df_channel * df_channel].reshape(obs.size()[0], df_channel,
                                                                                        df_channel, 1, 1)
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

class FCNInHandDynamicFilterPyramid(nn.Module):
    def __init__(self, n_input_channel=1, n_primitives=1, patch_shape=(1, 24, 24), domain_shape=(1, 100, 100)):
        super(FCNInHandDynamicFilterPyramid, self).__init__()
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

class FCNInHandDynamicFilterU(nn.Module):
    def __init__(self, n_input_channel=1, n_primitives=1, patch_shape=(1, 24, 24), domain_shape=(1, 100, 100)):
        super(FCNInHandDynamicFilterU, self).__init__()
        self.conv_down_1 = nn.Sequential(
            (nn.Conv2d(n_input_channel, 32, kernel_size=3, stride=1, padding=1)),
            (nn.ReLU(inplace=True)),
            (nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)),
            (nn.ReLU(inplace=True))
        )
        self.conv_down_2 = nn.Sequential(
            (nn.MaxPool2d(2)),
            (nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)),
            (nn.ReLU(inplace=True)),
            (nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)),
            (nn.ReLU(inplace=True))
        )
        self.conv_down_4 = nn.Sequential(
            (nn.MaxPool2d(2)),
            (nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)),
            (nn.ReLU(inplace=True)),
            (nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)),
            (nn.ReLU(inplace=True))
        )
        self.conv_down_8 = nn.Sequential(
            (nn.MaxPool2d(2)),
            (nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)),
            (nn.ReLU(inplace=True)),
            (nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)),
            (nn.ReLU(inplace=True))
        )
        self.conv_down_16 = nn.Sequential(
            (nn.MaxPool2d(2)),
            (nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)),
            (nn.ReLU(inplace=True)),
            (nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)),
            (nn.ReLU(inplace=True))
        )

        self.conv_up_8 = nn.Sequential(
            (nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)),
            (nn.ReLU(inplace=True)),
            (nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)),
            (nn.ReLU(inplace=True))
        )
        self.conv_up_4 = nn.Sequential(
            (nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)),
            (nn.ReLU(inplace=True)),
            (nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)),
            (nn.ReLU(inplace=True))
        )
        self.conv_up_2 = nn.Sequential(
            (nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)),
            (nn.ReLU(inplace=True)),
            (nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)),
            (nn.ReLU(inplace=True))
        )
        self.conv_up_1 = nn.Sequential(
            (nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)),
            (nn.ReLU(inplace=True)),
            (nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)),
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
        in_hand_conv_out_size = self._getInHandConvOut(patch_shape)
        self.in_hand_filter_fc = nn.Sequential(OrderedDict([
            ('place_fc1', nn.Linear(in_hand_conv_out_size, 1024)),
            ('place_fc_relu1', nn.ReLU(inplace=True)),
            ('place_fc2', nn.Linear(1024, 32 * 32 + 32))
        ]))

        self.pick_q_values = nn.Conv2d(32, 1, kernel_size=1, stride=1)
        self.place_q_values = nn.Conv2d(32, 1, kernel_size=1, stride=1)

        for m in self.named_modules():
            if isinstance(m[1], nn.Conv2d):
                # nn.init.kaiming_normal_(m[1].weight.data)
                nn.init.xavier_normal_(m[1].weight.data)
            elif isinstance(m[1], nn.BatchNorm2d):
                m[1].weight.data.fill_(1)
                m[1].bias.data.zero_()

    def _getInHandConvOut(self, shape):
        o = self.in_hand_conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, obs, in_hand):
        df_channel = 32
        feature_map_1 = self.conv_down_1(obs)
        feature_map_2 = self.conv_down_2(feature_map_1)
        feature_map_4 = self.conv_down_4(feature_map_2)
        feature_map_8 = self.conv_down_8(feature_map_4)
        feature_map_16 = self.conv_down_16(feature_map_8)
        feature_map_up_8 = self.conv_up_8(torch.cat((feature_map_8, F.interpolate(feature_map_16, size=16, mode='bilinear', align_corners=False)), dim=1))
        feature_map_up_4 = self.conv_up_4(torch.cat((feature_map_4, F.interpolate(feature_map_up_8, size=32, mode='bilinear', align_corners=False)), dim=1))
        feature_map_up_2 = self.conv_up_2(torch.cat((feature_map_2, F.interpolate(feature_map_up_4, size=64, mode='bilinear', align_corners=False)), dim=1))
        feature_map_up_1 = self.conv_up_1(torch.cat((feature_map_1, F.interpolate(feature_map_up_2, size=128, mode='bilinear', align_corners=False)), dim=1))

        in_hand_out = self.in_hand_conv(in_hand)
        in_hand_flatten = in_hand_out.view(in_hand.size()[0], -1)
        in_hand_filter_weights = self.in_hand_filter_fc(in_hand_flatten)

        in_hand_filter_weight = in_hand_filter_weights[:, :df_channel * df_channel].reshape(obs.size()[0], df_channel,
                                                                                        df_channel, 1, 1)
        in_hand_filter_bias = in_hand_filter_weights[:, df_channel * df_channel:].reshape(obs.size()[0], df_channel)
        in_hand_filter_weight = in_hand_filter_weight.reshape(obs.size()[0] * df_channel, df_channel, 1, 1)
        in_hand_filter_bias = in_hand_filter_bias.reshape(obs.size()[0] * df_channel)
        place_feature_map = feature_map_up_1.reshape(1, feature_map_up_1.size(0) * feature_map_up_1.size(1),
                                                      feature_map_up_1.size(2), feature_map_up_1.size(3))
        place_feature_map = F.conv2d(place_feature_map, weight=in_hand_filter_weight, bias=in_hand_filter_bias,
                                     groups=obs.size(0))
        place_feature_map = place_feature_map.reshape(obs.size(0), -1, place_feature_map.size(2),
                                                      place_feature_map.size(3))
        place_feature_map = F.relu(place_feature_map, inplace=True)

        place_q_values = self.place_q_values(place_feature_map)
        pick_q_values = self.pick_q_values(feature_map_up_1)
        q_values = torch.cat((pick_q_values, place_q_values), dim=1)

        return q_values

class UCat(nn.Module):
    def __init__(self, n_input_channel=1, n_primitives=1, patch_shape=(1, 24, 24), domain_shape=(1, 100, 100)):
        super().__init__()
        self.conv_down_1 = nn.Sequential(
            (nn.Conv2d(n_input_channel, 32, kernel_size=3, stride=1, padding=1)),
            (nn.ReLU(inplace=True)),
            (nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)),
            (nn.ReLU(inplace=True))
        )
        self.conv_down_2 = nn.Sequential(
            (nn.MaxPool2d(2)),
            (nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)),
            (nn.ReLU(inplace=True)),
            (nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)),
            (nn.ReLU(inplace=True))
        )
        self.conv_down_4 = nn.Sequential(
            (nn.MaxPool2d(2)),
            (nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)),
            (nn.ReLU(inplace=True)),
            (nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)),
            (nn.ReLU(inplace=True))
        )
        self.conv_down_8 = nn.Sequential(
            (nn.MaxPool2d(2)),
            (nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)),
            (nn.ReLU(inplace=True)),
            (nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)),
            (nn.ReLU(inplace=True))
        )
        self.conv_down_16 = nn.Sequential(
            (nn.MaxPool2d(2)),
            (nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)),
            (nn.ReLU(inplace=True)),
            (nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)),
            (nn.ReLU(inplace=True))
        )

        self.conv_cat_in_hand = nn.Sequential(
            (nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)),
            (nn.ReLU(inplace=True))
        )

        self.conv_up_8 = nn.Sequential(
            (nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)),
            (nn.ReLU(inplace=True)),
            (nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)),
            (nn.ReLU(inplace=True))
        )
        self.conv_up_4 = nn.Sequential(
            (nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)),
            (nn.ReLU(inplace=True)),
            (nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)),
            (nn.ReLU(inplace=True))
        )
        self.conv_up_2 = nn.Sequential(
            (nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)),
            (nn.ReLU(inplace=True)),
            (nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)),
            (nn.ReLU(inplace=True))
        )
        self.conv_up_1 = nn.Sequential(
            (nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)),
            (nn.ReLU(inplace=True)),
            (nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)),
            (nn.ReLU(inplace=True))
        )

        self.in_hand_conv = nn.Sequential(OrderedDict([
            ('cnn_conv1', nn.Conv2d(patch_shape[0], 64, kernel_size=3)),
            ('cnn_relu1', nn.ReLU(inplace=True)),
            ('cnn_conv2', nn.Conv2d(64, 128, kernel_size=3)),
            ('cnn_relu2', nn.ReLU(inplace=True)),
            ('cnn_pool2', nn.MaxPool2d(2)),
            ('cnn_conv3', nn.Conv2d(128, 256, kernel_size=3)),
            ('cnn_relu3', nn.ReLU(inplace=True)),
        ]))

        # self.in_hand_conv = nn.Sequential(OrderedDict([
        #     ('cnn_conv1', nn.Conv2d(patch_shape[0], 32, kernel_size=3)),
        #     ('cnn_relu1', nn.ReLU(inplace=True)),
        #     ('cnn_conv2', nn.Conv2d(32, 64, kernel_size=3)),
        #     ('cnn_relu2', nn.ReLU(inplace=True)),
        #     ('cnn_pool2', nn.MaxPool2d(2)),
        #     ('cnn_conv3', nn.Conv2d(64, 128, kernel_size=3)),
        #     ('cnn_relu3', nn.ReLU(inplace=True)),
        #     ('cnn_pool4', nn.MaxPool2d(2)),
        #     ('cnn_conv5', nn.Conv2d(128, 256, kernel_size=3, padding=1)),
        #
        # ]))

        self.pick_q_values = nn.Conv2d(32, 1, kernel_size=1, stride=1)
        self.place_q_values = nn.Conv2d(32, 1, kernel_size=1, stride=1)

        for m in self.named_modules():
            if isinstance(m[1], nn.Conv2d):
                # nn.init.kaiming_normal_(m[1].weight.data)
                nn.init.xavier_normal_(m[1].weight.data)
            elif isinstance(m[1], nn.BatchNorm2d):
                m[1].weight.data.fill_(1)
                m[1].bias.data.zero_()

    def forward(self, obs, in_hand):
        feature_map_1 = self.conv_down_1(obs)
        feature_map_2 = self.conv_down_2(feature_map_1)
        feature_map_4 = self.conv_down_4(feature_map_2)
        feature_map_8 = self.conv_down_8(feature_map_4)
        feature_map_16 = self.conv_down_16(feature_map_8)

        in_hand_out = self.in_hand_conv(in_hand)
        feature_map_16 = self.conv_cat_in_hand(torch.cat((feature_map_16, in_hand_out), dim=1))

        feature_map_up_8 = self.conv_up_8(
            torch.cat((feature_map_8, F.interpolate(feature_map_16, size=16, mode='bilinear', align_corners=False)),
                      dim=1))
        feature_map_up_4 = self.conv_up_4(
            torch.cat((feature_map_4, F.interpolate(feature_map_up_8, size=32, mode='bilinear', align_corners=False)),
                      dim=1))
        feature_map_up_2 = self.conv_up_2(
            torch.cat((feature_map_2, F.interpolate(feature_map_up_4, size=64, mode='bilinear', align_corners=False)),
                      dim=1))
        feature_map_up_1 = self.conv_up_1(
            torch.cat((feature_map_1, F.interpolate(feature_map_up_2, size=128, mode='bilinear', align_corners=False)),
                      dim=1))

        place_q_values = self.place_q_values(feature_map_up_1)
        pick_q_values = self.pick_q_values(feature_map_up_1)
        q_values = torch.cat((pick_q_values, place_q_values), dim=1)

        return q_values

def conv3x3(in_planes, out_planes, stride=1, dilation=1, bias=False):
    "3x3 convolution with padding"

    kernel_size = np.asarray((3, 3))

    # Compute the size of the upsampled filter with
    # a specified dilation rate.
    upsampled_kernel_size = (kernel_size - 1) * (dilation - 1) + kernel_size

    # Determine the padding that is necessary for full padding,
    # meaning the output spatial size is equal to input spatial size
    full_padding = (upsampled_kernel_size - 1) // 2

    # Conv2d doesn't accept numpy arrays as arguments
    full_padding, kernel_size = tuple(full_padding), tuple(kernel_size)

    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=kernel_size,
        stride=stride,
        padding=full_padding,
        dilation=dilation,
        bias=bias,
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(BasicBlock, self).__init__()

        self.stride = stride
        self.dilation = dilation
        self.downsample = downsample

        self.conv1 = conv3x3(inplanes, planes, stride, dilation=dilation)
        # self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, dilation=dilation)
        # self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        # out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        # out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual

        out = self.relu(out)
        return out

class Interpolate(nn.Module):
    def __init__(
            self,
            size=None,
            scale_factor=None,
            mode="nearest",
            align_corners=None,
    ):
        super().__init__()

        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, input):
        return F.interpolate(
            input, self.size, self.scale_factor, self.mode, self.align_corners
        )

class BaseModel(torch.nn.Module):
    """An abstract base class for a deep neural network.
    """

    def __init__(self):
        super().__init__()

    def forward(self, *args):
        raise NotImplementedError

    def load_weights(self, filename, device):
        """Loads the weights of a model from a given directory.
        """
        state_dict = torch.load(filename, map_location=device)
        try:
            self.load_state_dict(state_dict)
            print("Successfully loaded model weights from {}.".format(filename))
        except:
            print("[!] Could not load model weights. Training from scratch instead.")

    def save_weights(self, filename):
        """Saves the weights of a model to a given directory.
        """
        torch.save(self.state_dict(), filename)

    @property
    def num_params(self):
        return sum(param.numel() for param in self.parameters())

class From2Fit(BaseModel):
    def __init__(self, n_input_channel=1, n_primitives=1, patch_shape=(1, 24, 24), domain_shape=(1, 100, 100)):
        super().__init__()
        self._in_channels = n_input_channel
        self._out_channels = 1

        self._encoder = nn.Sequential(
            OrderedDict(
                [
                    (
                        "enc-conv0",
                        nn.Conv2d(
                            self._in_channels,
                            64,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    ("enc-relu0", nn.ReLU(inplace=True)),
                    ("enc-pool1", nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
                    (
                        "enc-resn2",
                        BasicBlock(
                            64,
                            128,
                            downsample=nn.Sequential(
                                nn.Conv2d(64, 128, kernel_size=1, bias=False),
                            ),
                            dilation=1,
                        ),
                    ),
                    ("enc-pool3", nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
                    (
                        "enc-resn4",
                        BasicBlock(
                            128,
                            256,
                            downsample=nn.Sequential(
                                nn.Conv2d(128, 256, kernel_size=1, bias=False),
                            ),
                            dilation=1,
                        ),
                    ),
                    (
                        "enc-resn5",
                        BasicBlock(
                            256,
                            512,
                            downsample=nn.Sequential(
                                nn.Conv2d(256, 512, kernel_size=1, bias=False),
                            ),
                            dilation=1,
                        ),
                    ),
                ]
            )
        )

        self._decoder = nn.Sequential(
            OrderedDict(
                [
                    (
                        "dec-resn0",
                        BasicBlock(
                            512,
                            256,
                            downsample=nn.Sequential(
                                nn.Conv2d(512, 256, kernel_size=1, bias=False),
                            ),
                            dilation=1,
                        ),
                    ),
                    (
                        "dec-resn1",
                        BasicBlock(
                            256,
                            128,
                            downsample=nn.Sequential(
                                nn.Conv2d(256, 128, kernel_size=1, bias=False),
                            ),
                            dilation=1,
                        ),
                    ),
                    (
                        "dec-upsm2",
                        Interpolate(
                            scale_factor=2, mode="bilinear", align_corners=True
                        ),
                    ),
                    (
                        "dec-resn3",
                        BasicBlock(
                            128,
                            64,
                            downsample=nn.Sequential(
                                nn.Conv2d(128, 64, kernel_size=1, bias=False),
                            ),
                            dilation=1,
                        ),
                    ),
                    (
                        "dec-upsm4",
                        Interpolate(
                            scale_factor=2, mode="bilinear", align_corners=True
                        ),
                    ),
                    (
                        "dec-conv5",
                        nn.Conv2d(64, 32, kernel_size=1, stride=1, bias=True),
                    ),
                ]
            )
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
            ('place_fc2', nn.Linear(1024, 32 * 32 + 32))
        ]))

        self.pick_q_values = nn.Conv2d(32, 1, kernel_size=1, stride=1, bias=False)
        self.place_q_values = nn.Conv2d(32, 1, kernel_size=1, stride=1, bias=False)

        self._init_weights()

    def _getInHandConvOut(self, shape):
        o = self.in_hand_conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, obs, in_hand):
        df_channel = 32
        out_enc = self._encoder(obs)
        out_dec = self._decoder(out_enc)
        pick_q_values = self.pick_q_values(out_dec)

        in_hand_out = self.in_hand_conv(in_hand)
        in_hand_flatten = in_hand_out.view(in_hand.size()[0], -1)
        place_filter_weights = self.place_filter_fc(in_hand_flatten)

        place_filter_weight = place_filter_weights[:, :df_channel * df_channel].reshape(obs.size()[0], df_channel,
                                                                                        df_channel, 1, 1)
        place_filter_bias = place_filter_weights[:, df_channel * df_channel:].reshape(obs.size()[0], df_channel)
        place_filter_weight = place_filter_weight.reshape(obs.size()[0] * df_channel, df_channel, 1, 1)
        place_filter_bias = place_filter_bias.reshape(obs.size()[0] * df_channel)
        place_feature_map = out_dec.reshape(1, out_dec.size(0) * out_dec.size(1),
                                                      out_dec.size(2), out_dec.size(3))
        place_feature_map = F.conv2d(place_feature_map, weight=place_filter_weight, bias=place_filter_bias,
                                     groups=obs.size(0))
        place_feature_map = place_feature_map.reshape(obs.size(0), -1, place_feature_map.size(2),
                                                      place_feature_map.size(3))

        place_q_values = self.place_q_values(place_feature_map)

        q_values = torch.cat((pick_q_values, place_q_values), dim=1)
        return q_values


class ResNet(BaseModel):
    def __init__(self, n_input_channel=1, n_primitives=1, patch_shape=(1, 24, 24), domain_shape=(1, 100, 100)):
        super().__init__()
        self._in_channels = n_input_channel
        self._out_channels = 1

        self._encoder = nn.Sequential(
            OrderedDict(
                [
                    (
                        "enc-conv1",
                        nn.Conv2d(
                            self._in_channels,
                            64,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    ("enc-relu1", nn.ReLU(inplace=True)),
                    ("enc-pool2", nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
                    (
                        "enc-resn2-1",
                        BasicBlock(
                            64,
                            64,
                            dilation=1,
                        ),
                    ),
                    (
                        "enc-resn2-2",
                        BasicBlock(
                            64,
                            128,
                            dilation=1,
                            downsample=nn.Sequential(
                                nn.Conv2d(64, 128, kernel_size=1, bias=False),
                            ),
                        ),
                    ),
                    ("enc-pool3", nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
                    (
                        "enc-resn3-1",
                        BasicBlock(
                            128,
                            128,
                            dilation=1,
                        ),
                    ),
                    (
                        "enc-resn3-2",
                        BasicBlock(
                            128,
                            256,
                            downsample=nn.Sequential(
                                nn.Conv2d(128, 256, kernel_size=1, bias=False),
                            ),
                            dilation=1,
                        ),
                    ),
                    ("enc-pool4", nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
                    (
                        "enc-resn4-1",
                        BasicBlock(
                            256,
                            256,
                            dilation=1,
                        ),
                    ),
                    (
                        "enc-resn4-2",
                        BasicBlock(
                            256,
                            512,
                            downsample=nn.Sequential(
                                nn.Conv2d(256, 512, kernel_size=1, bias=False),
                            ),
                            dilation=1,
                        ),
                    ),
                    ("enc-pool5", nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
                    (
                        "enc-resn5-1",
                        BasicBlock(
                            512,
                            512,
                            dilation=1,
                        ),
                    ),
                    (
                        "enc-resn5-2",
                        BasicBlock(
                            512,
                            512,
                            dilation=1,
                        ),
                    ),
                ]
            )
        )

        self._decoder = nn.Sequential(
            OrderedDict(
                [
                    (
                        "dec-upsm1",
                        Interpolate(
                            scale_factor=2, mode="bilinear", align_corners=True
                        ),
                    ),
                    (
                        "dec-resn1-1",
                        BasicBlock(
                            512,
                            256,
                            downsample=nn.Sequential(
                                nn.Conv2d(512, 256, kernel_size=1, bias=False),
                            ),
                            dilation=1,
                        ),
                    ),
                    (
                        "dec-resn1-2",
                        BasicBlock(
                            256,
                            256,
                            dilation=1,
                        ),
                    ),
                    (
                        "dec-upsm2",
                        Interpolate(
                            scale_factor=2, mode="bilinear", align_corners=True
                        ),
                    ),
                    (
                        "dec-resn2-1",
                        BasicBlock(
                            256,
                            128,
                            downsample=nn.Sequential(
                                nn.Conv2d(256, 128, kernel_size=1, bias=False),
                            ),
                            dilation=1,
                        ),
                    ),
                    (
                        "dec-resn2-2",
                        BasicBlock(
                            128,
                            128,
                            dilation=1,
                        ),
                    ),
                    (
                        "dec-upsm3",
                        Interpolate(
                            scale_factor=2, mode="bilinear", align_corners=True
                        ),
                    ),
                    (
                        "dec-resn3-1",
                        BasicBlock(
                            128,
                            64,
                            downsample=nn.Sequential(
                                nn.Conv2d(128, 64, kernel_size=1, bias=False),
                            ),
                            dilation=1,
                        ),
                    ),
                    (
                        "dec-resn3-2",
                        BasicBlock(
                            64,
                            64,
                            dilation=1,
                        ),
                    ),
                    (
                        "dec-upsm4",
                        Interpolate(
                            scale_factor=2, mode="bilinear", align_corners=True
                        ),
                    ),
                    (
                        "dec-resn4-1",
                        BasicBlock(
                            64,
                            32,
                            downsample=nn.Sequential(
                                nn.Conv2d(64, 32, kernel_size=1, bias=False),
                            ),
                            dilation=1,
                        ),
                    ),
                    (
                        "dec-resn4-2",
                        BasicBlock(
                            32,
                            32,
                            dilation=1,
                        )
                    )
                ]
            )
        )

        self.in_hand_conv = nn.Sequential(OrderedDict([
            ('cnn_conv1', nn.Conv2d(patch_shape[0], 32, kernel_size=3)),
            ('cnn_pool1', nn.MaxPool2d(2)),
            ('cnn_relu1', nn.ReLU(inplace=True)),
            ('cnn_conv2', nn.Conv2d(32, 64, kernel_size=3)),
            ('cnn_relu2', nn.ReLU(inplace=True)),
            ('cnn_conv3', nn.Conv2d(64, 64, kernel_size=3)),
            ('cnn_relu3', nn.ReLU(inplace=True)),
        ]))
        place_conv_out_size = self._getInHandConvOut(patch_shape)
        self.place_filter_fc = nn.Sequential(OrderedDict([
            ('place_fc1', nn.Linear(place_conv_out_size, 1024)),
            ('place_fc_relu1', nn.ReLU(inplace=True)),
            ('place_fc2', nn.Linear(1024, 32 * 32 + 32))
        ]))
        self.post_dynamic = nn.Sequential(OrderedDict([
            ('pd-relu', nn.ReLU(inplace=True))
        ]))

        self.pick_q_values = nn.Conv2d(32, 1, kernel_size=1, stride=1, bias=False)
        self.place_q_values = nn.Conv2d(32, 1, kernel_size=1, stride=1, bias=False)

        self._init_weights()

    def _getInHandConvOut(self, shape):
        o = self.in_hand_conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, obs, in_hand):
        df_channel = 32
        out_enc = self._encoder(obs)
        out_dec = self._decoder(out_enc)
        pick_q_values = self.pick_q_values(out_dec)

        in_hand_out = self.in_hand_conv(in_hand)
        in_hand_flatten = in_hand_out.view(in_hand.size()[0], -1)
        place_filter_weights = self.place_filter_fc(in_hand_flatten)

        place_filter_weight = place_filter_weights[:, :df_channel * df_channel].reshape(obs.size()[0], df_channel,
                                                                                        df_channel, 1, 1)
        place_filter_bias = place_filter_weights[:, df_channel * df_channel:].reshape(obs.size()[0], df_channel)
        place_filter_weight = place_filter_weight.reshape(obs.size()[0] * df_channel, df_channel, 1, 1)
        place_filter_bias = place_filter_bias.reshape(obs.size()[0] * df_channel)
        place_feature_map = out_dec.reshape(1, out_dec.size(0) * out_dec.size(1),
                                                      out_dec.size(2), out_dec.size(3))
        place_feature_map = F.conv2d(place_feature_map, weight=place_filter_weight, bias=place_filter_bias,
                                     groups=obs.size(0))
        place_feature_map = place_feature_map.reshape(obs.size(0), -1, place_feature_map.size(2),
                                                      place_feature_map.size(3))
        place_feature_map = self.post_dynamic(place_feature_map)
        place_q_values = self.place_q_values(place_feature_map)

        q_values = torch.cat((pick_q_values, place_q_values), dim=1)
        return q_values

class ResU(nn.Module):
    def __init__(self, n_input_channel=1, n_primitives=1, patch_shape=(1, 24, 24), domain_shape=(1, 100, 100)):
        super().__init__()
        self.conv_down_1 = nn.Sequential(
            OrderedDict(
                [
                    (
                        "enc-conv0",
                        nn.Conv2d(
                            n_input_channel,
                            32,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                        ),
                    ),
                    ("enc-relu0", nn.ReLU(inplace=True)),
                    (
                        'enc-res1',
                        BasicBlock(
                            32, 32,
                            dilation=1
                        )
                    )
                ]
            )
        )
        self.conv_down_2 = nn.Sequential(
            OrderedDict(
                [
                    (
                        'enc-pool2',
                        nn.MaxPool2d(2)
                    ),
                    (
                        'enc-res2',
                        BasicBlock(
                            32, 64,
                            downsample=nn.Sequential(
                                nn.Conv2d(32, 64, kernel_size=1, bias=False),
                            ),
                            dilation=1
                        )
                    )
                ]
            )
        )
        self.conv_down_4 = nn.Sequential(
            OrderedDict(
                [
                    (
                        'enc-pool3',
                        nn.MaxPool2d(2)
                    ),
                    (
                        'enc-res3',
                        BasicBlock(
                            64, 128,
                            downsample=nn.Sequential(
                                nn.Conv2d(64, 128, kernel_size=1, bias=False),
                            ),
                            dilation=1
                        )
                    )
                ]
            )
        )
        self.conv_down_8 = nn.Sequential(
            OrderedDict(
                [
                    (
                        'enc-pool4',
                        nn.MaxPool2d(2)
                    ),
                    (
                        'enc-res4',
                        BasicBlock(
                            128, 256,
                            downsample=nn.Sequential(
                                nn.Conv2d(128, 256, kernel_size=1, bias=False),
                            ),
                            dilation=1
                        )
                    )
                ]
            )
        )
        self.conv_down_16 = nn.Sequential(
            OrderedDict(
                [
                    (
                        'enc-pool5',
                        nn.MaxPool2d(2)
                    ),
                    (
                        'enc-res5',
                        BasicBlock(
                            256, 256,
                            dilation=1
                        )
                    )
                ]
            )
        )
        self.conv_up_8 = nn.Sequential(
            OrderedDict(
                [
                    (
                        'dec-res1',
                        BasicBlock(
                            512, 128,
                            downsample=nn.Sequential(
                                nn.Conv2d(512, 128, kernel_size=1, bias=False),
                            ),
                            dilation=1
                        )
                    )
                ]
            )
        )
        self.conv_up_4 = nn.Sequential(
            OrderedDict(
                [
                    (
                        'dec-res2',
                        BasicBlock(
                            256, 64,
                            downsample=nn.Sequential(
                                nn.Conv2d(256, 64, kernel_size=1, bias=False),
                            ),
                            dilation=1
                        )
                    )
                ]
            )
        )
        self.conv_up_2 = nn.Sequential(
            OrderedDict(
                [
                    (
                        'dec-res3',
                        BasicBlock(
                            128, 32,
                            downsample=nn.Sequential(
                                nn.Conv2d(128, 32, kernel_size=1, bias=False),
                            ),
                            dilation=1
                        )
                    )
                ]
            )
        )
        self.conv_up_1 = nn.Sequential(
            OrderedDict(
                [
                    (
                        'dec-res1',
                        BasicBlock(
                            64, 32,
                            downsample=nn.Sequential(
                                nn.Conv2d(64, 32, kernel_size=1, bias=False),
                            ),
                            dilation=1
                        )
                    )
                ]
            )
        )

        self.in_hand_conv = nn.Sequential(OrderedDict([
            ('cnn_conv1', nn.Conv2d(patch_shape[0], 32, kernel_size=3)),
            ('cnn_pool1', nn.MaxPool2d(2)),
            ('cnn_relu1', nn.ReLU(inplace=True)),
            ('cnn_conv2', nn.Conv2d(32, 64, kernel_size=3)),
            ('cnn_relu2', nn.ReLU(inplace=True)),
            ('cnn_conv3', nn.Conv2d(64, 64, kernel_size=3)),
            ('cnn_relu3', nn.ReLU(inplace=True)),
        ]))
        in_hand_conv_out_size = self._getInHandConvOut(patch_shape)
        self.in_hand_filter_fc = nn.Sequential(OrderedDict([
            ('place_fc1', nn.Linear(in_hand_conv_out_size, 1024)),
            ('place_fc_relu1', nn.ReLU(inplace=True)),
            ('place_fc2', nn.Linear(1024, 32 * 32 + 32))
        ]))

        self.post_dynamic = nn.Sequential(OrderedDict([
            ('pd-relu', nn.ReLU(inplace=True))
        ]))

        self.pick_q_values = nn.Conv2d(32, 1, kernel_size=1, stride=1)
        self.place_q_values = nn.Conv2d(32, 1, kernel_size=1, stride=1)

        for m in self.named_modules():
            if isinstance(m[1], nn.Conv2d):
                # nn.init.kaiming_normal_(m[1].weight.data)
                nn.init.xavier_normal_(m[1].weight.data)
            elif isinstance(m[1], nn.BatchNorm2d):
                m[1].weight.data.fill_(1)
                m[1].bias.data.zero_()

    def _getInHandConvOut(self, shape):
        o = self.in_hand_conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, obs, in_hand):
        df_channel = 32
        feature_map_1 = self.conv_down_1(obs)
        feature_map_2 = self.conv_down_2(feature_map_1)
        feature_map_4 = self.conv_down_4(feature_map_2)
        feature_map_8 = self.conv_down_8(feature_map_4)
        feature_map_16 = self.conv_down_16(feature_map_8)
        feature_map_up_8 = self.conv_up_8(torch.cat((feature_map_8, F.interpolate(feature_map_16, size=16, mode='bilinear', align_corners=False)), dim=1))
        feature_map_up_4 = self.conv_up_4(torch.cat((feature_map_4, F.interpolate(feature_map_up_8, size=32, mode='bilinear', align_corners=False)), dim=1))
        feature_map_up_2 = self.conv_up_2(torch.cat((feature_map_2, F.interpolate(feature_map_up_4, size=64, mode='bilinear', align_corners=False)), dim=1))
        feature_map_up_1 = self.conv_up_1(torch.cat((feature_map_1, F.interpolate(feature_map_up_2, size=128, mode='bilinear', align_corners=False)), dim=1))

        in_hand_out = self.in_hand_conv(in_hand)
        in_hand_flatten = in_hand_out.view(in_hand.size()[0], -1)
        in_hand_filter_weights = self.in_hand_filter_fc(in_hand_flatten)

        in_hand_filter_weight = in_hand_filter_weights[:, :df_channel * df_channel].reshape(obs.size()[0], df_channel,
                                                                                        df_channel, 1, 1)
        in_hand_filter_bias = in_hand_filter_weights[:, df_channel * df_channel:].reshape(obs.size()[0], df_channel)
        in_hand_filter_weight = in_hand_filter_weight.reshape(obs.size()[0] * df_channel, df_channel, 1, 1)
        in_hand_filter_bias = in_hand_filter_bias.reshape(obs.size()[0] * df_channel)
        place_feature_map = feature_map_up_1.reshape(1, feature_map_up_1.size(0) * feature_map_up_1.size(1),
                                                      feature_map_up_1.size(2), feature_map_up_1.size(3))
        place_feature_map = F.conv2d(place_feature_map, weight=in_hand_filter_weight, bias=in_hand_filter_bias,
                                     groups=obs.size(0))
        place_feature_map = place_feature_map.reshape(obs.size(0), -1, place_feature_map.size(2),
                                                      place_feature_map.size(3))
        place_feature_map = self.post_dynamic(place_feature_map)

        place_q_values = self.place_q_values(place_feature_map)
        pick_q_values = self.pick_q_values(feature_map_up_1)
        q_values = torch.cat((pick_q_values, place_q_values), dim=1)

        return q_values

class ResUCat(nn.Module):
    def __init__(self, n_input_channel=1, n_primitives=1, patch_shape=(1, 24, 24), domain_shape=(1, 100, 100)):
        super().__init__()
        self.conv_down_1 = nn.Sequential(
            OrderedDict(
                [
                    (
                        "enc-conv0",
                        nn.Conv2d(
                            n_input_channel,
                            32,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                        ),
                    ),
                    ("enc-relu0", nn.ReLU(inplace=True)),
                    (
                        'enc-res1',
                        BasicBlock(
                            32, 32,
                            dilation=1
                        )
                    )
                ]
            )
        )
        self.conv_down_2 = nn.Sequential(
            OrderedDict(
                [
                    (
                        'enc-pool2',
                        nn.MaxPool2d(2)
                    ),
                    (
                        'enc-res2',
                        BasicBlock(
                            32, 64,
                            downsample=nn.Sequential(
                                nn.Conv2d(32, 64, kernel_size=1, bias=False),
                            ),
                            dilation=1
                        )
                    )
                ]
            )
        )
        self.conv_down_4 = nn.Sequential(
            OrderedDict(
                [
                    (
                        'enc-pool3',
                        nn.MaxPool2d(2)
                    ),
                    (
                        'enc-res3',
                        BasicBlock(
                            64, 128,
                            downsample=nn.Sequential(
                                nn.Conv2d(64, 128, kernel_size=1, bias=False),
                            ),
                            dilation=1
                        )
                    )
                ]
            )
        )
        self.conv_down_8 = nn.Sequential(
            OrderedDict(
                [
                    (
                        'enc-pool4',
                        nn.MaxPool2d(2)
                    ),
                    (
                        'enc-res4',
                        BasicBlock(
                            128, 256,
                            downsample=nn.Sequential(
                                nn.Conv2d(128, 256, kernel_size=1, bias=False),
                            ),
                            dilation=1
                        )
                    )
                ]
            )
        )
        self.conv_down_16 = nn.Sequential(
            OrderedDict(
                [
                    (
                        'enc-pool5',
                        nn.MaxPool2d(2)
                    ),
                    (
                        'enc-res5',
                        BasicBlock(
                            256, 256,
                            dilation=1
                        )
                    )
                ]
            )
        )
        self.conv_cat_in_hand = nn.Sequential(
            OrderedDict(
                [
                    (
                        'enc-res6',
                        BasicBlock(
                            512, 256,
                            downsample=nn.Sequential(
                                nn.Conv2d(512, 256, kernel_size=1, bias=False),
                            ),
                            dilation=1
                        )
                    )
                ]
            )
        )

        self.conv_up_8 = nn.Sequential(
            OrderedDict(
                [
                    (
                        'dec-res1',
                        BasicBlock(
                            512, 128,
                            downsample=nn.Sequential(
                                nn.Conv2d(512, 128, kernel_size=1, bias=False),
                            ),
                            dilation=1
                        )
                    )
                ]
            )
        )
        self.conv_up_4 = nn.Sequential(
            OrderedDict(
                [
                    (
                        'dec-res2',
                        BasicBlock(
                            256, 64,
                            downsample=nn.Sequential(
                                nn.Conv2d(256, 64, kernel_size=1, bias=False),
                            ),
                            dilation=1
                        )
                    )
                ]
            )
        )
        self.conv_up_2 = nn.Sequential(
            OrderedDict(
                [
                    (
                        'dec-res3',
                        BasicBlock(
                            128, 32,
                            downsample=nn.Sequential(
                                nn.Conv2d(128, 32, kernel_size=1, bias=False),
                            ),
                            dilation=1
                        )
                    )
                ]
            )
        )
        self.conv_up_1 = nn.Sequential(
            OrderedDict(
                [
                    (
                        'dec-res1',
                        BasicBlock(
                            64, 32,
                            downsample=nn.Sequential(
                                nn.Conv2d(64, 32, kernel_size=1, bias=False),
                            ),
                            dilation=1
                        )
                    )
                ]
            )
        )

        self.in_hand_conv = nn.Sequential(OrderedDict([
            ('cnn_conv1', nn.Conv2d(patch_shape[0], 64, kernel_size=3)),
            ('cnn_relu1', nn.ReLU(inplace=True)),
            ('cnn_conv2', nn.Conv2d(64, 128, kernel_size=3)),
            ('cnn_relu2', nn.ReLU(inplace=True)),
            ('cnn_pool2', nn.MaxPool2d(2)),
            ('cnn_conv3', nn.Conv2d(128, 256, kernel_size=3)),
            ('cnn_relu3', nn.ReLU(inplace=True)),
        ]))

        # self.in_hand_conv = nn.Sequential(OrderedDict([
        #     ('cnn_conv1', nn.Conv2d(patch_shape[0], 32, kernel_size=3, padding=1)),
        #     ('cnn_relu1', nn.ReLU(inplace=True)),
        #     ('cnn_pool1', nn.MaxPool2d(2)),
        #
        #     ('cnn_conv2', nn.Conv2d(32, 64, kernel_size=3)),
        #     ('cnn_relu2', nn.ReLU(inplace=True)),
        #
        #     ('cnn_conv3', nn.Conv2d(64, 128, kernel_size=3)),
        #     ('cnn_relu3', nn.ReLU(inplace=True)),
        #
        #     ('cnn_conv4', nn.Conv2d(128, 256, kernel_size=3)),
        #     ('cnn_relu4', nn.ReLU(inplace=True)),
        #
        #     ('cnn_conv5', nn.Conv2d(256, 256, kernel_size=3)),
        #     ('cnn_relu5', nn.ReLU(inplace=True)),
        # ]))

        self.pick_q_values = nn.Conv2d(32, 1, kernel_size=1, stride=1)
        self.place_q_values = nn.Conv2d(32, 1, kernel_size=1, stride=1)

        for m in self.named_modules():
            if isinstance(m[1], nn.Conv2d):
                # nn.init.kaiming_normal_(m[1].weight.data)
                nn.init.xavier_normal_(m[1].weight.data)
            elif isinstance(m[1], nn.BatchNorm2d):
                m[1].weight.data.fill_(1)
                m[1].bias.data.zero_()

    def forward(self, obs, in_hand):
        df_channel = 32
        feature_map_1 = self.conv_down_1(obs)
        feature_map_2 = self.conv_down_2(feature_map_1)
        feature_map_4 = self.conv_down_4(feature_map_2)
        feature_map_8 = self.conv_down_8(feature_map_4)
        feature_map_16 = self.conv_down_16(feature_map_8)

        in_hand_out = self.in_hand_conv(in_hand)
        feature_map_16 = self.conv_cat_in_hand(torch.cat((feature_map_16, in_hand_out), dim=1))

        feature_map_up_8 = self.conv_up_8(torch.cat((feature_map_8, F.interpolate(feature_map_16, size=16, mode='bilinear', align_corners=False)), dim=1))
        feature_map_up_4 = self.conv_up_4(torch.cat((feature_map_4, F.interpolate(feature_map_up_8, size=32, mode='bilinear', align_corners=False)), dim=1))
        feature_map_up_2 = self.conv_up_2(torch.cat((feature_map_2, F.interpolate(feature_map_up_4, size=64, mode='bilinear', align_corners=False)), dim=1))
        feature_map_up_1 = self.conv_up_1(torch.cat((feature_map_1, F.interpolate(feature_map_up_2, size=128, mode='bilinear', align_corners=False)), dim=1))

        place_q_values = self.place_q_values(feature_map_up_1)
        pick_q_values = self.pick_q_values(feature_map_up_1)
        q_values = torch.cat((pick_q_values, place_q_values), dim=1)

        return q_values