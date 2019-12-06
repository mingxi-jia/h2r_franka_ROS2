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