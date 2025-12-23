import torch
from torch import nn
from torch_points_kernels import knn


def square_distance(src, dst):
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


class SharedMLP(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            transpose=False,
            padding_mode='zeros',
            bn=False,
            activation_fn=None
    ):
        super(SharedMLP, self).__init__()

        conv_fn = nn.ConvTranspose2d if transpose else nn.Conv2d

        self.conv = conv_fn(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding_mode=padding_mode
        )
        self.batch_norm = nn.BatchNorm2d(out_channels, eps=1e-6, momentum=0.99) if bn else None
        self.activation_fn = activation_fn

    def forward(self, input):
        x = self.conv(input)
        if self.batch_norm:
            x = self.batch_norm(x)
        if self.activation_fn:
            x = self.activation_fn(x)
        return x


class RFB_Branch(nn.Module):
    def __init__(self, in_channel, out_channel, neighbors):
        super(RFB_Branch, self).__init__()
        self.neighbors = neighbors
        self.mlp1 = SharedMLP(in_channel, out_channel // 2, activation_fn=nn.LeakyReLU(0.2))
        self.mlp2 = SharedMLP(10, out_channel // 2, bn=True, activation_fn=nn.ReLU())
        self.mlp3 = SharedMLP(2 * out_channel, out_channel, bn=True, activation_fn=nn.ReLU())

    def forward(self, coords, features, knn_output):
        features = self.mlp1(features)

        idx, dist = knn_output
        idx, dist = idx[..., :self.neighbors], dist[..., :self.neighbors]
        B, N, K = idx.size()
        extended_idx = idx.unsqueeze(1).expand(B, 3, N, K)
        extended_coords = coords.transpose(-2, -1).unsqueeze(-1).expand(B, 3, N, K)
        neighbors = torch.gather(extended_coords, 2, extended_idx)

        concat = torch.cat((
            extended_coords,
            neighbors,
            extended_coords - neighbors,
            dist.unsqueeze(-3)
        ), dim=-3)
        features = torch.cat((
            self.mlp2(concat),
            features.expand(B, -1, N, K)
        ), dim=-3)

        features = torch.cat(
            [torch.max(features, dim=-1, keepdim=True)[0],
             torch.mean(features, dim=-1, keepdim=True)
             ],
            dim=-3
        )
        features = self.mlp3(features)
        return features


class RFB(nn.Module):
    def __init__(self, in_channel, out_channel, neighbors):
        super(RFB, self).__init__()
        self.relu = nn.ReLU()
        self.max_neighbors = max(neighbors)
        self.shortcut = SharedMLP(in_channel, out_channel)
        self.branch0 = SharedMLP(in_channel, out_channel, bn=True, activation_fn=nn.ReLU())
        self.branch1 = RFB_Branch(in_channel, out_channel, neighbors[0])
        self.branch2 = RFB_Branch(in_channel, out_channel, neighbors[1])
        self.branch3 = RFB_Branch(in_channel, out_channel, neighbors[2])

        self.mlp_cat = SharedMLP(4 * out_channel, out_channel)

    def forward(self, coords, features):
        coords, features = coords.transpose(-1, -2), features.unsqueeze(-1)

        idx, dist = knn(coords.cpu().contiguous(), coords.cpu().contiguous(), self.max_neighbors)
        knn_output = [idx.cuda(), dist.cuda()]

        branch0 = self.branch0(features)
        branch1 = self.branch1(coords, features, knn_output)
        branch2 = self.branch2(coords, features, knn_output)
        branch3 = self.branch3(coords, features, knn_output)
        shortcut = self.shortcut(features)

        concat = torch.cat([
            branch0,
            branch1,
            branch2,
            branch3
        ], dim=-3)
        concat = self.mlp_cat(concat)
        return self.relu(concat + shortcut).squeeze(-1)


class Upsample(nn.Module):
    def __init__(self, in_channel, inter_channel=None, add_operation=False):
        super(Upsample, self).__init__()
        self.add_operation = add_operation
        self.inter_channel = inter_channel
        self.mlp_interpolated = nn.Conv1d(in_channel, inter_channel, 1)

    def forward(self, xyz1, xyz2, points2):

        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)

        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)   # B,4096,16
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm   # B,4096,3
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)
        interpolated_points = interpolated_points.permute(0, 2, 1)
        return self.mlp_interpolated(interpolated_points)
