import torch
from torch import nn
import torch.nn.functional as F
from pointnet2_utils import PointNetSetAbstraction, PointNetFeaturePropagation
from mynet_utils import RFB, Upsample
from vit import TransformerLayer
from correlation import GBCF


class get_model(nn.Module):
    def __init__(self, num_classes):
        super(get_model, self).__init__()
        self.sa1 = PointNetSetAbstraction(1024, 10, 32, 9 + 3, [32, 32, 64], False)  # origianal radius 0.1
        self.sa2 = PointNetSetAbstraction(256, 40, 32, 64 + 3, [64, 64, 128], False)  # origianal radius 0.2
        self.sa3 = PointNetSetAbstraction(64, 80, 32, 128 + 3, [128, 128, 256], False)  # origianal radius 0.4
        self.sa4 = PointNetSetAbstraction(16, 100, 32, 256 + 3, [256, 256, 512], False)  # origianal radius 0.8

        # self.rfb1 = RFB(512, 512, [4, 9, 16])
        # self.rfb2 = RFB(128, 128, [9, 25, 49])

        self.fp4 = PointNetFeaturePropagation(768, [256, 256])
        self.fp3 = PointNetFeaturePropagation(384, [256, 256])
        self.fp2 = PointNetFeaturePropagation(320, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])
        # self.conv1 = nn.Conv1d(129, 128, 1)
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

        self.deep_supervision = Upsample(512, 2)

        self.TLayer2 = TransformerLayer(dim=256, heads=8, dim_head=64, mlp_dim=64)
        self.TLayer3 = TransformerLayer(dim=256, heads=8, dim_head=64, mlp_dim=64)
        self.TLayer4 = TransformerLayer(dim=512, heads=8, dim_head=64, mlp_dim=64)

        self.gbcf_4 = GBCF(512, 16)
        self.gbcf_3 = GBCF(256, 64)
        self.gbcf_2 = GBCF(256, 256)
        #self.pnp3d_1 = PnP3D(128)

    def forward(self, xyz):   # xyz: B*9*4096
        l0_points = xyz
        l0_xyz = xyz[:, :3, :]   # N,3,4096

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)   # B,3,16   B,512,16

        l4_points = self.TLayer4(l4_points)   
        l4_points = self.gbcf_4(l4_points)   
        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)

        l3_points = self.TLayer3(l3_points)   
        l3_points = self.gbcf_3(l3_points)   
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)

        l2_points = self.TLayer2(l2_points)   
        l2_points = self.gbcf_2(l2_points)   
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)

        # multi_scale = self.rfb2(l1_xyz, l1_points)
        #l1_points = self.pnp3d_1(l1_points)   

        multi_scale = self.fp1(l0_xyz, l1_xyz, None, l1_points)

        enhanced_multi_scale = multi_scale

        enhanced_multi_scale = self.drop1(F.relu(self.bn1(self.conv1(enhanced_multi_scale))))
        pred = self.conv2(enhanced_multi_scale)   # pred: B*2*4096
        
        pred_o = pred.transpose(2, 1).contiguous()   # # pred_o: B*4096*2
        pred = F.log_softmax(pred_o, dim=-1)
        return pred, pred_o   # pred, pred_o: B*4096*2


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, weight):
        total_loss = F.nll_loss(pred, target, weight=weight)

        return total_loss
