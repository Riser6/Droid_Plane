import torch
import torch.nn as nn
import torch.nn.functional as F




class WarpingLayer(nn.Module):
    
    def __init__(self):
        super(WarpingLayer, self).__init__()

    def get_grid(self, x):
        torchHorizontal = torch.linspace(-1.0, 1.0, x.size(3)).view(1, 1, 1, x.size(3)).expand(x.size(0), 1, x.size(2), x.size(3))
        torchVertical = torch.linspace(-1.0, 1.0, x.size(2)).view(1, 1, x.size(2), 1).expand(x.size(0), 1, x.size(2), x.size(3))
        grid = torch.cat([torchHorizontal, torchVertical], 1)
        return grid
    
    def forward(self, x, flow):

        # WarpingLayer uses F.grid_sample, which expects normalized grid
        # we still output unnormalized flow for the convenience of comparing EPEs with FlowNet2 and original code
        # so here we need to denormalize the flow
        flow_for_grip = torch.zeros_like(flow).cuda()
        flow_for_grip[:,0,:,:] = flow[:,0,:,:] / ((flow.size(3) - 1.0) / 2.0)
        flow_for_grip[:,1,:,:] = flow[:,1,:,:] / ((flow.size(2) - 1.0) / 2.0)

        grid = (self.get_grid(x).cuda() + flow_for_grip).permute(0, 2, 3, 1)
        x_warp = F.grid_sample(x, grid)
        return x_warp

#warp_bsf = self.flow_warping(ref_bsf, flow_init)
#bsf_stack = torch.stack([bsf, warp_bsf], dim=1)
#bsf = self.tcea_fusion(bsf_stack)      # B,2,C,H,W -> B,1,C,H,W

class TCEA_Fusion(nn.Module):
    ''' Temporal Spatial Attention fusion module
    Temporal: correlation;
    Spatial: 1 pyramid levels.
    '''

    def __init__(self, nf=256, nframes=2, center=1):
        super(TCEA_Fusion, self).__init__()
        self.center = center
        # temporal attention (before fusion conv)
        self.tAtt_1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.tAtt_2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        # fusion conv: using 1x1 to save parameters and computation
        self.fea_fusion = nn.Conv2d(nframes * nf, nf, 1, 1, bias=True)

        # spatial attention (after fusion conv)
        self.sAtt_1 = nn.Conv2d(nframes * nf, nf, 1, 1, bias=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.avgpool = nn.AvgPool2d(3, stride=2, padding=1)
        self.sAtt_2 = nn.Conv2d(nf * 2, nf, 1, 1, bias=True)
        self.sAtt_3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.sAtt_4 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.sAtt_add_1 = nn.Conv2d(nf, nf, 1, 1, bias=True)
        self.sAtt_add_2 = nn.Conv2d(nf, nf, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, aligned_fea):
        B, N, C, H, W = aligned_fea.size()  # N video frames
        #### temporal attention



        emb_ref = self.tAtt_2(aligned_fea[:, self.center, :, :, :].clone())
        emb = self.tAtt_1(aligned_fea.view(-1, C, H, W)).view(B, N, -1, H, W)  # [B, N, C(nf), H, W]
        cor_l = []
        for i in range(N):
            emb_nbr = emb[:, i, :, :, :]
            cor_tmp = torch.sum(emb_nbr * emb_ref, 1).unsqueeze(1)  # B, 1, H, W
            cor_l.append(cor_tmp)
        cor_prob = torch.sigmoid(torch.cat(cor_l, dim=1))  # B, N, H, W
        cor_prob = cor_prob.unsqueeze(2).repeat(1, 1, C, 1, 1).view(B, -1, H, W)
        aligned_fea = aligned_fea.view(B, -1, H, W) * cor_prob
        #### fusion
        fea = self.lrelu(self.fea_fusion(aligned_fea))

        #### spatial attention
        att = self.lrelu(self.sAtt_1(aligned_fea))
        att_max = self.maxpool(att)
        att_avg = self.avgpool(att)
        att = self.lrelu(self.sAtt_2(torch.cat([att_max, att_avg], dim=1)))
        att = self.lrelu(self.sAtt_3(att))
        att = F.interpolate(att, scale_factor=2, mode='bilinear', align_corners=False)
        att = self.sAtt_4(att)
        att_add = self.sAtt_add_2(self.lrelu(self.sAtt_add_1(att)))
        att = torch.sigmoid(att)

        fea = fea * att * 2 + att_add
        return fea



if __name__ == "__main__":
    feature1 = torch.rand(1, 256, 160, 160).cuda()
    feature2 = torch.rand(1, 256, 160, 160).cuda()
    flow = torch.ones(1,2,160,160)*5
    flow = flow.cuda()

    flow_warping = WarpingLayer().cuda()
    tcea_fusion = TCEA_Fusion().cuda()

    print(feature1.shape)
    feature1_warp = flow_warping(feature1, flow)
    print("*****************")
    print(feature1_warp.shape)
    print(feature2.shape)
    feature_stack = torch.stack([feature2, feature1_warp], dim=1)
    print(feature_stack.shape)
    feature_fusion = tcea_fusion(feature_stack)
    print(feature_fusion.shape)

