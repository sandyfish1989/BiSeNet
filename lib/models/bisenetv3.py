from utils import *


class BiSeNetV3(nn.Module):
    def __init__(self, n_classes, aux_mode='train'):
        super(BiSeNetV3, self).__init__()
        self.aux_mode = aux_mode
        self.detail = DetailBranch()
        self.segment = SegmentBranch()

        self.bga = SimBGABlock(128, 128)
        # aux1 aux2 辅助训练
        self.head = SegmentHead(128, 512, n_classes, up_factor=8)
        self.aux1 = SegmentHead(128, 128, n_classes, up_factor=8)
        self.aux2 = SegmentHead(64, 128, n_classes, up_factor=16)

    def forward(self, x):
        detail = self.detail(x)  # [bs, 128, 80, 80]
        feat8, feat16 = self.segment(x)
        feat_head = self.bga(detail, feat8)
        out = self.head(feat_head)

        if self.training:
            out_aux1 = self.aux1(feat8)
            out_aux2 = self.aux2(feat16)
            return out, out_aux1, out_aux2
        return out


if __name__ == "__main__":
    model = BiSeNetV3(n_classes=19)

    import copy
    import thop  # 获取计算参量

    model_tmp = copy.deepcopy(model)
    flops, params = thop.profile(model_tmp, inputs=(torch.randn(1, 3, 640, 640),))
    print("%.2fG" % (flops / 1e9), "%.2fM" % (params / 1e6))
