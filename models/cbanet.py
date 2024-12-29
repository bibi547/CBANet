import torch
import torch.nn as nn
from .dgcnn_utils import STN, Backbone, SharedMLP1d, EdgeConv, knn
from .fps_utils import center_fps


class CBANet(nn.Module):
    def __init__(self, args):
        super(CBANet, self).__init__()
        self.num_points = args.num_points
        self.query_num = args.query_num
        self.out_channels = args.output_channels
        self.k = args.k
        if args.use_stn:
            self.stn = STN(args.k, args.norm)
        self.backbone = Backbone(args)

        self.bmap_decoder = SharedMLP1d([1344, 256], args.norm)
        self.bmap_head = nn.Sequential(SharedMLP1d([256, 256], args.norm),
                                       nn.Dropout(args.dropout),
                                       SharedMLP1d([256, 128], args.norm),
                                       nn.Conv1d(128, 3, kernel_size=1))
        self.dmap_decoder = SharedMLP1d([1344, 256], args.norm)
        self.dmap_head = nn.Sequential(SharedMLP1d([256, 256], args.norm),
                                       nn.Dropout1d(args.dropout),
                                       SharedMLP1d([256, 128], args.norm),
                                       nn.Conv1d(128, 1, kernel_size=1))

        self.decoder = SharedMLP1d([1344, 256], args.norm)
        self.mask_decoder = SharedMLP1d([256, 256], args.norm)

        self.pw_conv = EdgeConv([256 * 2, 256], self.k, args.norm)
        self.cw_conv = EdgeConv([256 * 2, 256], self.k, args.norm)

        self.class_head = nn.Sequential(SharedMLP1d([256, 256], args.norm),
                                        nn.Dropout(args.dropout),
                                        SharedMLP1d([256, 128], args.norm),
                                        nn.Conv1d(128, args.output_channels, kernel_size=1))
        self.score_head = nn.Sequential(SharedMLP1d([256, 256], args.norm),
                                        nn.Dropout(args.dropout),
                                        SharedMLP1d([256, 128], args.norm),
                                        nn.Conv1d(128, 1, kernel_size=1))
        self.mask_head = nn.Sequential(SharedMLP1d([self.query_num, self.query_num], args.norm),
                                       nn.Dropout(args.dropout),
                                       nn.Conv1d(self.query_num, self.query_num, kernel_size=1),
                                       nn.Sigmoid())

    def forward(self, x, eid):
        device = x.device
        batch_size = x.size(0)
        p = x[:, :3, :].contiguous()  # xyz

        if hasattr(self, "stn"):
            if not hasattr(self, 'c'):
                self.c = torch.zeros((x.shape[0], 15, 15), dtype=torch.float32, device=device)
                for i in range(0, 15, 3):
                    self.c[:, i:i + 3, i:i + 3] = 1

            t = self.stn(x[:, :3, :].contiguous())
            t = t.repeat(1, 5, 5)  # (batch_size, 15, 15)
            t1 = self.c * t
            x = torch.bmm(t1, x)
        else:
            t = torch.ones((1, 1), device=device)

        feats = self.backbone(x)  # (b, 1344, 10000)

        bmap_feats = self.bmap_decoder(feats)
        dmap_feats = self.dmap_decoder(feats)
        bmap_out = self.bmap_head(bmap_feats)
        dmap_out = self.dmap_head(dmap_feats)

        # #==========================mask stage===============================================
        sp_masks, sp_probs, g_ps = torch.zeros([batch_size, self.num_points, self.query_num]), torch.zeros([batch_size, 17, self.query_num]), torch.zeros([batch_size, 3, self.query_num])
        score = torch.zeros([batch_size, 1, self.query_num])
        all_idx = torch.zeros([batch_size, self.query_num])  # sampled idx
        if eid > 19:

            feats = self.decoder(feats)  # (b, 256, 10000)
            mask_feats = self.mask_decoder(feats)  # (b, 256, 10000)

            dmap = dmap_out.detach()
            bmap = bmap_out.detach().argmax(1)  # (b, 1, 10000)

            g_fs = []
            g_ps = []
            all_idx = []
            for i in range(0, batch_size):
                dm = dmap[i].squeeze()
                t_idx = torch.nonzero(dm > 0.2).squeeze()  # M tooth points
                ps = p[i].T[t_idx]  # (M, 3)
                fs = feats[i].T[t_idx]  # (M, 256)
                ps_idx = knn(ps.T.unsqueeze(0).contiguous(), self.k)  # 1, 20, M
                fs = self.pw_conv(fs.T.unsqueeze(0).contiguous(), ps_idx)  # 1, 256, M

                select_idx = center_fps(dm[t_idx], ps, self.query_num)
                select_ps = ps[select_idx].T.unsqueeze(0)  # 1, 3, 100
                select_fs = fs[:, :, select_idx]  # 1, 256, 100

                g_fs.append(select_fs)
                g_ps.append(select_ps)
                sampl_idx = t_idx[select_idx]
                all_idx.append(sampl_idx)

            g_fs = torch.concat(g_fs, dim=0)  # b, 256, 100
            g_ps = torch.concat(g_ps, dim=0)  # b, 3, 100

            idx = knn(g_ps.contiguous(), self.k)
            g_feats = self.cw_conv(g_fs, idx)  # b, 256, 100

            mask_feats = torch.einsum('bdn,bdm->bnm', g_feats, mask_feats)  # b, 100, 10000

            sp_probs = self.class_head(g_feats)  # b, 17, 100
            score = self.score_head(g_feats)  # b, 1, 100
            sp_masks = self.mask_head(mask_feats)

        return bmap_out, dmap_out, sp_masks, sp_probs, score, all_idx

    def inference(self, x):
        device = x.device
        batch_size = x.size(0)
        p = x[:, :3, :].contiguous()  # xyz

        if hasattr(self, "stn"):
            if not hasattr(self, 'c'):
                self.c = torch.zeros((x.shape[0], 15, 15), dtype=torch.float32, device=device)
                for i in range(0, 15, 3):
                    self.c[:, i:i + 3, i:i + 3] = 1

            t = self.stn(x[:, :3, :].contiguous())
            t = t.repeat(1, 5, 5)  # (batch_size, 15, 15)
            t1 = self.c * t
            x = torch.bmm(t1, x)
        else:
            t = torch.ones((1, 1), device=device)

        feats = self.backbone(x)  # (b, 1344, 10000)

        bmap_feats = self.bmap_decoder(feats)
        dmap_feats = self.dmap_decoder(feats)
        bmap_out = self.bmap_head(bmap_feats)
        dmap_out = self.dmap_head(dmap_feats)

        # #==========================mask stage===============================================

        feats = self.decoder(feats)  # (b, 256, 10000)
        mask_feats = self.mask_decoder(feats)  # (b, 256 * 3, 10000)

        dmap = dmap_out.detach()
        bmap = bmap_out.detach().argmax(1)  # (b, 1, 10000)

        g_fs = []
        g_ps = []
        all_idx = []
        for i in range(0, batch_size):
            dm = dmap[i].squeeze()
            t_idx = torch.nonzero(dm > 0.2).squeeze()  # M tooth points
            ps = p[i].T[t_idx]  # (M, 3)
            fs = feats[i].T[t_idx]  # (M, 256)
            ps_idx = knn(ps.T.unsqueeze(0).contiguous(), self.k)  # 1, 30, M
            fs = self.pw_conv(fs.T.unsqueeze(0).contiguous(), ps_idx)  # 1, 256, M

            select_idx = center_fps(dm[t_idx], ps, self.query_num)
            select_ps = ps[select_idx].T.unsqueeze(0)  # 1, 3, 100
            select_fs = fs[:, :, select_idx]  # 1, 256, 100

            g_fs.append(select_fs)
            g_ps.append(select_ps)
            sampl_idx = t_idx[select_idx]
            all_idx.append(sampl_idx)

        g_fs = torch.concat(g_fs, dim=0)  # b, 256, 100
        g_ps = torch.concat(g_ps, dim=0)  # b, 3, 100

        idx = knn(g_ps.contiguous(), self.k)
        g_feats = self.cw_conv(g_fs, idx)  # b, 256, 100

        mask_feats = torch.einsum('bdn,bdm->bnm', g_feats, mask_feats)  # b, 100, 10000

        sp_probs = self.class_head(g_feats)  # b, 17, 100
        score = self.score_head(g_feats)  # b, 1, 100
        sp_masks = self.mask_head(mask_feats)

        return bmap, dmap_out, sp_masks, sp_probs, score



