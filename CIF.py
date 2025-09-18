class CIF(nn.Module):
    def __init__(self, in_channels, patch_size=patch_size) -> None:
        super().__init__()
        self.patch_size = patch_size
        img_channel, feat_channel = in_channels
        self.conv_img = nn.Sequential(
            nn.Conv2d(img_channel, 64, kernel_size=(7, 7), padding=3),
            nn.Conv2d(64, 1, kernel_size=(3, 3), padding=1)
        )
        self.conv_feamap = nn.Sequential(
            nn.Conv2d(feat_channel, feat_channel, kernel_size=(1, 1), stride=1)
        )

        self.unfold = nn.Unfold(kernel_size=(self.patch_size, self.patch_size),
                                stride=(self.patch_size, self.patch_size))
        self.resolution_trans = nn.Sequential(
            nn.Linear(self.patch_size * self.patch_size, 2 * self.patch_size * self.patch_size, bias=False),
            nn.Linear(2 * self.patch_size * self.patch_size, self.patch_size * self.patch_size, bias=False),
            nn.ReLU()
        )

    def forward(self, img, feamap):
        ini_img = self.conv_img(img)
        feamap = self.conv_feamap(feamap) / 16
        attentions = []
        unfold_img = self.unfold(ini_img).transpose(-1, -2)  # [B, Num_Patch, Patch_Size^2]
        unfold_img = self.resolution_trans(unfold_img)  # 同上
        for i in range(feamap.size()[1]):
            # unfold_img = self.unfold(ini_img[:, i:i + 1, :, :]).transpose(-1, -2)
            # unfold_img = self.resolution_trans(unfold_img)

            unfold_feamap = self.unfold(feamap[:, i:i + 1, :, :])
            unfold_feamap = self.resolution_trans(unfold_feamap.transpose(-1, -2)).transpose(-1, -2)

            att = torch.matmul(unfold_img, unfold_feamap) / (self.patch_size * self.patch_size)

            att = torch.unsqueeze(att, 1)

            attentions.append(att)

        attentions = torch.cat(attentions, dim=1)

        return attentions
