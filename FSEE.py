class FSEE(nn.Module):
    def __init__(self, channel, cutoff_frequency=0.05,
                 use_attention=True, attention_type="cross",
                 attention_channels=64) -> None:
        super().__init__()

        self.cutoff_frequency = cutoff_frequency
        self.channel = channel
        self.use_attention = use_attention
        self.attention_type = attention_type

        if self.use_attention:
            self.attention_channels = attention_channels

            # self.channel_attention = nn.Sequential(
            #     nn.AdaptiveAvgPool2d(1),
            #     nn.Conv2d(channel, channel // 4, kernel_size=1),
            #     nn.ReLU(inplace=True),
            #     nn.Conv2d(channel // 4, channel, kernel_size=1),
            #     nn.Sigmoid()
            # )
            self.cbam = CBAM(channel, reduction=4)

            if attention_type == "cross":
                self.cross_attn_low2high = AttentionBlock(channel, attention_channels)
                self.cross_attn_high2low = AttentionBlock(channel, attention_channels)
            elif attention_type == "self":
                self.self_attn = AttentionBlock(2 * channel, attention_channels)

            self.fusion_conv = nn.Sequential(
                nn.Conv2d(2 * channel, channel, kernel_size=1),
                nn.BatchNorm2d(channel),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        device = x.device

        x_freq = torch.fft.fft2(x)
        x_freq_shifted = torch.fft.fftshift(x_freq)

        h, w = x.shape[2], x.shape[3]
        cy, cx = h // 2, w // 2
        y, x_grid = torch.meshgrid(torch.arange(h), torch.arange(w))
        y = y.to(device)
        x_grid = x_grid.to(device)

        freq_map = torch.sqrt((x_grid - cx).float() ** 2 + (y - cy).float() ** 2)
        cutoff = self.cutoff_frequency * max(h, w)
        high_pass_filter = (freq_map > cutoff).float().to(device)
        low_pass_filter = 1 - high_pass_filter 

        x_freq_high = x_freq_shifted * high_pass_filter
        x_freq_low = x_freq_shifted * low_pass_filter

        x_high = torch.abs(torch.fft.ifft2(torch.fft.ifftshift(x_freq_high)))
        x_low = torch.abs(torch.fft.ifft2(torch.fft.ifftshift(x_freq_low)))

        if self.use_attention:
            # channel_attn = self.channel_attention(x)
            # x_high = x_high * channel_attn
            # x_low = x_low * channel_attn
            x_high = self.cbam(x_high)
            x_low = self.cbam(x_low)

            if self.attention_type == "cross":
                # print("cross")
                enhanced_high = self.cross_attn_low2high(x_low, x_high)
                enhanced_low = self.cross_attn_high2low(x_high, x_low)
            elif self.attention_type == "self":
                combined = torch.cat([x_low, x_high], dim=1)
                enhanced = self.self_attn(combined, combined)
                enhanced_low, enhanced_high = torch.split(enhanced, self.channel, dim=1)

            combined_features = torch.cat([enhanced_low, enhanced_high], dim=1)
            x_filtered = self.fusion_conv(combined_features)

            x_filtered = x_filtered + x
        else:
            x_filtered = x_high

        return x_filtered
