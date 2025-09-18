def CIF_from_ground_truth(targets, img_shape, feat_shape, patch_size, device, B):
    """

    Args:
        targets: dict  'cls' (N,1), 'bboxes' (N,4), 'batch_idx' (N,)
        img_shape:  (H, W)
        feat_shape:  (Hf, Wf)
        patch_size: patch 
        device: torch.device

    Returns:
        golden_map: [B, 1, N_high, N_low]
    """
    # cls_all = targets['cls'].squeeze(1)  # [N]
    cls_all = targets['cls'].view(-1)
    bbox_all = targets['bboxes']  # [N, 4]
    batch_idx = targets['batch_idx']  # [N]

    H, W = img_shape
    Hf, Wf = feat_shape
    N_high = (H // patch_size) * (W // patch_size)
    N_low = (Hf // patch_size) * (Wf // patch_size)

    golden_map = torch.zeros((B, 1, N_high, N_low), device=device)

    for b in range(B):
        idxs = (batch_idx == b).nonzero(as_tuple=True)[0]
        cls_b = cls_all[idxs]
        bbox_b = bbox_all[idxs]
        num_objs = len(cls_b)

        patch_groups_high = []
        patch_groups_low = []

        for i in range(num_objs):
            cls_id = int(cls_b[i].item())
            cx, cy, bw, bh = bbox_b[i]

            cx_img, cy_img, bw_img, bh_img = cx * W, cy * H, bw * W, bh * H
            x1, x2 = int((cx_img - bw_img / 2) // patch_size), int((cx_img + bw_img / 2) // patch_size)
            y1, y2 = int((cy_img - bh_img / 2) // patch_size), int((cy_img + bh_img / 2) // patch_size)
            x1, x2 = max(0, x1), min(W // patch_size - 1, x2)
            y1, y2 = max(0, y1), min(H // patch_size - 1, y2)
            patch_high = [yy * (W // patch_size) + xx for yy in range(y1, y2 + 1) for xx in range(x1, x2 + 1)]

            scale_h, scale_w = H / Hf, W / Wf
            cx_f, cy_f, bw_f, bh_f = cx_img / scale_w, cy_img / scale_h, bw_img / scale_w, bh_img / scale_h
            xf1, xf2 = int((cx_f - bw_f / 2) // patch_size), int((cx_f + bw_f / 2) // patch_size)
            yf1, yf2 = int((cy_f - bh_f / 2) // patch_size), int((cy_f + bh_f / 2) // patch_size)
            xf1, xf2 = max(0, xf1), min(Wf // patch_size - 1, xf2)
            yf1, yf2 = max(0, yf1), min(Hf // patch_size - 1, yf2)
            patch_low = [yy * (Wf // patch_size) + xx for yy in range(yf1, yf2 + 1) for xx in range(xf1, xf2 + 1)]

            patch_groups_high.append((cls_id, patch_high))
            patch_groups_low.append((cls_id, patch_low))

        for i in range(num_objs):
            for j in range(num_objs):
                if patch_groups_high[i][0] == patch_groups_low[j][0]:  # 类别一致
                    p_high = patch_groups_high[i][1]
                    p_low = patch_groups_low[j][1]
                    for p1 in p_high:
                        for p2 in p_low:
                            golden_map[b, 0, p1, p2] = 1.0

    return golden_map
