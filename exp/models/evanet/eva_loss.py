import torch
from torch import nn

class ElevationLoss(nn.Module):
    
    def __init__(self):
        super(ElevationLoss, self).__init__()
        self.unfold = torch.nn.Unfold(kernel_size=(3, 3), padding=1)
        self.relu = torch.nn.ReLU()
        
    def forward(self, pred_labels, heights, gt_labels):
        """
        INPUT:
            pred_labels: Predicted logits.  Shape: (B, 2, H, W)
            heights:     GT elevation map.  Shape: (B, 1, H, W)
            gt_labels:   GT labels.         Shape: (B, 1, H, W)
                         0=dry, 1=flood, 255=ignore
        """

        # --- Softmax + get flood probability (channel 1) ---
        pred_prob  = torch.softmax(pred_labels, dim=1)   # (B, 2, H, W)
        flood_prob = pred_prob[:, 1:2]                   # (B, 1, H, W) flood confidence
        dry_prob   = pred_prob[:, 0:1]                   # (B, 1, H, W) dry confidence

        # Signed prediction: +flood_prob for predicted flood, -dry_prob for predicted dry
        pred_label_idx = torch.argmax(pred_prob, dim=1, keepdim=True)  # (B, 1, H, W)
        flood_mask = (pred_label_idx == 1).float()
        dry_mask   = (pred_label_idx == 0).float()
        unified_pred = flood_prob * flood_mask - dry_prob * dry_mask   # (B, 1, H, W)

        # --- Flatten pred: (B, 1, H*W, 1) ---
        pred_flat = unified_pred.reshape(unified_pred.shape[0], 1, -1, 1)

        # --- Unfold GT labels: (B, 1, H*W, 9) ---
        gt_unfolded = self.unfold(gt_labels.float())           # (B, 9, H*W)
        gt_unfolded = gt_unfolded.permute(0, 2, 1).unsqueeze(1)  # (B, 1, H*W, 9)

        # --- Unfold heights: (B, 1, H*W, 9) ---
        h_flat     = heights.reshape(heights.shape[0], 1, -1, 1)
        h_unfolded = self.unfold(heights).permute(0, 2, 1).unsqueeze(1)

        # --- GT class masks (no remapping, direct comparison) ---
        gt_flood_mask = (gt_unfolded == 1).float()   # neighbor is flood
        gt_dry_mask   = (gt_unfolded == 0).float()   # neighbor is dry
        ignore_mask   = (gt_unfolded != 255).float() # exclude ignore pixels

        # --- Elevation consistency masks ---
        delta = h_flat - h_unfolded                          # positive = center higher than neighbor
        pos_elev_mask = (delta > 0).float()
        neg_elev_mask = (delta < 0).float()

        # Penalize: flood pixel that is HIGHER than its neighbor (water flows down)
        # Penalize: dry pixel that is LOWER than its neighbor
        flood_pos_elev_mask = 1 - (gt_flood_mask * pos_elev_mask)
        dry_neg_elev_mask   = 1 - (gt_dry_mask   * neg_elev_mask)

        # --- Score: 1 - (gt_signed * pred_signed) ---
        # Convert GT to signed: flood=+1, dry=-1, ignore=0
        gt_signed = gt_flood_mask - gt_dry_mask              # +1, -1, or 0 (ignore)
        score = 1 - (gt_signed * pred_flat)                  # (B, 1, H*W, 9)

        # --- Combined mask and loss ---
        combined_mask = ignore_mask * flood_pos_elev_mask * dry_neg_elev_mask
        loss = self.relu(combined_mask * score)

        n_valid = combined_mask.sum().clamp(min=1)
        return loss.sum() / n_valid
    

class ElevationLossWrapper:
    def __init__(self, heights, device):
        self.loss_fn = ElevationLoss()
        self.heights = heights
        self.device = device

    def __call__(self, output, target):
        # target comes in as (B, H, W) long from computeMetrics
        # ElevationLoss expects (B, 1, H, W) float
        target_4d = target.unsqueeze(1).float()
        return self.loss_fn(output, self.heights.to(self.device), target_4d)