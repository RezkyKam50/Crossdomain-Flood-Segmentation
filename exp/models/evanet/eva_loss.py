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
            pred_labels: Predicted labels. Shape: (Batch, Channel, Height, Width)
            heights:     GT elevation map.  Shape: (Batch, Channel, Height, Width)
            gt_labels:   GT labels.         Shape: (Batch, Channel, Height, Width)
                         255 = Ignore, 0 = Non-flood (Dry), 1 = Flood
        OUTPUT:
            final_loss: A single value.
        """

        ## Get Argmax Index for each prediction channel 
        pred_label_idx = torch.argmax(pred_labels, dim=1)
        
        ## Split the prediction channels (channel 0 = flood, channel 1 = dry)
        flood_pred, dry_pred = torch.tensor_split(pred_labels, 2, dim=1)
        
        ## Generate Pred Masks
        ones = torch.ones_like(gt_labels)
        flood_mask = torch.where(pred_label_idx == 0, ones, 0)
        dry_mask   = torch.where(pred_label_idx == 1, ones, 0)
        
        ## Mask Each Prediction Channel and Combine
        flood_pred  = flood_pred * flood_mask
        dry_pred    = -dry_pred  * dry_mask
        unified_pred = flood_pred + dry_pred

        ## Flatten prediction: (B, C, H*W, 1)
        pred_flat = torch.reshape(unified_pred, (unified_pred.shape[0], unified_pred.shape[1], -1, 1))

        ## Unfold GT labels: (B, 1, H*W, 9)
        gt_unfolded = self.unfold(gt_labels.float())
        gt_unfolded = gt_unfolded.permute(0, 2, 1)
        gt_unfolded = torch.unsqueeze(gt_unfolded, dim=1)

        ## Flatten heights: (B, 1, H*W, 1)
        h_flat = heights.clone()
        h_flat = torch.reshape(h_flat, (h_flat.shape[0], 1, -1, 1))

        ## Unfold heights: (B, 1, H*W, 9)
        h_unfolded = self.unfold(heights)
        h_unfolded = h_unfolded.permute(0, 2, 1)
        h_unfolded = torch.unsqueeze(h_unfolded, dim=1)

        # ----------------------------------------------------------------
        # Remap GT labels for score calculation:
        #   255 (ignore) -> 0  (neutral, will be masked out anyway)
        #   0   (dry)    -> -1 (matches original convention)
        #   1   (flood)  -> +1 (unchanged)
        # ----------------------------------------------------------------
        gt_remapped = gt_unfolded.clone().float()
        gt_remapped = torch.where(gt_unfolded == 255, torch.zeros_like(gt_remapped), gt_remapped)  # ignore -> 0
        gt_remapped = torch.where(gt_unfolded == 0,  -torch.ones_like(gt_remapped),  gt_remapped)  # dry    -> -1
        # flood (1) stays as 1

        ## Calculate Score
        score = 1 - (gt_remapped * pred_flat)

        ## Calculate Elevation Delta
        delta_unfolded = h_flat - h_unfolded

        ## Build masks using NEW label convention on the original gt_unfolded
        ones  = torch.ones_like(gt_unfolded)
        zeros = torch.zeros_like(gt_unfolded)

        # Ignore mask: exclude pixels labelled 255
        ignore_mask   = torch.where(gt_unfolded == 255, zeros, ones)

        # Elevation direction masks
        pos_elev_mask = torch.where(delta_unfolded > 0, ones, zeros)
        neg_elev_mask = torch.where(delta_unfolded < 0, ones, zeros)

        # GT class masks (new convention)
        gt_flood_mask = torch.where(gt_unfolded == 1, ones, zeros)
        gt_dry_mask   = torch.where(gt_unfolded == 0, ones, zeros)

        # Elevation consistency masks (same logic as before)
        flood_pos_elev_mask = 1 - (gt_flood_mask * pos_elev_mask)
        dry_neg_elev_mask   = 1 - (gt_dry_mask   * neg_elev_mask)

        weight = torch.ones_like(gt_unfolded)

        loss = (weight * ignore_mask * flood_pos_elev_mask * dry_neg_elev_mask) * score

        n_valid = ignore_mask.sum().clamp(min=1)
        return torch.sum(loss) / n_valid
    

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