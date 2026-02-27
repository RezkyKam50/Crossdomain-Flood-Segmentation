from collections import OrderedDict
import torch.nn.functional as F
import torch
import torch.nn as nn
from models.prithvi_unet import PrithviUNet
from models.hydraunet.GhostNet import GhostModule
from models.hydraunet.PRCNPTN import PRCNPTNLayer
from models.hydraunet.Attention import *

# Dual Stream Classical UNet

# Reference from DS_Unet https://github.com/SebastianHafner/DS_UNet/blob/master/utils/networks.py
class DSGhostUnet(nn.Module):

    def __init__(self, cfg, use_prithvi=None, attn_scheme=None):
        super(DSGhostUnet, self).__init__()
        assert (cfg.DATASET.MODE == 'fusion')
        self._cfg = cfg
        out = cfg.MODEL.OUT_CHANNELS
        topology = cfg.MODEL.TOPOLOGY

        # sentinel-1 unet stream
        n_s1_bands = len(cfg.DATASET.SENTINEL1_BANDS)
        s1_in = n_s1_bands 
        self.s1_stream = UNet(cfg, n_channels=s1_in + 2, n_classes=out, topology=topology, enable_outc=False, attn_scheme=attn_scheme)
        self.n_s1_bands = n_s1_bands

        # sentinel-2 unet stream
        n_s2_bands = len(cfg.DATASET.SENTINEL2_BANDS)
        s2_in = n_s2_bands  
        self.s2_stream = UNet(cfg, n_channels=s2_in, n_classes=out, topology=topology, enable_outc=False, attn_scheme=attn_scheme)
        self.n_s2_bands = n_s2_bands

        # attention block
        # self.aux_se = CrossModalSqueezeExcite(
        #     aux_chs=2,
        #     s_chs=cfg.MODEL.TOPOLOGY[0]
        # )

        # self.s2_se = CrossModalSqueezeExcite(
        #     aux_chs=cfg.MODEL.TOPOLOGY[0],  # s1 attended as aux
        #     s_chs=cfg.MODEL.TOPOLOGY[0]
        # ) 


        # out block combining unet outputs
        self.use_prithvi = use_prithvi
            # prithvi
        if self.use_prithvi:
            self.prithvi = PrithviUNet(
                in_channels=n_s2_bands,
                out_channels=out,
                weights_path=cfg.MODEL.PRITHVI_PATH,
                device="cuda" if torch.cuda.is_available() else "cpu"
            ) # prithvi encoder + unet segmentation decoder
            out_dim = 2 * cfg.MODEL.TOPOLOGY[0]  # N channels x Topo First idx 
        else:
            out_dim = 2 * cfg.MODEL.TOPOLOGY[0] # N channels x Topo First idx 

        self.out_conv = OutConv(out_dim, out)

    def change_prithvi_trainability(self, trainable):
        if self.use_prithvi:
            self.prithvi.change_prithvi_trainability(trainable)

    # def forward(self, s1_img, s2_img, dem_img, water_occur): 

    #     del water_occur # 6, 224, 224 # unsqueeze(1) > 6, 1, 224, 224 
    #     del dem_img # 6, 1, 224, 224

    #     s1_feature = self.s1_stream(s1_img)
    #     s2_feature = self.s2_stream(s2_img)
 
    #     if self.use_prithvi:
    #         prithvi_features = self.prithvi(s2_img)
    #         fusion = torch.cat((s1_feature, s2_feature, prithvi_features), dim=1) # 2 ch + 2 ch prithvi
    #     else:
    #         fusion = torch.cat((s1_feature, s2_feature), dim=1) # 2 ch

    #     out = self.out_conv(fusion)
    #     return out

    
    def forward(self, s1_img, s2_img, dem_img, water_occur):
        
        # print(f"JRC SH: {water_occur.shape}")
        # print(f"DEM SH: {dem_img.shape}")
        s1_feature = torch.cat([dem_img, water_occur, s1_img], dim=1) # B, 2 + 2, H, W
        s1_feature = self.s1_stream(s1_img)
        s2_feature = self.s2_stream(s2_img)     

        # aux attention on S1 features
        # aux = torch.cat([dem_img, water_occur.unsqueeze(1)], dim=1)  # [B, 2, H, W]
        # s1_feature = self.aux_se(s1_feature, aux)                    # attended S1
        # s2_feature = self.s2_se(s2_feature, s1_feature)  
        
        if self.use_prithvi:
            prithvi_features = self.prithvi(s2_img)
            fusion = torch.cat((s1_feature, s2_feature, prithvi_features), dim=1)
        else:
            fusion = torch.cat((s1_feature, s2_feature), dim=1)

        out = self.out_conv(fusion)
        return out


class UNet(nn.Module):
    def __init__(self, cfg, n_channels=None, n_classes=None, topology=None, enable_outc=True, attn_scheme=None):

        self._cfg = cfg

        n_channels = cfg.MODEL.IN_CHANNELS if n_channels is None else n_channels
        n_classes = cfg.MODEL.OUT_CHANNELS if n_classes is None else n_classes
        topology = cfg.MODEL.TOPOLOGY if topology is None else topology

        super(UNet, self).__init__()

        first_chan = topology[0]
        self.inc = InConv(n_channels, first_chan, DoubleConv)
        self.enable_outc = enable_outc
        self.outc = OutConv(first_chan, n_classes)

        # Variable scale
        down_topo = topology
        down_dict = OrderedDict()
        n_layers = len(down_topo)
        up_topo = [first_chan]  # topography upwards
        up_dict = OrderedDict()

        # Downward layers
        for idx in range(n_layers):
            is_not_last_layer = idx != n_layers - 1
            in_dim = down_topo[idx]
            out_dim = down_topo[idx + 1] if is_not_last_layer else down_topo[idx]  # last layer

            layer = Down(in_dim, out_dim, DoubleConv)

            print(f'down{idx + 1}: in {in_dim}, out {out_dim}')
            down_dict[f'down{idx + 1}'] = layer
            up_topo.append(out_dim)
        self.down_seq = nn.ModuleDict(down_dict)

        # Upward layers
        for idx in reversed(range(n_layers)):
            is_not_last_layer = idx != 0
            x1_idx = idx
            x2_idx = idx - 1 if is_not_last_layer else idx
            in_dim = up_topo[x1_idx] * 2
            out_dim = up_topo[x2_idx]

            layer = Up(in_dim, out_dim, DoubleConv, attn_scheme=attn_scheme)

            print(f'up{idx + 1}: in {in_dim}, out {out_dim}')
            up_dict[f'up{idx + 1}'] = layer

        self.up_seq = nn.ModuleDict(up_dict)

    def forward(self, x1, x2=None, x3=None):
        if x2 is None and x3 is None:
            x = x1
        elif x3 is None:
            x = torch.cat((x1, x2), 1)
        else:
            x = torch.cat((x1, x2, x3), 1)   

        x1 = self.inc(x)

        inputs = [x1]
        # Downward U:
        for layer in self.down_seq.values():
            out = layer(inputs[-1])
            inputs.append(out)

        # Upward U:
        inputs.reverse()
        x1 = inputs.pop(0)
        for idx, layer in enumerate(self.up_seq.values()):
            x2 = inputs[idx]
            x1 = layer(x1, x2) 

        out = self.outc(x1) if self.enable_outc else x1

        return out


# sub-parts of the U-Net model
# DoubleConv with GhostNet
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, scheme="cnn"):
        super(DoubleConv, self).__init__()
        self.scheme = scheme

        if scheme == "ghost":
            self.conv = nn.Sequential(
                GhostModule(in_ch, out_ch, kernel_size=1, ratio=2, relu=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                GhostModule(out_ch, out_ch, kernel_size=1, ratio=2, relu=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )
        elif scheme == "cnn":
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )
        elif scheme == "rpc":
            self.proj = nn.Conv2d(in_ch, out_ch, 1, bias=False)
            self.bn_proj = nn.BatchNorm2d(out_ch)
            self.prc = PRCNPTNLayer(
                inch=out_ch,    
                outch=out_ch,
                G=8,
                CMP=2,
                kernel_size=3,
                padding=1
            )
            self.bn2 = nn.BatchNorm2d(out_ch)
            self.act2 = nn.ReLU(inplace=True)

    def forward(self, x):

        if self.scheme == "rpc":
            x = self.bn_proj(self.proj(x))   # project in_ch -> out_ch
            x = self.prc(x)
            x = self.act2(self.bn2(x))
            return x
        else:
            return self.conv(x)


class InConv(nn.Module):
    def __init__(self, in_ch, out_ch, conv_block):
        super(InConv, self).__init__()
        self.conv = conv_block(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class Down(nn.Module):
    def __init__(self, in_ch, out_ch, conv_block):
        super(Down, self).__init__()

        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            conv_block(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, conv_block, attn_scheme=None):
        super(Up, self).__init__()

        self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)
        self.conv = conv_block(in_ch, out_ch)

        self.attn_scheme = attn_scheme

        if attn_scheme == "SE":
            self.skip_attn = SEAttention(in_ch // 2)
        elif attn_scheme == "COORD":
            self.skip_attn = CoordAtt(in_ch // 2, in_ch // 2)
        elif attn_scheme == "CBAM":
            self.skip_attn = CBAMBlock(in_ch // 2)
        elif attn_scheme == "SHUFFLE":
            self.skip_attn = ShuffleAttention(in_ch // 2)
        elif attn_scheme == "CRICRO":
            self.skip_attn = CrissCrossAttention(in_ch // 2)
        elif attn_scheme == None:
            print("No attention sceheme is applied.")
            self.skip_attn = nn.Identity()

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # Handle padding for 2D images
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [
            diffX // 2, diffX - diffX // 2,
            diffY // 2, diffY - diffY // 2,
        ])

        if self.attn_scheme:
            x2 = self.skip_attn(x2)  #  attend skip features before concat
            
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x