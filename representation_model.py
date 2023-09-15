import torch
import torch.nn.functional as F
import torch.nn as nn

from NewResnets import resnet18


class MLP(nn.Module):

    def __init__(self, in_features, hidden_features, out_features, p=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.GELU(),
            nn.Linear(hidden_features, out_features),
            nn.Dropout(p),
        )


    def forward(self, x):
        x = self.mlp(x)
        return x

class Multi_Heads_Self_Attention(nn.Module):
    def __init__(self, dim, n_heads=16, qkv_bias=True, attn_p=0.1, proj_p=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_p)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_p)

    def forward(self, x):
        n_samples, n_tokens, dim = x.shape

        if dim != self.dim:
            raise ValueError

        qkv = self.qkv(x)  # (n_samples, n_patches, 3 * dim)
        qkv = qkv.reshape(
                n_samples, n_tokens, 3, self.n_heads, self.head_dim
        )  # (n_smaples, n_patches, 3, n_heads, head_dim)
        qkv = qkv.permute(
                2, 0, 3, 1, 4
        )  # (3, n_samples, n_heads, n_patches , head_dim)

        q, k, v = qkv[0], qkv[1], qkv[2]
        k_t = k.transpose(-2, -1)  # (n_samples, n_heads, head_dim, n_patches)
        dp = (
           q @ k_t
        ) * self.scale # (n_samples, n_heads, n_patches, n_patches )
        attn = dp.softmax(dim=-1)  # (n_samples, n_heads, n_patches, n_patches)
        attn = self.attn_drop(attn)

        weighted_avg = attn @ v  # (n_samples, n_heads, n_patches, head_dim)
        weighted_avg = weighted_avg.transpose(
                1, 2
        )  # (n_samples, n_patches, n_heads, head_dim)
        weighted_avg = weighted_avg.flatten(2)  # (n_samples, n_patches, dim)

        x = self.proj(weighted_avg)  # (n_samples, n_patches, dim)
        x = self.proj_drop(x)  # (n_samples, n_patches, dim)

        return x
class Block(nn.Module):
    def __init__(self, dim, n_heads, mlp_ratio, qkv_bias, p, attn_p):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Multi_Heads_Self_Attention(
                dim,
                n_heads=n_heads,
                qkv_bias=qkv_bias,
                attn_p=attn_p,
                proj_p=p
        )
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        hidden_features = int(dim * mlp_ratio)
        self.mlp = MLP(
                in_features=dim,
                hidden_features=hidden_features,
                out_features=dim,
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))

        return x
class TransformerEncoder(nn.Module):
    def __init__(
            self,
            patches=49,
            embed_dim=512,
            depth=1,
            n_heads=8,
            mlp_ratio=1,
            qkv_bias=True,
            p=0.1,
            attn_p=0.1,
    ):
        super().__init__()
        self.pos_embed = nn.Parameter(
                torch.zeros(1, patches, embed_dim)
        )
        self.pos_drop = nn.Dropout(p=p)

        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    n_heads=n_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    p=p,
                    attn_p=attn_p,
                )
                for _ in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

    def forward(self, x):

        x = x + self.pos_embed  # (bs, n_patches, embed_dim)
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)


        return x


class LabelEncoder(nn.Module):
    def __init__(self, num_classes):
        super(LabelEncoder, self).__init__()
        self.Encoder = TransformerEncoder(patches=num_classes,embed_dim=512)
    def forward(self,x):
        x = x.unsqueeze(dim=0)
        x = self.Encoder(x)
        x = F.normalize(x)
        x = x.squeeze(dim=0)
        return x

class RepresentationModel(nn.Module):
    def __init__(self, fold, pre_path, TrainIDs, au_keys, dataset):
        super(RepresentationModel, self).__init__()

        self.device = torch.device("cuda:0")
        self.fold = fold
        self.pre_path = pre_path
        self.TrainIDs = TrainIDs
        self.dataset = dataset

        # Initialize Modules
        self.au_keys = au_keys
        self.backbone = self._init_backbone()
        self.FeatureEncoder = self._init_feature_encoder()
        self.LabelEncoder = LabelEncoder(len(self.au_keys)).to(self.device)
        self.IDnet = self._init_IDnet()

        # Initial Embeddings
        self.init_of_embed = self._initialize_embeddings()

    def _init_backbone(self):
        model = resnet18(backbone=False, num_classes=len(self.au_keys)).to(self.device)
        if self.dataset == "BP4D":
            pretrained = torch.load(self.pre_path + "Backbone_R18_{}.pth".format(self.fold))
        elif self.dataset == "DISFA":
            pretrained = torch.load(self.pre_path + "Backbone_D_{}.pth".format(self.fold))
        else:
            pretrained = torch.load(self.pre_path + "R18_mixed.pth")
        model_dict = model.state_dict()
        pretrained = {k: v for k, v in pretrained["model_state_dict"].items()
                      if k in model_dict and "fc" not in k}
        model_dict.update(pretrained)
        model.load_state_dict(model_dict)
        # Freezing the backbone parameters
        for param in model.parameters():
            param.requires_grad = False
        return model

    def _init_feature_encoder(self):
        return TransformerEncoder(
            patches=49, embed_dim=512, depth=4, n_heads=32,
            mlp_ratio=2, qkv_bias=False, p=0.1, attn_p=0.1
        ).to(self.device)

    def _init_IDnet(self):
        model = IDNet(self.fold, self.pre_path, self.TrainIDs, self.au_keys, self.dataset).to(self.device)
        if self.dataset == "BP4D":
            pretrained = torch.load(self.pre_path + "IDnet_{}_300.pth".format(self.fold))
        elif self.dataset == "DISFA":
            pretrained = torch.load(self.pre_path + "IDnet_{}_400.pth".format(self.fold))
        else:
            pretrained = torch.load(self.pre_path + "ID_mixed.pth")
        model_dict = model.state_dict()
        pretrained = {k: v for k, v in pretrained["model_state_dict"].items()
                      if k in model_dict and "fc" not in k}
        model_dict.update(pretrained)
        model.load_state_dict(model_dict)
        # Freezing the backbone parameters
        for param in model.parameters():
            param.requires_grad = False
        return model

    def _initialize_embeddings(self):
        AU_mean = torch.rand(len(self.au_keys)).to(self.device)
        AU_var = torch.rand(len(self.au_keys)).to(self.device)
        return torch.stack(
            [torch.normal(mean=mean, std=var, size=(512,)) for mean, var in zip(AU_mean, AU_var)]
        ).to(self.device)

    def forward(self, x, IDs, val_mode=None):
        x_processed = self.backbone(x)
        x_processed = x_processed.reshape(x_processed.shape[0], x_processed.shape[2] * x_processed.shape[3],
                                          x_processed.shape[1])
        feature_encode = self.FeatureEncoder(x_processed).mean(dim=1)

        if val_mode:
            return self._validation_mode_forward(feature_encode)
        else:
            return self._training_mode_forward(feature_encode, x, IDs)

    def _validation_mode_forward(self, feature_encode):
        list_of_embed = self.LabelEncoder(self.init_of_embed)
        predictions = torch.stack([torch.mv(feature_encode, embed) for embed in list_of_embed])
        return feature_encode, list_of_embed, predictions.t()

    def _training_mode_forward(self, feature_encode, x, IDs):
        _, _, ID_feature_encode = self.IDnet(x, IDs=IDs)
        self.label_embeds = self.LabelEncoder(self.init_of_embed)
        predictions = torch.stack([torch.mv(feature_encode, embed) for embed in self.label_embeds])
        return feature_encode, ID_feature_encode, self.label_embeds, predictions.t()


class IDNet(nn.Module):
    def __init__(self, fold, pre_path, TrainIDs, au_keys, dataset):
        super(IDNet, self).__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.fold = fold
        self.pre_path = pre_path
        self.au_keys = au_keys
        self.embedding_dims = 512
        self.TrainIDs = TrainIDs
        self.dataset = dataset

        # Defining network modules
        self.backbone = resnet18(backbone=False, num_classes=len(au_keys)).to(self.device)
        self.FeatureEncoder = TransformerEncoder().to(self.device)

        # Initialize embeddings for each TrainID with random data
        self.IDlabel_embeds = torch.nn.Parameter(
            torch.zeros(len(self.TrainIDs), len(self.au_keys), self.embedding_dims)).to(self.device)
        self._init_embeddings()

        # Load pretrained model weights for the backbone
        self._load_pretrained_weights()

    def _init_embeddings(self):
        ID_mean = torch.rand(len(self.TrainIDs))
        ID_var = torch.rand(len(self.TrainIDs))
        for i in range(len(self.TrainIDs)):
            ID_emb = torch.empty(len(self.au_keys), self.embedding_dims).normal_(mean=ID_mean[i], std=ID_var[i]).to(
                self.device)
            self.IDlabel_embeds[i, :, :] = ID_emb + self.IDlabel_embeds[i, :, :]

    def _load_pretrained_weights(self):
        if self.dataset == "BP4D":
            pretrained = torch.load(f"{self.pre_path}Backbone_R18_{self.fold}.pth")
        elif self.dataset == "DISFA":
            pretrained = torch.load(f"{self.pre_path}Backbone_D_{self.fold}.pth")
        else:
            pretrained = torch.load(f"{self.pre_path}R18_mixed.pth")
        model_dict = self.backbone.state_dict()
        pretrained = {k: v for k, v in pretrained["model_state_dict"].items()
                                          if k in model_dict and "fc" not in k}

        model_dict.update(pretrained)
        self.backbone.load_state_dict(model_dict)

        # Freezing the backbone parameters
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, x, IDs):
        # Feature extraction
        x = self.backbone(x)
        x = x.reshape(x.shape[0], x.shape[2] * x.shape[3], x.shape[1])
        feature_encode = self.FeatureEncoder(x)
        feature_encode = feature_encode.mean(dim=1)

        # Compute predictions for each feature encoding
        pred_inBatch = torch.zeros((len(feature_encode), len(self.au_keys))).to(self.device)
        for j in range(len(feature_encode)):
            ID = IDs[j]
            index = self.TrainIDs.index(ID)
            pred = torch.stack(
                [torch.dot(feature_encode[j], self.IDlabel_embeds[index, i, :]) for i in range(len(self.au_keys))]).to(
                self.device)
            pred_inBatch[j, :] = pred_inBatch[j, :] + pred

        return self.IDlabel_embeds, pred_inBatch, feature_encode


