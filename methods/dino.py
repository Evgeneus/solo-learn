import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from base import Model
except:
    from .base import Model

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from losses.dino import DINOLoss
from utils.metrics import accuracy_at_k
from utils.momentum import initialize_momentum_params, MomentumUpdater
from utils.trunc_normal import trunc_normal_

class DINOHead(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        use_bn=True,
        norm_last_layer=True,
        nlayers=3,
        hidden_dim=2048,
        bottleneck_dim=256
    ):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(
            nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x


class DINO(Model):
    def __init__(self, args):
        super().__init__(args)

        self.clip_grad = args.clip_grad
        self.freeze_last_layer = args.freeze_last_layer

        # dino head
        self.head = DINOHead(
            in_dim=self.features_size,
            hidden_dim=args.hidden_dim,
            bottleneck_dim=args.encoding_dim,
            out_dim=args.num_prototypes,
            norm_last_layer=args.norm_last_layer
        )

        # instantiate and initialize momentum encoder
        self.momentum_encoder = self.base_model(zero_init_residual=args.zero_init_residual)
        self.momentum_encoder.fc = nn.Identity()
        if args.cifar:
            self.momentum_encoder.conv1 = nn.Conv2d(
                3, 64, kernel_size=3, stride=1, padding=2, bias=False
            )
            self.momentum_encoder.maxpool = nn.Identity()
        initialize_momentum_params(self.encoder, self.momentum_encoder)

        # instantiate and initialize momentum dino head
        self.momentum_head = DINOHead(
            in_dim=self.features_size,
            hidden_dim=args.hidden_dim,
            bottleneck_dim=args.encoding_dim,
            out_dim=args.num_prototypes,
            norm_last_layer=args.norm_last_layer
        )
        initialize_momentum_params(self.head, self.momentum_head)

        # momentum updater
        self.momentum_updater = MomentumUpdater(args.base_tau_momentum, args.final_tau_momentum)

        # dino loss
        self.dino_loss_func = DINOLoss(
            out_dim=args.num_prototypes,
            student_temp=args.student_temperature,
            warmup_teacher_temp=args.warmup_teacher_temperature,
            teacher_temp=args.teacher_temperature,
            warmup_teacher_temp_epochs=args.warmup_teacher_temperature_epochs,
            nepochs=args.epochs
        )

    def clip_gradients(self, clip):
        for name, p in self.encoder.named_parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                clip_coef = clip / (param_norm + 1e-6)
                if clip_coef < 1:
                    p.grad.data.mul_(clip_coef)

    def on_train_epoch_start(self, ):
        self.dino_loss_func.epoch = self.current_epoch

    def forward(self, X, classify_only=True):
        features, y = super().forward(X, classify_only=False)
        if classify_only:
            return y
        else:
            p = self.head(features)
            return p, y

    @torch.no_grad()
    def forward_momentum(self, X):
        features_momentum = self.momentum_encoder(X)
        p_momentum = self.momentum_head(features_momentum)
        return p_momentum

    def training_step(self, batch, _):
        indexes, (X1, X2), target = batch

        # forward online encoder
        p1, output1 = self(X1, classify_only=False)
        p2, output2 = self(X2, classify_only=False)
        p = torch.cat((p1, p2))

        # forward momentum encoder
        p1_momentum = self.forward_momentum(X1)
        p2_momentum = self.forward_momentum(X2)
        p_momentum = torch.cat((p1_momentum, p2_momentum))

        # ------- contrastive loss -------
        dino_loss = self.dino_loss_func(p, p_momentum)

        # ------- classification loss -------
        # for datasets with unsupervised data
        output = torch.cat((output1, output2))
        target = target.repeat(2)

        # ------- classification loss -------
        class_loss = F.cross_entropy(output, target, ignore_index=-1)

        # just add together the losses to do only one backward()
        # we have stop gradients on the output y of the model
        loss = dino_loss + class_loss

        # ------- metrics -------
        acc1, acc5 = accuracy_at_k(output, target, top_k=(1, 5))

        metrics = {
            "train_ce_loss": dino_loss,
            "train_class_loss": class_loss,
            "train_acc1": acc1,
            "train_acc5": acc5,
        }
        self.log_dict(metrics, on_epoch=True, sync_dist=True)
        return loss

    def on_after_backward(self):
        # clip gradients
        if self.clip_grad:
            self.clip_gradients(self.clip_grad)
        # zero gradients on last layer
        if self.current_epoch < self.freeze_last_layer:
            for p in self.head.last_layer.parameters():
                p.grad = None

    def on_train_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
        # log tau momentum
        self.log("tau", self.momentum_updater.cur_tau)
        # update momentum encoder
        self.momentum_updater.update(
            online_nets=[self.encoder, self.head],
            momentum_nets=[self.momentum_encoder, self.momentum_head],
            cur_step=self.trainer.global_step,
            max_steps=len(self.trainer.train_dataloader) * self.trainer.max_epochs,
        )