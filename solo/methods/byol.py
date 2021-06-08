import torch
import torch.nn as nn
import torch.nn.functional as F
from solo.losses.byol import byol_loss_func
from solo.methods.base import BaseModel
from solo.utils.metrics import accuracy_at_k
from solo.utils.momentum import MomentumUpdater, initialize_momentum_params


class BYOL(BaseModel):
    def __init__(
        self,
        output_dim,
        proj_hidden_dim,
        pred_hidden_dim,
        base_tau_momentum,
        final_tau_momentum,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.last_step = 0

        # projector 1
        self.projector1 = nn.Sequential(
            nn.Linear(self.features_size, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, output_dim),
        )

        # projector 2
        self.projector2 = nn.Sequential(
            nn.Linear(self.features_size, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, output_dim),
        )

        # predictor 1
        self.predictor1 = nn.Sequential(
            nn.Linear(output_dim, pred_hidden_dim),
            nn.BatchNorm1d(pred_hidden_dim),
            nn.ReLU(),
            nn.Linear(pred_hidden_dim, output_dim),
        )

        # predictor 2
        self.predictor2 = nn.Sequential(
            nn.Linear(output_dim, pred_hidden_dim),
            nn.BatchNorm1d(pred_hidden_dim),
            nn.ReLU(),
            nn.Linear(pred_hidden_dim, output_dim),
        )

        # instantiate and initialize momentum encoder
        self.momentum_encoder = self.base_model(zero_init_residual=self.zero_init_residual)
        self.momentum_encoder.fc = nn.Identity()
        if self.cifar:
            self.momentum_encoder.conv1 = nn.Conv2d(
                3, 64, kernel_size=3, stride=1, padding=2, bias=False
            )
            self.momentum_encoder.maxpool = nn.Identity()
        initialize_momentum_params(self.encoder, self.momentum_encoder)

        # instantiate and initialize momentum projector1
        self.momentum_projector1 = nn.Sequential(
            nn.Linear(self.features_size, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, output_dim),
        )

        # instantiate and initialize momentum projector2
        self.momentum_projector2 = nn.Sequential(
            nn.Linear(self.features_size, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, output_dim),
        )
        initialize_momentum_params(self.projector1, self.momentum_projector1)
        initialize_momentum_params(self.projector2, self.momentum_projector2)

        # momentum updater
        self.momentum_updater = MomentumUpdater(base_tau_momentum, final_tau_momentum)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("byol")
        # projector
        parser.add_argument("--output_dim", type=int, default=256)
        parser.add_argument("--proj_hidden_dim", type=int, default=2048)

        # predictor
        parser.add_argument("--pred_hidden_dim", type=int, default=512)

        # momentum settings
        parser.add_argument("--base_tau_momentum", default=0.99, type=float)
        parser.add_argument("--final_tau_momentum", default=1.0, type=float)
        return parent_parser

    @property
    def extra_learnable_params(self):
        return [{"params": self.projector1.parameters()}, {"params": self.predictor1.parameters()},
                {"params": self.projector2.parameters()}, {"params": self.predictor2.parameters()}]

    def on_after_backward(self, *args):
        for pred in [self.predictor1, self.predictor2,
                     self.projector1, self.projector2]:
            for params in pred.parameters():
                params.grad *= 2.

    def forward(self, X, view_id=-1):
        out = super().forward(X)
        if view_id == 1:
            z = self.projector1(out["feat"])
            p = self.predictor1(z)
        elif view_id == 2:
            z = self.projector2(out["feat"])
            p = self.predictor2(z)
        elif view_id == -1:
            return out
        else:
            raise ValueError('Wrong view_id value')
        return {**out, "z": z, "p": p}

    @torch.no_grad()
    def forward_momentum(self, X, view_id):
        features_momentum = self.momentum_encoder(X)
        if view_id == 1:
            z_momentum = self.momentum_projector1(features_momentum)
        elif view_id == 2:
            z_momentum = self.momentum_projector2(features_momentum)
        else:
            raise ValueError('Wrong view_id value')

        return z_momentum

    def training_step(self, batch, batch_idx):
        indexes, (X1, X2), target = batch

        # forward online encoder
        out1 = self(X1, 1)
        out2 = self(X2, 2)

        z1 = out1["z"]
        z2 = out2["z"]
        p1 = out1["p"]
        p2 = out2["p"]
        logits1 = out1["logits"]
        logits2 = out2["logits"]

        # forward momentum encoder
        z1_momentum = self.forward_momentum(X1, 1)
        z2_momentum = self.forward_momentum(X2, 2)

        # ------- contrastive loss -------
        neg_cos_sim = byol_loss_func(p1, z2_momentum) / 2 + byol_loss_func(p2, z1_momentum) / 2

        # ------- classification loss -------
        logits = torch.cat((logits1, logits2))
        target = target.repeat(2)
        class_loss = F.cross_entropy(logits, target, ignore_index=-1)

        # just add together the losses to do only one backward()
        # we have stop gradients on the output y of the model
        loss = neg_cos_sim + class_loss

        # ------- metrics -------
        acc1, acc5 = accuracy_at_k(logits, target, top_k=(1, 5))

        # z_std = F.normalize(torch.cat((z1, z2), dim=0), dim=1).std(dim=0).mean()
        z1_std = F.normalize(z1, dim=1).std(dim=0).mean()
        z2_std = F.normalize(z2, dim=1).std(dim=0).mean()
        z_std = (z1_std + z2_std) / 2

        metrics = {
            "train_neg_cos_sim": neg_cos_sim,
            "train_class_loss": class_loss,
            "train_z_std": z_std,
            "train_acc1": acc1,
            "train_acc5": acc5,
        }
        self.log_dict(metrics, on_epoch=True, sync_dist=True)
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
        if self.trainer.global_step > self.last_step:
            # log tau momentum
            self.log("tau", self.momentum_updater.cur_tau)
            # update momentum encoder
            self.momentum_updater.update(
                online_nets=[self.encoder, self.projector1, self.projector2],
                momentum_nets=[self.momentum_encoder, self.momentum_projector1, self.momentum_projector2],
                cur_step=self.trainer.global_step * self.trainer.accumulate_grad_batches,
                max_steps=len(self.trainer.train_dataloader) * self.trainer.max_epochs,
            )
        self.last_step = self.trainer.global_step
