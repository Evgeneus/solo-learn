import torch
import torch.nn as nn
import torch.nn.functional as F
from solo.losses.vicreg import vicreg_loss_func
from solo.methods.base import BaseModel
from solo.utils.metrics import accuracy_at_k


class VICReg(BaseModel):
    def __init__(
        self,
        output_dim,
        proj_hidden_dim,
        sim_loss_weight,
        var_loss_weight,
        cov_loss_weight,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.sim_loss_weight = sim_loss_weight
        self.var_loss_weight = var_loss_weight
        self.cov_loss_weight = cov_loss_weight

        # projector
        self.projector = nn.Sequential(
            nn.Linear(self.features_size, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, output_dim),
        )

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("vicreg")
        # projector
        parser.add_argument("--output_dim", type=int, default=2048)
        parser.add_argument("--proj_hidden_dim", type=int, default=2048)

        # parameters
        parser.add_argument("--sim_loss_weight", default=25, type=float)
        parser.add_argument("--var_loss_weight", default=25, type=float)
        parser.add_argument("--cov_loss_weight", default=1.0, type=float)
        return parent_parser

    @property
    def extra_learnable_params(self):
        return [{"params": self.projector.parameters()}]

    def forward(self, X):
        out = super().forward(X)
        z = self.projector(out["feat"])
        return {**out, "z": z}

    def training_step(self, batch, batch_idx):
        indexes, (X1, X2), target = batch

        out1 = self(X1)
        out2 = self(X2)

        z1 = out1["z"]
        z2 = out2["z"]
        logits1 = out1["logits"]
        logits2 = out2["logits"]

        # ------- loss -------
        vicreg_loss = vicreg_loss_func(
            z1,
            z2,
            sim_loss_weight=self.sim_loss_weight,
            var_loss_weight=self.var_loss_weight,
            cov_loss_weight=self.cov_loss_weight,
        )

        # ------- classification loss -------
        logits = torch.cat((logits1, logits2))
        target = target.repeat(2)
        class_loss = F.cross_entropy(logits, target, ignore_index=-1)

        # just add together the losses to do only one backward()
        # we have stop gradients on the output y of the model
        loss = vicreg_loss + class_loss

        # ------- metrics -------
        acc1, acc5 = accuracy_at_k(logits, target, top_k=(1, 5))

        metrics = {
            "train_vicreg_loss": vicreg_loss,
            "train_class_loss": class_loss,
            "train_acc1": acc1,
            "train_acc5": acc5,
        }
        self.log_dict(metrics, on_epoch=True, sync_dist=True)
        return loss
