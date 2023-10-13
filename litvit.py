from torch.nn.functional import binary_cross_entropy, cross_entropy
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
# for parallelization
import lightning.pytorch as li
from vit3D import ViT

# Define the LightningModule
class LitViT(li.LightningModule):
    def __init__(self, *, image_size, image_patch_size, 
                 frames, frame_patch_size, 
                 dim, depth, heads, mlp_dim, neck_dim, 
                 pool = 'cls', channels = 3, 
                 dim_head = 64, 
                 dropout = 0., emb_dropout = 0., 
                 learning_rate=1e-2, t_max):
        super().__init__()
        self.vit = ViT(image_size=image_size, 
                       image_patch_size=image_patch_size, 
                       frames=frames, 
                       frame_patch_size=frame_patch_size, 
                       dim=dim, 
                       depth=depth, 
                       heads=heads, 
                       mlp_dim=mlp_dim, 
                       neck_dim=neck_dim, 
                       pool=pool, 
                       channels=channels, 
                       dim_head=dim_head, 
                       dropout=dropout, 
                       emb_dropout=emb_dropout)
        self.lr = learning_rate
        self.t_max = t_max
        self.save_hyperparameters()
        
    def forward(self, x):
        return self.vit(x)
    
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        return self._shared_eval("train")
    
    def validation_step(self, batch, batch_idx):
        return self._shared_eval("val")
    
    def _shared_eval(self, batch, batch_idx, prefix):
        inputs, labels = batch
        outputs = self.vit(inputs)
        loss_bowel = binary_cross_entropy(outputs[:,0:1], labels[:,0:1])
        self.log(f"{prefix}/loss_bowel", loss_bowel)
        loss_extra = binary_cross_entropy(outputs[:,1:2], labels[:,1:2])
        self.log(f"{prefix}/loss_extra", loss_extra)
        loss_kidney = cross_entropy(outputs[:,2:5], labels[:,2:5])
        self.log(f"{prefix}/loss_kidney", loss_kidney)
        loss_liver = cross_entropy(outputs[:,5:8], labels[:,5:8])
        self.log(f"{prefix}/loss_liver", loss_liver)
        loss_spleen = cross_entropy(outputs[:,8:11], labels[:,8:11])
        self.log(f"{prefix}/loss_spleen", loss_spleen)
        loss = loss_bowel + loss_extra + loss_kidney + loss_liver + loss_spleen
        self.log(f"{prefix}/loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        # Define custom cosine annealing scheduler
        lr_scheduler = {
            'scheduler': CosineAnnealingLR(optimizer, T_max=self.t_max, eta_min=0),  # T_max is just an example value
            'name': 'learning_rate',
            'interval': 'step',
            'frequency': 1
        }
        return [optimizer], [lr_scheduler]
