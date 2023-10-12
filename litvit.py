from torch.nn.functional import binary_cross_entropy, cross_entropy
import torch.optim as optim
# for parallelization
import lightning.pytorch as li

# Define the LightningModule
class LitViT(li.LightningModule):
    def __init__(self, vit):
        super().__init__()
        self.vit = vit
        
    def forward(self, x):
        return self.vit(x)
    
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        inputs, labels = batch
        outputs = self.vit(inputs)
        loss_bowel = binary_cross_entropy(outputs[:,0:1], labels[:,0:1])
        self.log("train/loss_bowel", loss_bowel)
        loss_extra = binary_cross_entropy(outputs[:,1:2], labels[:,1:2])
        self.log("train/loss_extra", loss_extra)
        loss_kidney = cross_entropy(outputs[:,2:5], labels[:,2:5])
        self.log("train/loss_kidney", loss_kidney)
        loss_liver = cross_entropy(outputs[:,5:8], labels[:,5:8])
        self.log("train/loss_liver", loss_liver)
        loss_spleen = cross_entropy(outputs[:,8:11], labels[:,8:11])
        self.log("train/loss_spleen", loss_spleen)
        loss = loss_bowel + loss_extra + loss_kidney + loss_liver + loss_spleen
        self.log("train/loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self.vit(inputs)
        loss_bowel = binary_cross_entropy(outputs[:,0:1], labels[:,0:1])
        self.log("val/loss_bowel", loss_bowel)
        loss_extra = binary_cross_entropy(outputs[:,1:2], labels[:,1:2])
        self.log("val/loss_extra", loss_extra)
        loss_kidney = cross_entropy(outputs[:,2:5], labels[:,2:5])
        self.log("val/loss_kidney", loss_kidney)
        loss_liver = cross_entropy(outputs[:,5:8], labels[:,5:8])
        self.log("val/loss_liver", loss_liver)
        loss_spleen = cross_entropy(outputs[:,8:11], labels[:,8:11])
        self.log("val/loss_spleen", loss_spleen)
        loss = loss_bowel + loss_extra + loss_kidney + loss_liver + loss_spleen
        self.log("val/loss", loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
