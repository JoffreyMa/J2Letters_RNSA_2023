from torch.nn.functional import cross_entropy
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
        loss = cross_entropy(outputs, labels)
        self.log("train_loss", loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
