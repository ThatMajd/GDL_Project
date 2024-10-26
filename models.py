import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GATv2Conv, global_mean_pool, global_add_pool, global_max_pool, SAGEConv
from torch.nn import ModuleList, Linear, LayerNorm
from torch_geometric.nn.conv import TransformerConv 


AGG_METHODS = {
    "sum": global_add_pool,
    "avg": global_mean_pool,
    "max": global_max_pool,
}

class GAT(torch.nn.Module):
    def __init__(self, n_layer, agg_hidden, fc_hidden, in_channels=7, num_classes=2, agg_method="avg", dropout=0.5):
        super(GAT, self).__init__()

        heads = 1
        
        self.agg_method = AGG_METHODS[agg_method]
        self.dropout = dropout

        # Define the first GAT layer
        self.convs = torch.nn.ModuleList()
        self.convs.append(GATv2Conv(in_channels, agg_hidden, heads=heads, concat=True, dropout=self.dropout, edge_dim=4))
        
        # Add additional GAT layers
        for _ in range(n_layer - 1):
            self.convs.append(GATv2Conv(agg_hidden, agg_hidden, heads=heads, concat=True, dropout=self.dropout, edge_dim=4))
        
        # Fully connected layers after GAT aggregation
        self.fc1 = torch.nn.Linear(agg_hidden, fc_hidden)
        self.fc2 = torch.nn.Linear(fc_hidden, num_classes)


    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # x = F.dropout(x, p=self.dropout, training=self.training)

        # Apply GAT layers
        for conv in self.convs:
            x = F.elu(conv(x, edge_index))
        
        # Pooling (mean pooling across nodes)
        x = self.agg_method(x, batch)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)
    

class GraphTransformer(torch.nn.Module):
    def __init__(self, n_layer, agg_hidden, fc_hidden, in_channels=7, 
                 num_classes=2, agg_method="avg", num_heads=4):
        super(GraphTransformer, self).__init__()
        
        self.agg_method = AGG_METHODS[agg_method]  # Pooling function
        self.convs = ModuleList()  # List to hold Transformer layers
        self.norms = ModuleList()  # LayerNorms for stability

        # Define the first TransformerConv layer
        self.convs.append(TransformerConv(in_channels, agg_hidden, heads=num_heads, dropout=0.5))
        self.norms.append(LayerNorm(agg_hidden * num_heads))

        # Add additional TransformerConv layers
        for _ in range(n_layer - 1):
            self.convs.append(TransformerConv(agg_hidden * num_heads, agg_hidden, heads=num_heads, dropout=0.5))
            self.norms.append(LayerNorm(agg_hidden * num_heads))

        # Fully connected layers for classification
        self.fc1 = Linear(agg_hidden * num_heads, fc_hidden)
        self.fc2 = Linear(fc_hidden, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Apply TransformerConv layers with LayerNorm
        for conv, norm in zip(self.convs, self.norms):
            x = F.elu(conv(x, edge_index))
            x = norm(x)  # Normalize layer outputs

        # Apply pooling across nodes in the graph
        x = self.agg_method(x, batch)

        # Fully connected layers for graph-level classification
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)

class GraphSAGE(torch.nn.Module):
    def __init__(self, n_layer, agg_hidden, fc_hidden, in_channels=7, num_classes=2, agg_method="avg", dropout=0.5):
        super(GraphSAGE, self).__init__()
        
        self.agg_method = AGG_METHODS[agg_method]
        
        # Define the GraphSAGE layers
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, agg_hidden))
        
        for _ in range(n_layer - 1):
            self.convs.append(SAGEConv(agg_hidden, agg_hidden))
        
        # Fully connected layers after GraphSAGE aggregation
        self.fc1 = torch.nn.Linear(agg_hidden, fc_hidden)
        self.fc2 = torch.nn.Linear(fc_hidden, num_classes)
        
        self.dropout = dropout

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Apply GraphSAGE layers
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Pooling (sum, mean, or max pooling across nodes)
        x = self.agg_method(x, batch)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)

from torch.distributions import Normal

AGG_METHODS = {
    "sum": global_add_pool,
    "avg": global_mean_pool,
    "max": global_max_pool,
}

class GaussianGraphSAGE(torch.nn.Module):
    def __init__(self, n_layer, agg_hidden, fc_hidden, in_channels=7, num_classes=2, agg_method="avg", dropout=0.5, kl_weight=1e-4):
        super(GaussianGraphSAGE, self).__init__()
        
        # Set aggregation method (sum, mean, or max pooling)
        self.agg_method = AGG_METHODS[agg_method]
        self.dropout = dropout
        self.kl_weight = kl_weight
        
        # Define Gaussian embedding layers (mean and log-variance)
        self.mean_conv = torch.nn.ModuleList()
        self.var_conv = torch.nn.ModuleList()
        self.mean_conv.append(SAGEConv(in_channels, agg_hidden))
        self.var_conv.append(SAGEConv(in_channels, agg_hidden))
        
        # Additional SAGE layers
        for _ in range(n_layer - 1):
            self.mean_conv.append(SAGEConv(agg_hidden, agg_hidden))
            self.var_conv.append(SAGEConv(agg_hidden, agg_hidden))
        
        # Fully connected layers after SAGE aggregation
        self.fc1 = torch.nn.Linear(agg_hidden, fc_hidden)
        self.fc2 = torch.nn.Linear(fc_hidden, num_classes)

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)  # Standard deviation
        eps = torch.randn_like(std)     # Random noise
        return mean + eps * std         # Sampled embedding using the reparameterization trick

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Apply Gaussian embedding layers
        mean = F.relu(self.mean_conv[0](x, edge_index))
        log_var = F.relu(self.var_conv[0](x, edge_index))
        
        for i in range(1, len(self.mean_conv)):
            mean = F.relu(self.mean_conv[i](mean, edge_index))
            mean = F.dropout(mean, p=self.dropout, training=self.training)
            log_var = F.relu(self.var_conv[i](log_var, edge_index))
            log_var = F.dropout(log_var, p=self.dropout, training=self.training)
        
        z = self.reparameterize(mean, log_var)  # Sampled embedding

        # Pooling (sum, mean, or max pooling across nodes)
        z = self.agg_method(z, batch)
        
        # Fully connected layers
        z = F.relu(self.fc1(z))
        z = F.dropout(z, p=self.dropout, training=self.training)
        z = self.fc2(z)

        return F.log_softmax(z, dim=1), mean, log_var

    def kl_divergence(self, mean, log_var):
        # Compute KL divergence for each node embedding
        kl_div = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        return kl_div

    def loss(self, pred, target, mean, log_var, weight=None):
        # weight = None
        # Standard classification loss
        ce_loss = F.cross_entropy(pred, target, weight=weight)
        
        # KL divergence regularization
        kl_div = self.kl_divergence(mean, log_var)
        
        # Total loss with KL divergence weighted
        return ce_loss + self.kl_weight * kl_div

class GaussianGAT(torch.nn.Module):
    def __init__(self, n_layer, agg_hidden, fc_hidden, in_channels=7, num_classes=2, agg_method="avg", dropout=0.5, kl_weight=1e-4):
        super(GaussianGAT, self).__init__()
        
        # Set aggregation method (sum, mean, or max pooling)
        self.agg_method = AGG_METHODS[agg_method]
        self.dropout = dropout
        self.kl_weight = kl_weight
        
        # Define Gaussian embedding layers (mean and log-variance)
        self.mean_conv = torch.nn.ModuleList()
        self.var_conv = torch.nn.ModuleList()
        
        # Initial GAT layers for mean and variance
        self.mean_conv.append(GATv2Conv(in_channels, agg_hidden, heads=1, concat=True, dropout=dropout, edge_dim=4))
        self.var_conv.append(GATv2Conv(in_channels, agg_hidden, heads=1, concat=True, dropout=dropout, edge_dim=4))
        
        # Additional GAT layers
        for _ in range(n_layer - 1):
            self.mean_conv.append(GATv2Conv(agg_hidden, agg_hidden, heads=1, concat=True, dropout=dropout, edge_dim=4))
            self.var_conv.append(GATv2Conv(agg_hidden, agg_hidden, heads=1, concat=True, dropout=dropout, edge_dim=4))
        
        # Fully connected layers after GAT aggregation
        self.fc1 = torch.nn.Linear(agg_hidden, fc_hidden)
        self.fc2 = torch.nn.Linear(fc_hidden, num_classes)

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)  # Standard deviation
        eps = torch.randn_like(std)     # Random noise
        return mean + eps * std         # Sampled embedding using the reparameterization trick

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Apply Gaussian embedding layers
        mean = F.elu(self.mean_conv[0](x, edge_index))
        log_var = F.elu(self.var_conv[0](x, edge_index))
        
        for i in range(1, len(self.mean_conv)):
            mean = F.elu(self.mean_conv[i](mean, edge_index))
            mean = F.dropout(mean, p=self.dropout, training=self.training)
            log_var = F.elu(self.var_conv[i](log_var, edge_index))
            log_var = F.dropout(log_var, p=self.dropout, training=self.training)
        
        # Sample embeddings using the reparameterization trick
        z = self.reparameterize(mean, log_var)

        # Pooling (sum, mean, or max pooling across nodes)
        z = self.agg_method(z, batch)
        
        # Fully connected layers
        z = F.relu(self.fc1(z))
        z = F.dropout(z, p=self.dropout, training=self.training)
        z = self.fc2(z)

        return F.log_softmax(z, dim=1), mean, log_var

    def kl_divergence(self, mean, log_var):
        # Compute KL divergence for each node embedding
        kl_div = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        return kl_div

    def loss(self, pred, target, mean, log_var):
        # Standard classification loss
        ce_loss = F.nll_loss(pred, target)
        
        # KL divergence regularization
        kl_div = self.kl_divergence(mean, log_var)
        
        # Total loss with KL divergence weighted
        return ce_loss + self.kl_weight * kl_div