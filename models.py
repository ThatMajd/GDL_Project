import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GATv2Conv, global_mean_pool, global_add_pool, global_max_pool, SAGEConv
from torch.nn import ModuleList, Linear, LayerNorm
import torch.nn as nn
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
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # x = F.dropout(x, p=self.dropout, training=self.training)

        # Apply GAT layers
        for conv in self.convs:
            x = F.elu(conv(x, edge_index, edge_attr))
        
        # Pooling (mean pooling across nodes)
        x = self.agg_method(x, batch)

        # Fully connected layers
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
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # Apply Gaussian embedding layers
        mean = F.relu(self.mean_conv[0](x, edge_index, edge_attr))  # edge_attr not used directly in SAGEConv
        log_var = F.relu(self.var_conv[0](x, edge_index,))
        
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
        # Standard classification loss
        ce_loss = F.cross_entropy(pred, target, weight=weight)
        
        # KL divergence regularization
        kl_div = self.kl_divergence(mean, log_var)
        
        # Total loss with KL divergence weighted
        return ce_loss + self.kl_weight * kl_div

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

class GaussianGAT(torch.nn.Module):
    def __init__(self, n_layer, agg_hidden, fc_hidden, in_channels=7, num_classes=2, heads=1, dropout=0.5, kl_weight=1e-4):
        super(GaussianGAT, self).__init__()

        self.n_layer = n_layer
        self.heads = heads
        self.dropout = dropout
        self.kl_weight = kl_weight

        # Define Gaussian embedding layers (mean and log-variance) with attention
        self.mean_conv = torch.nn.ModuleList()
        self.var_conv = torch.nn.ModuleList()
        self.mean_bn = torch.nn.ModuleList()
        self.var_bn = torch.nn.ModuleList()

        # First GATv2 layer
        self.mean_conv.append(GATv2Conv(in_channels, agg_hidden, heads=heads, dropout=dropout, edge_dim=4, concat=False))
        self.var_conv.append(GATv2Conv(in_channels, agg_hidden, heads=heads, dropout=dropout, edge_dim=4, concat=False))
        self.mean_bn.append(torch.nn.BatchNorm1d(agg_hidden))
        self.var_bn.append(torch.nn.BatchNorm1d(agg_hidden))

        # Additional GATv2 layers
        for _ in range(n_layer - 1):
            self.mean_conv.append(GATv2Conv(agg_hidden, agg_hidden, heads=heads, dropout=dropout, edge_dim=4, concat=False))
            self.var_conv.append(GATv2Conv(agg_hidden, agg_hidden, heads=heads, dropout=dropout, edge_dim=4, concat=False))
            self.mean_bn.append(torch.nn.BatchNorm1d(agg_hidden))
            self.var_bn.append(torch.nn.BatchNorm1d(agg_hidden))

        # Fully connected layers after GATv2 aggregation
        self.fc1 = torch.nn.Linear(agg_hidden, fc_hidden)
        self.fc2 = torch.nn.Linear(fc_hidden, num_classes)

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)  # Standard deviation
        eps = torch.randn_like(std)     # Random noise
        return mean + eps * std         # Sampled embedding using the reparameterization trick

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # Apply Gaussian embedding layers with attention
        mean = F.relu(self.mean_bn[0](self.mean_conv[0](x, edge_index, edge_attr)))
        log_var = F.relu(self.var_bn[0](self.var_conv[0](x, edge_index, edge_attr)))
        
        for i in range(1, len(self.mean_conv)):
            mean = F.relu(self.mean_bn[i](self.mean_conv[i](mean, edge_index, edge_attr)))
            mean = F.dropout(mean, p=self.dropout, training=self.training)
            log_var = F.relu(self.var_bn[i](self.var_conv[i](log_var, edge_index, edge_attr)))
            log_var = F.dropout(log_var, p=self.dropout, training=self.training)
        
        # Sampled embedding
        z = self.reparameterize(mean, log_var)

        # Pooling (mean across nodes in the batch)
        z = global_mean_pool(z, batch)
        
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
        # Standard classification loss
        ce_loss = F.cross_entropy(pred, target, weight=weight)
        
        # KL divergence regularization
        kl_div = self.kl_divergence(mean, log_var)
        
        # Total loss with KL divergence weighted
        return ce_loss + self.kl_weight * kl_div