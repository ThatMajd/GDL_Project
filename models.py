import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GATv2Conv, global_mean_pool, global_add_pool, global_max_pool
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
        
        self.agg_method = AGG_METHODS[agg_method]

        # Define the first GAT layer
        self.convs = torch.nn.ModuleList()
        self.convs.append(GATv2Conv(in_channels, agg_hidden, heads=1, concat=True, dropout=dropout, edge_dim=4))
        
        # Add additional GAT layers
        for _ in range(n_layer - 1):
            self.convs.append(GATv2Conv(agg_hidden, agg_hidden, heads=1, concat=True, dropout=dropout, edge_dim=4))
        
        # Fully connected layers after GAT aggregation
        self.fc1 = torch.nn.Linear(agg_hidden, fc_hidden)
        self.fc2 = torch.nn.Linear(fc_hidden, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
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
        self.convs.append(TransformerConv(in_channels, agg_hidden, heads=num_heads))
        self.norms.append(LayerNorm(agg_hidden * num_heads))

        # Add additional TransformerConv layers
        for _ in range(n_layer - 1):
            self.convs.append(TransformerConv(agg_hidden * num_heads, agg_hidden, heads=num_heads))
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