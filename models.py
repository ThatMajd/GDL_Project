import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_add_pool, global_max_pool

AGG_METHODS = {
    "sum": global_add_pool,
    "avg": global_mean_pool,
    "max": global_max_pool,
}

class GAT(torch.nn.Module):
    def __init__(self, n_layer, agg_hidden, fc_hidden, in_channels=7, num_classes=2, agg_method="avg"):
        super(GAT, self).__init__()
        
        self.agg_method = AGG_METHODS[agg_method]

        # Define the first GAT layer
        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(in_channels, agg_hidden, heads=1, concat=True))
        
        # Add additional GAT layers
        for _ in range(n_layer - 1):
            self.convs.append(GATConv(agg_hidden, agg_hidden, heads=1, concat=True))
        
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