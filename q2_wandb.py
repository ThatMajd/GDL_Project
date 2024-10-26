import wandb
import random

wandb_username = 'majedbishara-technion-israel-institute-of-technology'
wandb_project = 'GDP_Project'

command = ['${env}', '${interpreter}', 'q2_code.py', '${args}']

sweep_config={
    'method': 'grid',
    'metric': {
        'name': 'Val/Avg_Accuracy',
        'goal': 'maximize'
    },
    'parameters': {
        'epochs': {
            'value': 100
        },
        'lr': {
            'values': [0.0005, 0.0007, 0.001, 0.005, 0.01]
        },
        'weight_decay': {
            'values': [0, 1e-3, 1e-4]
        },
        'batch_size': {
            'values': [1, 5]
        },
        'n_layer': {
            'values': list(range(1, 7))
        },
        'agg_hidden': {
            'values': [8, 16, 32, 64, 128, 256]
        },
        'fc_hidden': {
            'values': [8, 16, 32, 64, 128, 256]
        },
        'agg_method':{
            'values': ["sum", "max"] # Sum and mean perform the same
        },
        'wandb': {
            'value': 1
        }
    },
    'command': command
}

sweep_id = wandb.sweep(sweep_config, project=wandb_project)

print(f"wandb agent {wandb_username}/{wandb_project}/{sweep_id}")
