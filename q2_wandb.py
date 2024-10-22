import wandb
import random

wandb_username = 'majd_bishara'
wandb_project = 'GDP_Project'

command = ['${env}', '${interpreter}', 'q2_code.py', '${args}']

sweep_config={
    'method': 'random',
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
        'batch_size': {
            'values': [5, 10]
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
            'values': ["sum", "mean", "max"]
        },
        'wandb': {
            'value': 1
        }
    },
    'command': command
}

sweep_id = wandb.sweep(sweep_config, project=wandb_project)

print(f"wandb agent {wandb_username}/{wandb_project}/{sweep_id}")
