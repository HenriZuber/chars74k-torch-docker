
import torch  # pylint: disable=import-error
import torch.nn as nn  # pylint: disable=import-error
import torch.optim as optim  # pylint: disable=import-error
import numpy as np

import load_sets
import models

USE_GPU = True
dtype = torch.float32
if USE_GPU and torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    print("using cpu")                                                       



def get_config():    
    lr1 = 1e-4
    lr2 = 7e-4
    channel_pics = 1
        
    model = models.advanced_model(channel_pics, 62)
    model = model.to(device)
    model.apply(models.weights_init_conv)
    
    optimizer1 = optim.SGD(
        model.parameters(), lr=lr1, weight_decay=1e-4, momentum=0.9, nesterov=True
    )
    optimizer2 = optim.SGD(
        model.parameters(), lr=lr2, weight_decay=1e-4, momentum=0.9, nesterov=True
    )
    optimizer3 = optim.Adam(model.parameters(), lr=lr1)
    optimizer4 = optim.Adam(model.parameters(), lr=lr2)
    optimizer5 = optim.RMSprop(model.parameters(), lr=1e-4)
                                                                                        
    sets = load_sets.get_datasets()

    config_dict ={}
    config_dict['save_model'] = True
    config_dict['model'] = model
    config_dict['model_name'] = "model_de_test_nouvelle_archi.pt"
    config_dict['optimizer'] = optimizer5
    config_dict['epoch'] = 60
    config_dict['long_patience'] = False
    config_dict['train_set'] = sets['fatset']
    config_dict['test_set'] = sets['fatset']
    return config_dict                                                                    
