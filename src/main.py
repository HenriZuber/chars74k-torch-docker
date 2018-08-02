import random
import numpy as np
import torch  # pylint: disable=import-error


import train_test
import config

USE_GPU = True
DTYPE = torch.float32
if USE_GPU and torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
    print("using cpu")

# Makin it reproductible
torch.backends.cudnn.deterministic = True
torch.manual_seed(11)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(11)
RANDOM_SEED = np.random.seed(11)
random.seed(11)


# Vrai main
def main():
    config_dict = config.get_config()    
    log_dir = train_test.create_log_dir("/opt/code/my_logs")
    train_test.copy("/opt/code/src/config.py", log_dir)
    train_test.copy("/opt/code/src/models.py", log_dir)   
    model_name = config_dict['model_name']
    model = config_dict['model']
    save_model = config_dict['save_model']
    optimizer = config_dict['optimizer']
    epoch = config_dict['epoch']
    long_patience = config_dict['long_patience']
    train_set = config_dict['train_set']
    test_set = config_dict['test_set']
                                                                        
    print()
    
    train_test.train(
        train_set,
        test_set,
        model,
        optimizer,
        log_dir,
        epochs=epoch,
        long_patience=long_patience,
    )
       
    if save_model:
        torch.save(model, model_name)


if __name__ == '__main__':
    main()
