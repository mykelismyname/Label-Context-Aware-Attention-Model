import torch
import shutil

def save_model_check_point(state, is_best, checkpoint_path, best_model_path):
    """
    :param modelstate:
    :param is_best: checkpoint we would want to save
    :param checkpoint_path: Is the current checkpoint having a better performe, i.e. is current val loss > previous val loss
    :param best_model_path: path to save checkpoint
    :return:
    """
    torch.save(state, checkpoint_path)
    if is_best:
        new_best_path = best_model_path
        shutil.copyfile(checkpoint_path, new_best_path)

def load_check_point(checkpoint_path, model, optimizer):
    """
    :param checkpoint_path: path to the saved checkpoint
    :param model: model to load checkpoint weights into
    :param optimizer: optimizer defined during training
    """
    chk_point = torch.load(checkpoint_path)
    model.load_state_dict(chk_point['state_dict'])
    optimizer.load_state_dict(chk_point['optimizer'])
    val_loss_min = chk_point['valid_loss_min']
    return model, optimizer, chk_point['epoch'], val_loss_min.item()