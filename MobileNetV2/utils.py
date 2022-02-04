import os
import config
import torch

def plot_to_tensorboard(writer, train_loss, test_loss, test_accuracy, tensorboard_step):
   writer.add_scalar("train_loss", train_loss, global_step=tensorboard_step)
   writer.add_scalar("test_loss", test_loss, global_step=tensorboard_step)
   writer.add_scalar("test_accuracy", test_accuracy, global_step=tensorboard_step)


def save_checkpoint(model, optimizer, file_name):
    file_path = os.path.join(config.CHECKPOINTS_DIR, file_name)
    print("===> Saving checkpoint")

    checkpoint = {
        "sate_dict": model.state_dict(),
        "optimizer": optimizer.state_dict()
    }

    torch.save(checkpoint, file_path)

def load_checkpoint(file_name, model, optimizer, lr):

    print("===> Loading checkpoint")
    file_path = os.path.join(config.CHECKPOINTS_DIR, file_name)
    checkpoint = torch.load(file_path, map_location="cuda")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr