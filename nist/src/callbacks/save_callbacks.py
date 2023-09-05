import torch
from pytorch_lightning import Callback


class SaveCb(Callback):
    def __init__(self, save_interval=1) -> None:
        super().__init__()
        self.save_interval = save_interval

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        if (pl_module.current_epoch + 1) % self.save_interval == 0:
            if pl_module.cfg.ema:
                with pl_module.ema_map.average_parameters():
                    torch.save(
                        pl_module.map_t.state_dict(),
                        f"map_{pl_module.current_epoch+1}_ema.pth",
                    )
            else:
                torch.save(
                    pl_module.map_t.state_dict(), f"map_{pl_module.current_epoch+1}.pth"
                )
            torch.save(
                pl_module.f_net.state_dict(), f"f_net_{pl_module.current_epoch+1}.pth"
            )
