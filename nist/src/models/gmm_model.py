from src.models.base_model import BaseModule
from src.viz.points import draw_source_target

# pylint: disable=R0901,abstract-method,


class GMMModule(BaseModule):
    def get_real_data(self, batch):
        x_data, y_data = batch
        if self.global_step == 1:
            # TODO add a plot_size interface
            draw_source_target(
                x_data,
                y_data,
                "train_data.png",
                self.cfg.num_source_class,
                self.cfg.num_target_class,
            )
        return x_data, y_data
