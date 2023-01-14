
import datetime as dt
from pathlib import Path
from typing import Union
from torch.utils.tensorboard import SummaryWriter

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if n > 0:
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = float(self.sum) / self.count

def _is_aws_or_gcloud_path(tb_log_dir: str) -> bool:
    return tb_log_dir.startswith("gs://") or tb_log_dir.startswith("s3://")

def _make_path_if_local(tb_log_dir: Union[str, Path]) -> Union[str, Path]:
    if isinstance(tb_log_dir, str) and _is_aws_or_gcloud_path(tb_log_dir):
        return tb_log_dir

    tb_log_dir = Path(tb_log_dir)
    tb_log_dir.mkdir(parents=True, exist_ok=True)
    return tb_log_dir

class SSCL_logger():
    def __init__(self, path):
        self.path = path
        date = dt.datetime.now()
        date = date.strftime("%Y_%m_%d_%H_%M_%S")
        tb_log_dir = self.path + date

        tb_log_dir = _make_path_if_local(tb_log_dir)
        self.logger = SummaryWriter(tb_log_dir)

    def result(self, title, log_data, n_iter):
        self.logger.add_scalar(title, log_data, n_iter)

    def model_graph(self, model, dummy_input):
        self.logger.add_graph(model, dummy_input)

__all__ = ['AverageMeter', 'SSCL_logger']