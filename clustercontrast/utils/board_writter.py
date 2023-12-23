from torch.utils.tensorboard import SummaryWriter


class BoardWriter(object):
    boardWriter = None

    @staticmethod
    def setWriter(cls, name):
        cls.boardWriter = SummaryWriter(name)

