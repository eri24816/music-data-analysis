from .data_access import Song


class Processor:
    def prepare(self):
        """
        Must be called once before process is called.
        Load resources (e.g. models) here instead of __init__ so it can
        be copied to other processes when using multiprocessing.
        """
        pass

    def process(self, song: Song):
        raise NotImplementedError()
