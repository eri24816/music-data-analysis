from .data_access import Song


class Processor:

    max_num_processes = 128
    input_props: list[str]|None = None
    output_props: list[str]|None = None

    def prepare_main_process(self):
        '''
        Called in the main process before any other process is started.
        '''
        pass

    def prepare(self):
        """
        Must be called once before process is called.
        Load resources (e.g. models) here instead of __init__, so they won't be copied to other processes when using multiprocessing.
        When using multiprocessing, prepare() is called in each process.
        """
        pass

    def process(self, song: Song):

        self.process_impl(song)

    def process_impl(self, song: Song):
        raise NotImplementedError()
