from .data_access import Song


class Processor:

    max_num_processes = 128
    input_props: list[str]|None = None
    output_props: list[str]|None = None

    def prepare(self):
        """
        Must be called once before process is called.
        Load resources (e.g. models) here instead of __init__, so they won't be copied to other processes when using multiprocessing.
        When using multiprocessing, prepare() is called in each process.
        """
        pass

    def process(self, song: Song):
        if self.input_props is not None:
            for input_prop in self.input_props:
                if not song.exists(input_prop):
                    raise FileNotFoundError(f"File {input_prop} not found for song {song.song_name}. It is required by {self.__class__.__name__}")
                
        if self.output_props is not None:
            all_output_props_exist = True

            for output_prop in self.output_props:
                if not song.exists(output_prop):
                    all_output_props_exist = False
                    break
            if all_output_props_exist:
                return
            
            for output_prop in self.output_props:
                song.get_new_path(output_prop).parent.mkdir(parents=True, exist_ok=True)

        self.process_impl(song)

    def process_impl(self, song: Song):
        raise NotImplementedError()
