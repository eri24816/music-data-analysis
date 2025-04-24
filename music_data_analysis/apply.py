import multiprocessing
import signal
import traceback
import time
from tqdm import tqdm
from .processor import Processor
from .data_access import Dataset, Song

def check_song_should_be_processed(song: Song, processor: Processor) -> bool:
    if processor.input_props is not None:
        for input_prop in processor.input_props:
            if not song.exists(input_prop):
                raise FileNotFoundError(f"File {input_prop} not found for song {song.song_name}. It is required by {processor.__class__.__name__}")
            
    if processor.output_props is not None:
        all_output_props_exist = True
    
        for output_prop in processor.output_props:
            if not song.exists(output_prop):
                all_output_props_exist = False
                break
        if all_output_props_exist:
            return False
        
        for output_prop in processor.output_props:
            song.get_new_path(output_prop).parent.mkdir(parents=True, exist_ok=True)

    return True

def apply_to_dataset(dataset: Dataset, processor: Processor, num_processes: int = 1, verbose=True, num_shards: int = 1, shard_id: int = 0, overwrite_existing: bool = False):
    if verbose:
        shard_str = f"[shard {shard_id}] " if num_shards > 1 else ""
        print(f"{shard_str}Applying {processor.__class__.__name__} to dataset {dataset.dataset_path}")

    if num_processes > processor.max_num_processes:
        num_processes = processor.max_num_processes
        
    if num_processes == 1:
        apply_to_dataset_single_proc(dataset, processor, verbose, num_shards, shard_id, overwrite_existing)
    else:
        apply_to_dataset_multi_proc(dataset, processor, num_processes, verbose, num_shards, shard_id, overwrite_existing)


def worker_signal_handler(sig, frame):
    """
    Multiprocessing ignores KeyboardInterrupt, so we need to catch it and raise another exception
    """
    print("SIGINT received, terminating process")
    raise RuntimeError("SIGINT received, terminating process")


def process_init(processor_: Processor):
    # signal.signal(signal.SIGINT, worker_signal_handler)
    global processor
    processor = processor_
    processor.prepare()

def process_task(song):
    global processor
    try:
        if check_song_should_be_processed(song, processor):
            processor.process(song)
    except Exception:
        traceback.print_exc()
    
def sigint_handler(sig, frame):
    print("SIGINT received, terminating processes")
    time.sleep(1)
    exit(1)

def apply_to_dataset_multi_proc(
    dataset: Dataset, processor: Processor, num_processes: int, verbose=True, num_shards: int = 1, shard_id: int = 0, overwrite_existing: bool = False
):
    processor.prepare_main_process()
    songs = dataset.songs(num_shards, shard_id)
    def iterable_wrapper(iterable):
        pbar = tqdm(iterable, desc=f"{processor.__class__.__name__} shard {shard_id}/{num_shards} ({num_processes} processes)")
        pbar.set_postfix(skipped=0)
        n_skipped = 0
        for song in pbar:
            if check_song_should_be_processed(song, processor) or overwrite_existing:
                yield song
            else:
                n_skipped += 1
                pbar.set_postfix(skipped=n_skipped)
        pbar.close()

    with multiprocessing.Pool(
        num_processes, initializer=process_init, initargs=(processor,)
    ) as p:
        signal.signal(signal.SIGINT, sigint_handler)
        list(p.imap(process_task, iterable_wrapper(songs)))



def apply_to_dataset_single_proc(dataset: Dataset, processor: Processor, verbose=True, num_shards: int = 1, shard_id: int = 0, overwrite_existing: bool = False):
    processor.prepare_main_process()
    processor.prepare()

    iterable = dataset.songs(num_shards, shard_id)

    iterable = tqdm(iterable, desc=f"{processor.__class__.__name__} shard {shard_id}/{num_shards}")
    iterable.set_postfix(skipped=0)

    n_skipped = 0
    for song in iterable:
        if check_song_should_be_processed(song, processor) or overwrite_existing:
            processor.process(song)
        else:
            n_skipped += 1
            if isinstance(iterable, tqdm):
                iterable.set_postfix(skipped=n_skipped)
