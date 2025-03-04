import multiprocessing
import signal
import traceback
import time
from tqdm import tqdm
from .processor import Processor
from .data_access import Dataset


def apply_to_dataset(dataset: Dataset, processor: Processor, num_processes: int = 1, verbose=True, num_shards: int = 1, shard_id: int = 0):
    if verbose:
        shard_str = f"[shard {shard_id}] " if num_shards > 1 else ""
        print(f"{shard_str}Applying {processor.__class__.__name__} to dataset {dataset.dataset_path}")

    if num_processes > processor.max_num_processes:
        num_processes = processor.max_num_processes
        
    if num_processes == 1:
        apply_to_dataset_single_proc(dataset, processor, verbose, num_shards, shard_id)
    else:
        apply_to_dataset_multi_proc(dataset, processor, num_processes, verbose, num_shards, shard_id)


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
        processor.process(song)
    except Exception as e:
        traceback.print_exc()
    
def sigint_handler(sig, frame):
    print("SIGINT received, terminating processes")
    time.sleep(1)
    exit(1)

def apply_to_dataset_multi_proc(
    dataset: Dataset, processor: Processor, num_processes: int, verbose=True, num_shards: int = 1, shard_id: int = 0
):
    processor.prepare_main_process()
    songs = dataset.songs(num_shards, shard_id)
    with multiprocessing.Pool(
        num_processes, initializer=process_init, initargs=(processor,)
    ) as p:
        signal.signal(signal.SIGINT, sigint_handler)
        if verbose:
            list(tqdm(p.imap(process_task, songs), total=len(songs), desc=f"{processor.__class__.__name__} ({num_processes} processes)"))
        else:
            list(p.imap(process_task, songs))


def apply_to_dataset_single_proc(dataset: Dataset, processor: Processor, verbose=True, num_shards: int = 1, shard_id: int = 0):
    processor.prepare_main_process()
    processor.prepare()
    if verbose:
        for song in tqdm(dataset.songs(num_shards, shard_id)):
            processor.process(song)
    else:
        for song in dataset.songs(num_shards, shard_id):
            processor.process(song)
