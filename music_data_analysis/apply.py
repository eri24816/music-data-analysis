import multiprocessing
import signal
from tqdm import tqdm
from .processor import Processor
from .data_access import Dataset


def apply_to_dataset(dataset: Dataset, processor: Processor, num_processes: int = 1, verbose=True):
    if verbose:
        print(f"Applying {processor.__class__.__name__} to dataset {dataset.dataset_path}")
    if num_processes == 1:
        apply_to_dataset_single_proc(dataset, processor, verbose)
    else:
        apply_to_dataset_multi_proc(dataset, processor, num_processes, verbose)


def worker_signal_handler(sig, frame):
    """
    Multiprocessing ignores KeyboardInterrupt, so we need to catch it and raise another exception
    """
    print("SIGINT received, terminating process")
    raise RuntimeError("SIGINT received, terminating process")


def process_init(processor_: Processor):
    signal.signal(signal.SIGINT, worker_signal_handler)
    global processor
    processor = processor_
    processor.prepare()


def process_task(song):
    global processor
    processor.process(song)

def sigint_handler(sig, frame):
    print("SIGINT received, terminating processes")
    exit(1)

def apply_to_dataset_multi_proc(
    dataset: Dataset, processor: Processor, num_processes: int, verbose=True
):
    songs = dataset.songs()
    with multiprocessing.Pool(
        num_processes, initializer=process_init, initargs=(processor,)
    ) as p:
        signal.signal(signal.SIGINT, sigint_handler)
        if verbose:
            list(tqdm(p.imap(process_task, songs), total=len(songs)))
        else:
            list(p.imap(process_task, songs))


def apply_to_dataset_single_proc(dataset: Dataset, processor: Processor, verbose=True):
    processor.prepare()
    if verbose:
        for song in tqdm(dataset.songs()):
            processor.process(song)
    else:
        for song in dataset.songs():
            processor.process(song)
