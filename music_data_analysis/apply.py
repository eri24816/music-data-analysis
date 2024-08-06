import multiprocessing
import signal
from tqdm import tqdm
from .processor import Processor
from .data_access import Dataset


def apply_to_dataset(dataset: Dataset, processor: Processor, num_processes: int = 1):
    if num_processes == 1:
        apply_to_dataset_single_proc(dataset, processor)
    else:
        apply_to_dataset_multi_proc(dataset, processor, num_processes)


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
    dataset: Dataset, processor: Processor, num_processes: int
):
    songs = dataset.songs()
    with multiprocessing.Pool(
        num_processes, initializer=process_init, initargs=(processor,)
    ) as p:
        signal.signal(signal.SIGINT, sigint_handler)
        list(tqdm(p.imap(process_task, songs), total=len(songs)))


def apply_to_dataset_single_proc(dataset: Dataset, processor: Processor):
    processor.prepare()
    for song in tqdm(dataset.songs()):
        processor.process(song)
