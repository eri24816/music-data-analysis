import multiprocessing
from tqdm import tqdm
from .processor import Processor
from .data_access import Dataset


def apply_to_dataset(dataset: Dataset, processor: Processor, num_processes: int = 1):
    if num_processes == 1:
        apply_to_dataset_single_proc(dataset, processor)
    else:
        apply_to_dataset_multi_proc(dataset, processor, num_processes)


def process_init(processor_: Processor):
    global processor
    processor = processor_
    processor.prepare()


def process_task(song):
    global processor
    processor.process(song)


def apply_to_dataset_multi_proc(
    dataset: Dataset, processor: Processor, num_processes: int
):
    songs = dataset.songs()
    with multiprocessing.Pool(
        num_processes, initializer=process_init, initargs=(processor,)
    ) as p:
        list(tqdm(p.imap(process_task, songs), total=len(songs)))


def apply_to_dataset_single_proc(dataset: Dataset, processor: Processor):
    processor.prepare()
    for song in tqdm(dataset.songs()):
        processor.process(song)
