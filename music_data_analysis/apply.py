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
    with multiprocessing.Pool(
        num_processes, initializer=process_init, initargs=(processor,)
    ) as p:
        list(tqdm(p.imap(process_task, dataset.songs()), total=len(dataset.songs())))


def apply_to_dataset_single_proc(dataset: Dataset, processor: Processor):
    for song in dataset.songs():
        processor.process(song)
