from os import listdir
from os.path import isfile, join
import Queue
import threading


# list files in a folder and put them in to a queue for multi-threading processing
def multi_thread_process_files(dir_path, file_extension, num_threads, process_func,
                               proc_desc='processed', args=None, multi=None):
    onlyfiles = [f for f in listdir(dir_path) if isfile(join(dir_path, f))]
    pdf_queque = Queue.Queue(len(onlyfiles))
    num_pdfs = 0
    files = None if multi is None else []
    for f in onlyfiles:
        if f.endswith('.' + file_extension):
            if multi is None:
                pdf_queque.put_nowait(join(dir_path, f))
            else:
                files.append(join(dir_path, f))
                if len(files) >= multi:
                    pdf_queque.put_nowait(files)
                    files = []
            num_pdfs += 1
    if files is not None and len(files) > 0:
        pdf_queque.put_nowait(files)
    thread_num = min(num_pdfs, num_threads)
    arr = [process_func] if args is None else [process_func] + args
    arr.insert(0, pdf_queque)
    for i in range(thread_num):
        t = threading.Thread(target=multi_thread_do, args=tuple(arr))
        t.daemon = True
        t.start()
    pdf_queque.join()
    print('{0} files {1}'.format(num_pdfs, proc_desc))


def multi_thread_do(q, func, *args):
    while True:
        p = q.get()
        func(p, *args)
        q.task_done()


def simple_do(f, t):
    print("{}".format(f))


def test():
    multi_thread_process_files('/Users/jackey.wu/Documents/working/Alistair/fasion brain', 'docx', 3,
                               simple_do, args=[' -'])

if __name__ == "__main__":
    test()
