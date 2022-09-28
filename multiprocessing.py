import multiprocessing
import time

def multiprocessing_func():
    print('Starting to sleep')
    time.sleep(1)
    print('done sleeping')

process_list = []

if __name__ == '__main__':
    tic = time.time()
    for i in range(10):
        p = multiprocessing.Process(target= multiprocessing_func)
        p.start()
        process_list.append(p)

    for process in process_list:
        process.join()

    toc = time.time()

    print('Done in {:.4f} seconds'.format(toc-tic))