import argparse
import numpy as np
from utils import getCSR3, getPrmedVec, runSum, getMatrixVector, totalSum, process_matrix
from Pyfhel import Pyfhel, PyPtxt, PyCtxt
import time
import tarfile
import psutil
import ssgetpy
from scipy.io import mmread
import os
import logging
import math
import sys
import traceback
import tracemalloc

cipherSize = 2**13
HE = Pyfhel()
HE.contextGen(scheme='bfv', n=cipherSize, t_bits=20) #m is 2048 or 8192
HE.keyGen()
HE.relinKeyGen()
HE.rotateKeyGen()

# logging.basicConfig(filename='CSR3permvec.log', level=logging.INFO, 
#                     format='%(asctime)s - %(levelname)s - %(message)s')

def measure_memory_usage(func, *args, **kwargs):
    """Measure peak memory usage of a function."""
    process = psutil.Process(os.getpid())
    tracemalloc.start()

    result = func(*args, **kwargs)

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    logging.info(f"Peak memory usage: {peak / 1024**2:.2f} MB")

    return result

def run(matrix, vector):
    logging.info("preprocessing matrix and vector")
    print("preprocessing matrix and vector")

    # nonzeros_per_row = np.count_nonzero(matrix, axis=1)
    
    # # Compute metrics
    # avg_nonzeros_per_row = np.mean(nonzeros_per_row)
    # std_nonzeros_per_row = np.std(nonzeros_per_row)
    # max_nonzeros_in_row = np.max(nonzeros_per_row)

    # logging.info(f"Average nonzeros per row: {avg_nonzeros_per_row} Standard deviation of nonzeros per row: {std_nonzeros_per_row} Maximum nonzeros in a row: {max_nonzeros_in_row}")

    # exit()
    # start_time = time.process_time()
    UpperTriangleMatrix, column_indices, row_mapping = process_matrix(matrix)
    # end_time = time.process_time()
    # elapsed_time = end_time - start_time
    # logging.info(f"pre-process in {elapsed_time:.2f} seconds")

    global cipherSize
    cipherSize = int(cipherSize/2)
    # cipherSize = 7
    n = UpperTriangleMatrix.shape[0]
    finalRes = [0]*len(row_mapping)
    total_cipher = 0
    total_rot = 0
    total_cmul = 0
    total_mult = 0    # ct-ct mult
    total_add = 0     # ct add
    elapsed_time = 0

    size = 0.0

    for i in range(0, n, cipherSize):
        sub_ut_matrix = UpperTriangleMatrix[i:i + cipherSize]  # Select cipherSize rows
        
        if sub_ut_matrix.sum() == 0:
            break #this is upper triangle matrix, if this is all zero matrix, the followings are all zeros

        sub_col_idx = column_indices[i:i + cipherSize]  # Select cipherSize elements
        sub_row_map = row_mapping[i:i + cipherSize]  # Select cipherSize elements

        tempRes = [0]*len(row_mapping)

        start_time = time.process_time()
        processedMatrix = getCSR3(sub_ut_matrix, sub_col_idx, cipherSize)
        # size += sys.getsizeof(processedMatrix.col_idx)
        # continue
        permedVector = getPrmedVec(processedMatrix.col_idx, vector)

        encryVector = []
        encryValue = []
        
        numBlock = len(permedVector)
        total_cipher += numBlock
        total_mult += numBlock
        
        for j in range(numBlock):
            encryVector.append(HE.encrypt(permedVector[j]))
            encryValue.append(HE.encrypt(processedMatrix.split_matrix[j]))
        
        encRes = []
        for k in range(numBlock):
            extRes = encryValue[k] * encryVector[k]
            HE.relinearize(extRes)
            encRes.append(extRes)
        
        res, num_rot, num_cmul, num_add = totalSum(
            HE, encRes, processedMatrix.num_pivot_row, processedMatrix.num_col, logging)
        total_add += num_add
        total_rot += num_rot
        total_cmul += num_cmul

        for sorted_index, original_index in enumerate(sub_row_map):
            if sorted_index < len(res):
                tempRes[original_index] = res[sorted_index]
            else:
                break

        finalRes = [a + b for a, b in zip(finalRes, tempRes)]
        end_time = time.process_time()
        elapsed_time += end_time - start_time

    logging.info(f"Total time taken: {elapsed_time:.2f} seconds")

    logging.info("===== HE Operation Counts (Whole Algorithm) =====")
    logging.info(f"HE-rot          : {total_rot}")
    logging.info(f"HE-add          : {total_add}")
    logging.info(f"HE-mult (ct-ct) : {total_mult}")
    logging.info(f"HE-const-mult   : {total_cmul}  (ct-pt)")
    logging.info("=================================================")
        

    return finalRes


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='')
    args = parser.parse_args()
    logging.basicConfig(filename='HEcount.log', level=logging.INFO, 
                    format='%(message)s')
    try:
        logging.info(f"{args.name}")
        matrix, vector = getMatrixVector(args.name)

        # matrix = np.array([
        #     [0, 0, 0, 0, 0, 5, 0, 3, 0, 3],
        #     [0, 7, 0, 0, 5, 0, 4, 2, 0, 0],
        #     [0, 3, 9, 0, 0, 5, 1, 0, 3, 0],
        #     [9, 8, 2, 2, 3, 0, 3, 6, 8, 7],
        #     [0, 0, 9, 0, 0, 0, 0, 0, 0, 0],
        #     [0, 0, 2, 2, 0, 8, 0, 0, 7, 0],
        #     [6, 1, 4, 0, 6, 3, 0, 0, 6, 0],
        #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        #     # [0, 0, 0, 0, 0, 0, 0, 0, 0, 3],
        #     # [8, 7, 6, 3, 1, 3, 1, 6, 2, 2]
        # ])
        # vector = np.random.randint(low=0, high=10, size=(10))

        res = measure_memory_usage(run, matrix, vector)

        # truth = np.matmul(matrix, vector, dtype=np.int32)

        # if (res == truth).all():
        #     elapsed_time = end_time - start_time
        #     logging.info(f"right: {args.name}")
        #     logging.info(f"completed in {elapsed_time:.2f} seconds")
        # else:
        #     elapsed_time = end_time - start_time
        #     logging.info(f"wrong: {args.name}")
        #     logging.info(f"completed in {elapsed_time:.2f} seconds")
    except Exception as e:
        logging.error(f"Got error message: {e} {args.name}")
        traceback.print_exc()

    logging.info(f"-----------------------------------------------------")