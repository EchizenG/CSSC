import numpy as np
import math
from Pyfhel import Pyfhel, PyPtxt, PyCtxt
import tarfile
import psutil
import ssgetpy
from scipy.io import mmread
import os
from tqdm import tqdm

import numpy as np

class Result:
    def __init__(self, split_matrix, col_idx, num_pivot_row, num_col, row_map=None):
        self.split_matrix = split_matrix
        self.col_idx = col_idx
        self.row_map = row_map
        self.num_pivot_row = num_pivot_row
        self.num_col = num_col

def split_matrix(matrix, column_indices, max_size):
    """
    Splits a triangular matrix into column blocks such that each block minimizes zeros
    while staying within a fixed size.

    Args:
        matrix (np.ndarray): The input triangular matrix.
        max_size (int): The maximum size (number of non-zero elements) for each block.

    Returns:
        list: A list of blocks, where each block is a 2D numpy array.
    """
    # rows, cols = matrix.shape
    non_zero_columns = np.any(matrix != 0, axis=0)
    cols = np.sum(non_zero_columns)
    non_zero_rows = np.any(matrix != 0, axis=1)
    rows = np.sum(non_zero_rows)

    blocks = []
    indices = []
    col_start = 0
    num_pivot_row = rows
    res_max_row = []
    res_num_col = []

    while col_start < cols:
        col_end = col_start
        store_in_cipher = 0
        max_row = num_pivot_row

        # Expand columns until the block size exceeds max_size
        while col_end < cols:
            if store_in_cipher + num_pivot_row > max_size:
                col_non_zero = np.count_nonzero(matrix[:, col_end])
                num_pivot_row = col_non_zero
                break
            store_in_cipher += num_pivot_row
            col_end += 1

        # Extract the block
        block = matrix[:max_row, col_start:col_end].ravel(order='F')
        indices_block_array = [
            col_index[col_start:col_end]
            for col_index in column_indices
        ]
        max_length = max(len(arr) for arr in indices_block_array)

        # Pad arrays to the maximum length
        index = np.array([
            np.pad(arr, (0, col_end - col_start - len(arr)), constant_values=-1) 
            for arr in indices_block_array 
            if len(arr) > 0  # Ignore empty arrays
        ]).ravel(order='F')

        blocks.append(block)
        indices.append(index)
        res_max_row.append(max_row)
        res_num_col.append(col_end - col_start)

        # Move to the next set of columns
        col_start = col_end

    return blocks, indices, res_max_row, res_num_col

def process_matrix(matrix):
    # Step 1: Count non-zero elements in each row
    non_zero_counts = np.count_nonzero(matrix, axis=1)

    # Step 2: Sort rows based on non-zero counts (descending)
    sorted_indices = np.argsort(-non_zero_counts)  # Negative for descending order
    matrix = matrix[sorted_indices]  # Directly reorder without extra list
    row_mapping = sorted_indices.tolist()  # Convert to list for memory efficiency

    # Step 3: Shift non-zero elements to the left
    column_indices = []
    for i in tqdm(range(matrix.shape[0])):
        row = matrix[i]
        nz_idx = np.flatnonzero(row)  # Find non-zero indices
        row[:len(nz_idx)] = row[nz_idx]  # Move non-zero elements to the left
        row[len(nz_idx):] = 0  # Zero out the remaining part
        column_indices.append(nz_idx.tolist())  # Store original indices

    return matrix, column_indices, row_mapping

def reverse_process(permuted_matrix, column_indices, row_mapping):
    # Step 1: Reconstruct rows by placing non-zero elements at original column indices
    original_order_matrix = np.zeros_like(permuted_matrix)

    for i, (row, col_indices) in enumerate(zip(permuted_matrix, column_indices)):
        for val, col in zip(row[row != 0], col_indices):
            original_order_matrix[i, col] = val

    # Step 2: Rearrange rows back to the original order
    original_matrix = np.zeros_like(original_order_matrix)
    for sorted_index, original_index in enumerate(row_mapping):
        original_matrix[original_index] = original_order_matrix[sorted_index]

    return original_matrix

def getCSR3(permuted_matrix, column_indices, block_size):
    blocks, indices, num_pivot_row, num_col = split_matrix(permuted_matrix, column_indices, block_size)

    return Result(blocks, indices, num_pivot_row, num_col)

def getPrmedVec(multip_col_idx, vector):
    res = []

    for col_idx in multip_col_idx:
        one_vec = []
        for idx in col_idx:
            if idx >=0:
                one_vec.append(vector[idx])
            else:
                one_vec.append(0)
        res.append(one_vec)

    return res

def runSum(HE, obj, num_pivot_row, num_col):
    step_one_res = []

    #step1: add all together in its own ciphertext
    for oneRes, rows, cols in zip(obj, num_pivot_row, num_col):
        step = 1
        cipher = PyCtxt(copy_ctxt=oneRes)
        while step < cols:
            cipher += HE.rotate(cipher, step*rows, in_new_ctxt=True)
            step *= 2
        step_one_res.append(cipher)

    #step2: add between the ciphertexts
    max_row = num_pivot_row[0]
    for idx, cipher in enumerate(step_one_res):
        if idx == 0:
            res = cipher
        if idx != 0:
            mask = HE.encode([1]*num_pivot_row[idx])
            res += cipher * mask

    res = res.decrypt()[:max_row]

    return res

def totalSum(HE, obj, num_pivot_row, num_col, logging):
    step_one_res = []
    num_rot = 0
    num_cmul = 0
    num_add = 0

    # step1: add all together in its own ciphertext
    for oneRes, rows, cols in zip(obj, num_pivot_row, num_col):
        step = 1
        cipher = PyCtxt(copy_ctxt=oneRes)

        for j in range(cols.bit_length() - 2, -1, -1):
            cipher += HE.rotate(cipher, step * rows, in_new_ctxt=True)
            num_rot += 1
            num_add += 1
            step *= 2

            if (cols >> j) & 1:
                cipher = oneRes + HE.rotate(cipher, rows, in_new_ctxt=True)
                num_rot += 1
                num_add += 1
                step += 1

        step_one_res.append(cipher)

    # step2: add between the ciphertexts (with ct-pt masks)
    max_row = num_pivot_row[0]
    for idx, cipher in enumerate(step_one_res):
        if idx == 0:
            res = cipher
        else:
            mask = HE.encode([1] * num_pivot_row[idx])
            res += cipher * mask
            num_cmul += 1      # ct-pt mult
            num_add += 1       # add into res

    res = res.decrypt()[:max_row]
    return res, num_rot, num_cmul, num_add


def getMatrixVector(matrixName):
    if matrixName == 'cca':
        matrix = ssgetpy.fetch(2437)[0]
    else:
        matrix = ssgetpy.fetch(matrixName)[0]
    tar_path = matrix.download(destpath='../../data/')[0]
    extracted_folder = '../../data/' + matrixName
    
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=extracted_folder)

    mtx_file = None
    for root, dirs, files in os.walk(extracted_folder):
        for file in files:
            if len(files) > 1:
                mtx_file = os.path.join(root, matrixName+".mtx")
                break
            else:
                mtx_file = os.path.join(root, file)
                break

    if mtx_file:        
        matrix_sparse = mmread(mtx_file).tocsr()
        matrix_dense = matrix_sparse.toarray()

        vector_length = matrix_dense.shape[1]
        vector_dense = np.tile(np.arange(1, 11), vector_length // 10 + 1)[:vector_length]

    return matrix_dense, vector_dense