# Parallel version for common matrix operations, using numpy and threading API

import numpy as np
from threading import Thread


def thread_addition_worker(matrix_a, matrix_b, matrix_c, num_threads=4, rank=0):
    """
    Add two matrices in parallel
    :param matrix_a: matrix A
    :param matrix_b: matrix B
    :param matrix_c: matrix C -- result
    :param num_threads: number of threads
    :param rank: rank of current thread
    :return: matrix A + matrix B

    This is a row-wise parallelization of matrix addition.
    """
    # check if matrix_a and matrix_b are the same size
    if matrix_a.shape != matrix_b.shape:
        raise ValueError("Matrix A and Matrix B must be the same size")

    # get the size of matrix_a
    row, _ = matrix_a.shape

    # calculate the number of rows each thread should handle
    num_rows = row // num_threads

    # calculate the starting and ending index for each thread
    start_index = rank * num_rows
    end_index = start_index + num_rows

    # add the matrices and store the result in matrix_c
    matrix_c[start_index:end_index, :] = (
        matrix_a[start_index:end_index, :] + matrix_b[start_index:end_index, :]
    )


def matrix_addition(matrix_a, matrix_b, num_threads=4):
    threads = []
    matrix_sum = np.zeros(matrix_a.shape)

    # create threads
    for i in range(num_threads):
        threads.append(
            Thread(
                target=thread_addition_worker,
                args=(matrix_a, matrix_b, matrix_sum, num_threads, i),
            )
        )

    # start threads
    for thread in threads:
        thread.start()

    # wait for threads to finish
    for thread in threads:
        thread.join()

    return matrix_sum


def thread_multiplication_worker(matrix_a, matrix_b, matrix_c, num_threads=4, rank=0):
    """
    Multiply two matrices in parallel
    :param matrix_a: matrix A
    :param matrix_b: matrix B
    :param matrix_c: matrix C -- result
    :param num_threads: number of threads
    :param rank: rank of current thread
    """

    # check if matrix_a and matrix_b can be multiplied
    if matrix_a.shape[1] != matrix_b.shape[0]:
        raise ValueError("Matrix A and Matrix B cannot be multiplied")

    # get the size of matrix_a
    row, _ = matrix_a.shape

    # calculate the number of rows each thread should handle
    num_rows = row // num_threads

    # calculate the starting and ending index for each thread
    start_index = rank * num_rows
    end_index = start_index + num_rows

    # multiply the matrices and store the result in matrix_c
    matrix_c[start_index:end_index, :] = matrix_a[start_index:end_index, :] @ matrix_b


def matrix_multiplication_rowwise(matrix_a, matrix_b, num_threads=4):
    threads = []
    matrix_product = np.zeros((matrix_a.shape[0], matrix_b.shape[1]))

    # create threads
    for i in range(num_threads):
        threads.append(
            Thread(
                target=thread_multiplication_worker,
                args=(matrix_a, matrix_b, matrix_product, num_threads, i),
            )
        )

    # start threads
    for thread in threads:
        thread.start()

    # wait for threads to finish
    for thread in threads:
        thread.join()

    return matrix_product


def thread_multiplication_block_worker(
    matrix_a, matrix_b, matrix_c, num_blocks=(2, 2), rank=(0, 0)
):
    """
    C_ij = A_ik * B_kj

    """
    # final shape of matrix_c
    row, col = matrix_c.shape
    # the starting and ending index for each block
    start_row = rank[0] * row // num_blocks[0]
    end_row = start_row + row // num_blocks[0]

    start_col = rank[1] * col // num_blocks[1]
    end_col = start_col + col // num_blocks[1]

    # calculate the product for each block
    matrix_c[start_row:end_row, start_col:end_col] = (
        matrix_a[start_row:end_row, :] @ matrix_b[:, start_col:end_col]
    )


def matrix_multiplication_blockwise(matrix_a, matrix_b, num_blocks=(2, 2)):
    threads = []
    matrix_product = np.zeros((matrix_a.shape[0], matrix_b.shape[1]))

    # create threads
    for i in range(num_blocks[0]):
        for j in range(num_blocks[1]):
            threads.append(
                Thread(
                    target=thread_multiplication_block_worker,
                    args=(matrix_a, matrix_b, matrix_product, num_blocks, (i, j)),
                )
            )

    # start threads
    for thread in threads:
        thread.start()

    # wait for threads to finish
    for thread in threads:
        thread.join()

    return matrix_product
