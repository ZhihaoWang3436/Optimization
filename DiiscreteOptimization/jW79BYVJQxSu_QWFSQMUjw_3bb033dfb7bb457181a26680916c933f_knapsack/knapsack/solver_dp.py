#!/usr/bin/python
# -*- coding: utf-8 -*-

def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    firstLine = lines[0].split()
    item_count = int(firstLine[0])
    capacity = int(firstLine[1])

    values = [0] * item_count
    weights = [0] * item_count
    for i in range(1, item_count + 1):
        line = lines[i]
        parts = line.split()
        values[i - 1] = int(parts[0])
        weights[i - 1] = int(parts[1])

    dp_array = [[0] * (capacity + 1) for _ in range(item_count)]
    for j in range(weights[0], capacity + 1):
        dp_array[0][j] = values[0]
    for i in range(1, item_count):
        for j in range(capacity + 1):
            if weights[i] > j:
                dp_array[i][j] = dp_array[i - 1][j]
            else:
                dp_array[i][j] = max(dp_array[i - 1][j], dp_array[i - 1][j - weights[i]] + values[i])

    value = dp_array[item_count - 1][capacity]
    taken = [0] * item_count
    i, j = item_count - 1, capacity
    while i > 0 and j > 0:
        if dp_array[i][j] != dp_array[i - 1][j]:
            taken[i] = 1
            j -= weights[i]
        i -= 1
    if j >= weights[0]:
        taken[0] = 1


    # prepare the solution in the specified output format
    output_data = str(value) + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, taken))
    return output_data


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/ks_4_0)')
