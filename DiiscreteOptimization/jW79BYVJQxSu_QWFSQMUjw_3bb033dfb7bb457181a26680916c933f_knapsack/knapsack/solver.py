#!/usr/bin/python
# -*- coding: utf-8 -*-
from collections import namedtuple
Item = namedtuple("Item", ['index', 'value', 'weight'])

from queue import PriorityQueue


class Node:
    def __init__(self, index, value, weight, father, flag):
        self.index = index
        self.value = value
        self.weight = weight
        self.father = father
        self.flag = flag
        self.bound = 0

    def __lt__(self, other):
        return self.bound > other.bound

    # def __str__(self):
    #     return f"{self.index}, {self.value}, {self.weight}, {self.flag}, {self.bound}"

class BranchAndBound:
    def __init__(self, items, capacity):
        self.items = items
        self.capacity = capacity
        root_node = Node(-1, 0, 0, None, -1)
        root_node.bound = self.calculate_bound(root_node)
        self.root = root_node
        self.contain = PriorityQueue()  # 优先级队列，按照bound排列
        self.contain.put(self.root)
        self.best_node = self.root  # 值最优节点

    def calculate_bound(self, node):
        bound = node.value  # 注意上界bound要从当前值value开始
        cum_weight = node.weight
        for i in range(node.index+1, len(self.items)):
            item = self.items[i]
            cum_weight += item.weight
            if cum_weight <= self.capacity:
                bound += item.value
            else:
                bound += ((self.capacity - cum_weight) / item.weight + 1) * item.value
                break  # 注意到分割item要退出，不然后面反向减
        return bound

    def branch(self):
        while self.contain.qsize():
            node = self.contain.get()
            index = node.index + 1
            if index < len(self.items):
                item = self.items[index]
                if node.bound > self.best_node.value:
                    if node.weight + item.weight <= self.capacity:
                        left_node = Node(index, node.value+item.value, node.weight+item.weight, node, 1)
                        left_node.bound = self.calculate_bound(left_node)
                        self.contain.put(left_node)
                        if left_node.value > self.best_node.value:
                            self.best_node = left_node
                    right_node = Node(index, node.value, node.weight, node, 0)
                    right_node.bound = self.calculate_bound(right_node)
                    self.contain.put(right_node)

def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    firstLine = lines[0].split()
    item_count = int(firstLine[0])
    capacity = int(firstLine[1])

    items = []

    for i in range(1, item_count+1):
        line = lines[i]
        parts = line.split()
        items.append(Item(i-1, int(parts[0]), int(parts[1])))
    items.sort(key=lambda x: x.value/x.weight, reverse=True)

    bab = BranchAndBound(items, capacity)
    bab.branch()
    value = bab.best_node.value
    taken = [0] * item_count
    node = bab.best_node
    while node.father:
        taken[items[node.index].index] = node.flag
        node = node.father
    
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
