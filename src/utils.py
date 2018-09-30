# Andreea Musat, May 2018

import sys
import heapq
import numpy as np


def exit(message, code):
	print(message)
	sys.exit(code)


class FixedCapacityMaxHeap(object):

	def __init__(self, capacity):
		self.capacity = capacity
		self.h = []

	def push(self, value, priority):
		if len(self.h) < self.capacity:
			heapq.heappush(self.h, (priority, value))
		else:
			heapq.heappushpop(self.h, (priority, value))

	def pop(self):
		if len(self.h) >= 0:
			return heapq.nlargest(1, self.h)[0]
		else:
			return None

if __name__ == '__main__':
	h = FixedCapacityMaxHeap(3)
	h.push('ana', 0.3)
	h.push('aka', 0.2)
	h.push('bana', 0.8)
	h.push('aza', 0.99)
	h.push('kk', 0.87)

	print(h.pop())
