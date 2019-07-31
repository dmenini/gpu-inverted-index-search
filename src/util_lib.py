#!/usr/bin/env python3

import sys

class ProgressBar(object):
	BAR_LEN = 60

	def __init__(self, total, status=''):
		self._total = total
		self._status = status

	def update(self, count):
		filled_len = int(round(self.BAR_LEN * count / float(self._total-1)))
		percents = round(100.0 * count / float(self._total-1), 1)
		bar = '=' * filled_len + '-' * (self.BAR_LEN - filled_len)
		sys.stdout.write('[%s] %s%s %s\r' % (bar, percents, '%', self._status))
		sys.stdout.flush()
		if count == self._total-1:
			print("\n")