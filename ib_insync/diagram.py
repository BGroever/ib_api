
from .objects import BarData
from .objects import BarDataList
from .ticker import Ticker
from datetime import datetime, timedelta
from dataclasses import dataclass
import inspect 
import numpy as np
import sys
from typing import Union

def validate(f):
	"""
		Validation Decorator 
		
		If the datatype of a parameter is >> self << or unspecified, the entry 
		is skipped during type checking.
	"""
	def wrapper(*args):
		for parameter, arg in zip(inspect.signature(f).parameters.values(), args):
			if parameter.annotation != inspect._empty:
				if not isinstance(arg, parameter.annotation):
					raise TypeError("of function parameter of {}()".format(f.__name__))
		return f(*args)
	return wrapper


class BarDiagram:
	
	def __init__(self, bar_chart_file_name='',verbose=False):
		
		self.count_updates = 0
		self.verbose = verbose
		if bar_chart_file_name: 
			self.bar_chart_file = open(bar_chart_file_name, "w")
		self.indicators = []

	def __iadd__(self, indicator):

		if indicator not in self.indicators:

			# Add required averages for the new indicator to indicator list 
			# if not already existent in self.indicators
			if hasattr(indicator,'averages'):
				for i in range(len(indicator.averages)):
					if indicator.averages[i] not in self.indicators:
						self.indicators.append(indicator.averages[i])
					else:
						indicator.averages[i] = self.indicators[self.indicators.index(indicator.averages[i])]

			self.indicators.append(indicator)
		
		return self 

	def setHistoricalData(self, bars: BarDataList):
		
		assert(bars.keepUpToDate==True)
		
		self.bars = bars
		self.updatedUntil = self.bars[-1].date

		# Initalize statistics
		closes = self.__get_closes_from_bars__()

		for indicator in self.indicators:
			try:
				indicator.setHistoricalData(closes)
			except TypeError:
				indicator.setHistoricalData(bars)
			except TypeError:
				indicator.setHistoricalData()
			except:
				raise Exception("error when update of type '{}' ".format(indicator.__class__))

		self.indicators.sort(reverse=False)

	
	def __get_closes_from_bars__(self) -> []:
		closes   = []
		for index in range(0,len(self.bars)-1):
			closes.append(    self.bars[index].close)
		return closes

	def update(self):
		
		self.count_updates += 1

		# Determine weather a new period has started
		if(self.updatedUntil < self.bars[-1].date):
			self.__update__(self.bars[-2])
			# if self.verbose:
			#   print(self.bars[-2].date,
			#         "{:0.10f}, {:0.10f}, {:0.3f}, {:0.3f}".format(
			#           self.macd.OSMA,
			#           self.bollband.BW,
			#           self.lso.D, self.lso.K))
			#print(self)
			self.updatedUntil = self.bars[-1].date

		else:
			self.__set_intermediate__(self.bars[-1])
			# now = datetime.now()
			# if self.verbose:
			#   print(now - timedelta(microseconds=now.microsecond),
			#         "{:0.10f}, {:0.10f}, {:0.3f}, {:0.3f}      ".format(
			#           self.macd.OSMA,
			#           self.bollband.BW,
			#           self.lso.D, self.lso.K), end='\r')

	def __update__(self, bar: BarData):
		for indicator in self.indicators:
			try:
				indicator.update(bar)
			except TypeError:
				indicator.update(bar.close)
			except TypeError:
				indicator.update()
			except:
				raise Exception("error when update of type '{}' ".format(indicator.__class__))


	def getValues(self):
		values = []
		for indicator in self.indicators:
			if isinstance(indicator,SMA) or isinstance(indicator,EMA):
				values.append(indicator.current)
			if isinstance(indicator,BollingerBand):
				values.append(indicator.TL)
				values.append(indicator.BL)
				values.append(indicator.BW)
			if isinstance(indicator,MACD):
				values.append(indicator.MACD)
				values.append(indicator.SIGNAL)
				values.append(indicator.OSMA)
			if isinstance(indicator,LSO):
				values.append(indicator.D)
				values.append(indicator.K)
			if isinstance(indicator,ZigZag):
				values.append(0 if indicator.last.kind=='MIN' else 1)
				values.append(0)
		return values

	def getValueNames(self):
		names = []
		for indicator in self.indicators:
			if isinstance(indicator,SMA):
				names.append('SMA')
			if isinstance(indicator,EMA):
				names.append(str(indicator))
			if isinstance(indicator,BollingerBand):
				names.append('TL')
				names.append('BL')
				names.append('BW')
			if isinstance(indicator,MACD):
				names.append('MACD')
				names.append('SIG')
				names.append('OSMA')
			if isinstance(indicator,LSO):
				names.append('D')
				names.append('K')
			if isinstance(indicator,ZigZag):
				names.append('trend')
				names.append('ZZ')
		return names

	def __str__(self):
		values = self.getValues()
		string = str(self.bars[-2].date) + ' '
		for value in values[:-1]:
			string += str(value) + ', '
		string += str(values[-1])
		return string


	@validate
	def __set_intermediate__(self, bar: BarData):
		for indicator in self.indicators:
			try:
				indicator.set_intermediate(bar)
			except TypeError:
				indicator.set_intermediate(bar.close)
			except TypeError:
				indicator.set_intermediate()
			except:
				raise Exception("error when set intermediate of type", indicator.__class__)


	def __write_bar_to_file__(self, bar: BarData):
		entry = "{0},{1},{2},{3},{4},{5},{6},{7},{8}\n".format(
			bar.date.timestamp(),   # 0
			bar.open,               # 1
			bar.high,               # 2
			bar.low,                # 3
			bar.close,              # 4
			bar.volume,             # 5
			bar.average,            # 6
			bar.barCount,           # 7
			self.macd)              # 8
		self.bar_chart_file.write(entry)


class Indicator:
	ids = {'SMA': 1, 'EMA': 2, 'MACD': 3, 'BollingerBand': 4, 'LSO': 5, 'ZigZag': 6}
	
	def __lt__(self, other):
		return self.ids[self.__class__.__name__] * 10000 + self.N < self.ids[other.__class__.__name__] * 10000 + other.N


class Average(Indicator):
	"""
		Average
	"""
	def __init__(self):
		self.history = []
		self.current = None

	@validate
	def setHistoricalData(self, array: list):
		pass

	@validate
	def update(self, bar: BarData):
		pass

	@validate
	def set_intermediate(self, price: float):
		pass



class SMA(Average):
	"""
		Simple Moving Average
	"""
	def __init__(self, N: Union[int,float], depth: int=150):
		self.N = N
		self.depth = depth
		Average.__init__(self)


	@validate
	def setHistoricalData(self, array: list):

		assert(len(array) >= self.depth + self.N)

		index = len(array) - self.depth + 1
		while(index <= len(array)):
			self.history.append(np.mean(array[index-self.N:index]))
			index += 1

		self.buffer = array[-N:].copy()
		self.current = self.history[-1]


	@validate
	def update(self, close: float):
		self.buffer.append(close)
		self.buffer.pop(0)
		self.history.append(np.mean(self.buffer))
		self.history.pop(0)
		self.current = self.history[-1]

	@validate
	def set_intermediate(self, close: float):
		self.current = np.mean([*self.buffer[1:], close])

	def __eq__(self, other):
		if self.__class__ != other.__class: return False
		return self.N == other.N and self.depth == other.depth

	def __repr__(self):
		return 'SMA({})'.format(self.N)


class EMA(Average):
	"""
		Exponential Moving Average
	"""
	def __init__(self, N: Union[int,float]):
		self.N = N
		self.alpha = 2/(N+1)
		Average.__init__(self)


	@validate
	def setHistoricalData(self, array: list):

		assert(len(array))

		index = 1
		self.history.append(array[0])
		while index < len(array):
			self.history.append(self.alpha * array[index] + (1-self.alpha) * self.history[-1])
			index += 1

		self.depth   = len(self.history)
		self.current = self.history[-1]

		assert(len(array)==len(self.history))

	@validate
	def update(self, price: float):
		# First IN First OUT (FIFO)
		self.history.append(self.alpha * price + (1-self.alpha) * self.history[-1])     # Append at the end
		self.history.pop(0)                                                             # Remove at the beginning
		self.current = self.history[-1]

	@validate
	def set_intermediate(self, price: float):
		self.current = self.alpha * price + (1-self.alpha) * self.history[-1]

	def __eq__(self, other):
		if self.__class__ != other.__class__: return False
		return self.N == other.N and self.alpha == other.alpha 

	def __repr__(self):
		return 'EMA({})'.format(self.N)




class BollingerBand(Indicator):
	"""
		Bollinger Band
	"""
	def __init__(self,average: Average, N_std: int=2):
		self.averages = []
		self.averages.append(average)
		self.N = average.N
		self.N_std = N_std

	@validate
	def setHistoricalData(self, prices: list):

		assert(len(prices) >= self.averages[0].depth)

		self.prices = prices[-self.averages[0].depth:].copy()
		self.__update__(self.prices)


	@validate
	def update(self, price: float):
		self.prices.append(price)
		self.prices.pop(0)

		self.__update__(self.prices)


	@validate
	def set_intermediate(self, price: float):
		prices = [*self.prices[1:], price]

		self.__update__(prices)


	def __update__(self, prices: []):

		diff = []
		index = len(prices) - self.N
		while(index < len(prices)):
			diff.append(prices[index]-self.averages[0].current)
			index += 1

		sigma = np.sqrt(np.sum(np.array(diff)**2)/(self.N))

		self.TL = self.averages[0].current + self.N_std * sigma     # top line
		self.BL = self.averages[0].current - self.N_std * sigma     # bottom line
		self.ML = self.averages[0].current                          # middle line
		self.BW = (self.TL - self.BL)/self.ML                       # bandwidth

	def __eq__(self, other):
		if self.__class__ != other.__class__: return False
		return self.averages == other.averages and self.N_std == other.N_std 

	def __repr__(self):
		name = 'BollingerBand({}'.format(self.averages[0])
		if self.N_std != 2:
			name += ', N_std={}'.format(self.N_std)
		return name + ')'



class MACD(Indicator):
	"""
		Moving Average Convergence Divergence (MACD)
	"""
	def __init__(self, faster: Average, slower: Average, N_signal: int=9):
		self.averages = [faster, slower]
		self.N_signal = N_signal
		self.N        = N_signal + faster.N + slower.N

	@validate
	def setHistoricalData(self, prices: list):

		assert(self.averages[0].depth == self.averages[1].depth)

		self.macds = []
		for index in range(self.averages[0].depth):
			self.macds.append(self.averages[0].history[index] - self.averages[1].history[index])

		self.signalLine = EMA(self.N_signal)
		self.signalLine.setHistoricalData(self.macds)
		self.__update__()

	@validate
	def update(self, price: float):
		self.signalLine.update(self.averages[0].current - self.averages[1].current)
		self.__update__()

	@validate
	def set_intermediate(self, price: float):
		self.signalLine.set_intermediate(self.averages[0].current - self.averages[1].current)
		self.__update__()

	def __update__(self):
		self.MACD   = self.averages[0].current - self.averages[1].current        # MACD
		self.SIGNAL = self.signalLine.current                                    # 9 period EMA signal of MACDs
		self.OSMA   = self.MACD - self.SIGNAL                                    # OSMA = MACD - 9 period EMA signal of MACDs

	def __eq__(self, other):
		if self.__class__ != other.__class__: return False
		return self.averages == other.averages and self.N_signal == other.N_signal

	def __repr__(self):
		name = 'MACD({},{}'.format(self.averages[0],self.averages[1])
		if self.N_signal != 9:
			name +=  ', N_signal={}'.format(self.N_signal)
		return name + ')'



class LSO(Indicator):
	"""
		Full Stochastic Oscillator

		Default is Lane's Stochastic Oscillator

	"""
	def __init__(self, N: int=20, N_k: int=3, N_d: int=3):
		self.N  = N
		self.N_k            = N_k
		self.N_d            = N_d


	@validate
	def setHistoricalData(self, bars: BarDataList):


		# Remove last element from bars
		bars = bars.copy()
		bars.pop()     

		# Technically we only need (N + N_k-1 + N_d-1) elements but for a nice code style we require one extra one
		assert(len(bars) >= self.N + self.N_k-1 + self.N_d-1 + 1) 

		self.k_buffer = []  # k buffer for k values
		self.d_buffer = []  # d buffer for d vlaues
		self.maxima   = []  # maxima buffer
		self.minima   = []  # minima buffer

		# bars[start] will be poped in the first iteration of the next while loop, before maxima and minima are calculated
		# bars[start] does not affect the final result, but it is carried along to simplifiy the code
		start = len(bars) - (self.N + self.N_k-1 + self.N_d-1) - 1

		# Fill maxima and minima buffer
		for bar in bars[ start : start+self.N ]:
			self.maxima.append(bar.high)
			self.minima.append(bar.low)

		assert(len(self.maxima)==self.N)
		assert(len(self.minima)==self.N)

		# Fill k_buffer of length N_k
		index = start+self.N
		while(index < len(bars) - (self.N_d-1) ):
			self.maxima.append(bars[index].high)
			self.minima.append(bars[index].low)
			self.maxima.pop(0)
			self.minima.pop(0)
			maximum = max(self.maxima)
			minimum = min(self.minima)

			self.k_buffer.append((bars[index].close - minimum) / (maximum - minimum + 1e-8) * 100)
			index += 1

		assert(len(self.k_buffer)==self.N_k)
		assert(len(self.maxima)==self.N)
		assert(len(self.minima)==self.N)

		self.d_buffer.append(np.mean(self.k_buffer))

		# Fill d_buffer of length N_d
		while(index < len(bars)):
			self.maxima.append(bars[index].high)
			self.minima.append(bars[index].low)
			self.maxima.pop(0)
			self.minima.pop(0)
			maximum = max(self.maxima)
			minimum = min(self.minima)

			self.k_buffer.append((bars[index].close - minimum) / (maximum - minimum + 1e-8) * 100)
			self.k_buffer.pop(0)

			self.d_buffer.append(np.mean(self.k_buffer))
			index += 1

		assert(len(self.k_buffer)==self.N_k)
		assert(len(self.d_buffer)==self.N_d)
		assert(len(self.maxima)==self.N)
		assert(len(self.minima)==self.N)

		self.current = self.k_buffer[-1]
		self.K = np.mean(self.k_buffer)
		self.D = np.mean(self.d_buffer)
		self.max = maximum
		self.min = minimum


	@validate
	def update(self, bar: BarData):

		# Update max and min with new value
		self.maxima.append(bar.high)
		self.minima.append(bar.low)
		self.maxima.pop(0)
		self.minima.pop(0)
		maximum = max(self.maxima)
		minimum = min(self.minima)

		# Update k_buffer  
		if maximum != minimum:                                                          # First IN First OUT (FIFO)
			self.k_buffer.append((bar.close - minimum) / (maximum - minimum + 1e-8) * 100)     # Append at the end
		else:
			self.k_buffer.append(50)
		self.k_buffer.pop(0)                                                            # Remove at the beginning

		# Update C and K
		self.current = self.k_buffer[-1]
		self.K = np.mean(self.k_buffer)

		self.d_buffer.append(self.K)
		self.d_buffer.pop(0)
		self.D = np.mean(self.d_buffer)

		self.max = maximum
		self.min = minimum

	@validate
	def set_intermediate(self, bar: BarData):

		# Determine temporary maxima and minima
		maximum = max([*self.maxima[1:], bar.high])
		minimum = min([*self.minima[1:], bar.low])

		# Update C and K
		if maximum != minimum:
			self.current = (bar.close - minimum) / (maximum - minimum) * 100
		else:
			self.current = 50
		self.K = np.mean([*self.k_buffer[1:], self.current])
		self.D = np.mean([*self.d_buffer[1:], self.K])

		self.max = maximum
		self.min = minimum

	def __eq__(self, other):
		if self.__class__ != other.__class__: return False
		return self.N == other.N and self.N_k == other.N_k and self.N_d == other.N_d

	def __repr__(self):
		name = 'LSO('
		previous = False
		if self.N != 20: 
			name += 'N={}'.format(self.N)
			previous = True
		if self.N_k != 3:
			if previous: name += ', '
			name += 'N_k={}'.format(self.N_k)
			previous = True
		if self.N_d != 3:
			if previous: name += ', '
			name += 'N_d={}'.format(self.N_d)
			previous = True
		return name + ')'


class tickHistogram:
	
	def __init__(self, bars: BarDataList, N: int=30):
		
		self.alpha = 2/(N+1)
		self.histogram = {}
		for bar in bars:
			self.update(bar)
	
	@validate
	def update(self, bar: BarData):
		
		ticks = []
		value = 0
		
		if bar.volume > 0:
			count = int(round( (bar.high - bar.low)/tickSize + 1))
			value = bar.volume / count
			for i in range(count):
				ticks.append(round(bar.low + i*tickSize, ndigits=1))

		for key in self.histogram.keys():
			if key in ticks:
				self.histogram[key] = self.alpha * value + (1-self.alpha) * self.histogram[key]
				ticks.remove(key)
			else:
				self.histogram[key] = (1-self.alpha) * self.histogram[key]

		for key in ticks:
			self.histogram[key] = value




@dataclass
class Extrema:
	price: float
	date: datetime
	kind: str
	
	def __le__(self, other):
		return self.price <= other.price


class ZigZag(Indicator):
	"""
		Zig Zag indicator
		- plots points on the chart whenever prices reverse by a percentage greater than a pre-chosen variable
	"""
	def __init__(self,minimumChange: float, depth: int=1):
		self.minimumChange = minimumChange
		self.N             = depth


	@validate
	def setHistoricalData(self, bars: BarDataList):

		# Remove last element from bars
		bars = bars.copy()
		bars.pop()

		self.last          = Extrema(bars[0].close,bars[0].date,'MAX')
		self.history       = []
		self.extremas      = []
		
		for bar in bars[1:]:
			self.update(bar, True)


	@validate
	def update(self, bar: BarData, verbose=True):
		
		# if(datetime(year=2020,month=5,day=13,hour=12,minute=52,second=0) >= bar.date):
		#   self.last = Extrema(bar.close,bar.date,'MAX')
		#   if(datetime(year=2020,month=5,day=13,hour=12,minute=52,second=0) == bar.date):
		#       print(bar.date,bar.close)
		#   return

		self.history.append(Extrema(bar.close, bar.date, 'UNKNOWN'))
		
		# Find potential next candidate
		if self.last.kind == 'MAX':
			# Downward trend
			candidate = np.min(np.array(self.history))
		elif(self.last.kind == 'MIN'):
			# Upward trend
			candidate = np.max(np.array(self.history))
		else:
			assert(False)
		
		# Confirm if the candidate meets the requirement
		if np.abs(candidate.price - self.history[-1].price) < self.minimumChange:
			return
		
		# Clear history buffer up to candidate
		while self.history[0].date <= candidate.date:
			self.history.pop(0)
		
		
		candidate.kind = 'MAX' if self.last.kind == 'MIN' else 'MIN'
		self.last      = candidate
		if verbose:
			#print(self.last.date, self.last.price, self.last.kind,'@',bar.date)
			self.extremas.append(self.last)

	@validate
	def set_intermediate(self, bar: BarData):
		pass


	def __eq__(self, other):
		if self.__class__ != other.__class__: return False
		return self.minimumChange == other.minimumChange and self.depth == other.depth


	def __repr__(self):
		name = 'ZigZag({}'.format(self.minimumChange)
		if self.N != 1: 
			name += ', depth={}'.format(self.N)
		return name + ')'




















