import threading
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')

from mplfinance.original_flavor import candlestick_ohlc
from agent import Agent

lock = threading.Lock()

class Visualizer:
    COLORS = ['r', 'b', 'g']

    def __init__(self, vnet=False):
        self.canvas = None
        self.fig = None
        self.axes = None
        self.title = ''

    def prepare(self, chart_data, title):
        self.title = title
        with lock:
            self.fig, self.axes = plt.subplots(nrows= 5, ncols=1, facecolor='w', sharex=True)
            for ax in self.axes:
                ax.get_xaxis().get_major_formatter().set_scientific(False)
                ax.get_yaxis().get_major_formatter().set_scientific(False)
                ax.yaxis.tick_right()

            self.axes[0].set_ylabel('Env.')
            x = np.arange(len(chart_data))
            ohlc = np.hstack((x.reshape(-1, 1), np.array(chart_data)[:, 1:-1]))
            candlestick_ohlc(self.axes[0], ohlc, colorup='r', colordown='b')
            ax = self.axes[0].twinx()
            volume = np.array(chart_data)[:, -1].tolist()
            ax.bar(x, volume, color='b', alpha=0.3)
    
    def plot(self, epoch_str=None, num_epoches=None, epsilon=None, 
    action_list=None, actions=None, num_stocks=None, 
    outvals_value=[], outvals_policy=[], exps=None, 
    learning_idxes=None, initial_balance=None, pvs=None):
        with lock:
            x = np.arange(len(actions))
            actions = np.array(actions)
            outvals_value = np.array(outvals_value)
            outvals_policy = np.array(outvals_policy)
            pvs_base = np.zeros(len(actions)) + initial_balance

            for action, color in zip(action_list, self.COLORS):
                for i in x[actions == action]:
                    self.axes[1].axvline(i, color=color, alpha=0.1)
            self.axes[1].plot(x, num_stocks, '-k')

            if len(outvals_value) > 0:
                max_actions = np.argmax(outvals_value, axis=1)
                for action, color in zip(action_list, self.COLORS):
                    for idx in x:
                        if max_actions[idx] == action:
                            self.axes[2].axvline(idx, color=color, alpha=0.1)
                    self.axes[2].plot(x, outvals_value[:, action], color=color, linestyle='-')
            
            for exp_idx in exps:
                self.axes[3].axvline(exp_idx, color='y')
            
            _outvals = outvals_policy if len(outvals_policy) > 0 else outvals_value
            
            for idx, outval in zip(x, _outvals):
                color = 'white'
                if np.isnan(outval.max()):
                    continue
                if outval.argmax() == Agent.ACTION_BUY:
                    color = 'r'
                elif outval.argmax() == Agent.ACTION_SELL:
                    color = 'b'
                self.axes[3].axvline(idx, color=color, alpha=0.1)

            if len(outvals_policy) > 0:
                for action, color in zip(action_list, self.COLORS):
                    self.axes[3].plot(x, outvals_policy[:, action], color=color, linestyle='-')
            
            self.axes[4].axhline(initial_balance, linestyle='-', color='gray')
            self.axes[4].fill_between(x, pvs, pvs_base, where=pvs > pvs_base, facecolor='r', alpha=0.1)
            self.axes[4].fill_between(x, pvs, pvs_base, where=pvs < pvs_base, facecolor='b', alpha=0.1)
            self.axes[4].plot(x, pvs, '-k')
            
            for learning_idx in learning_idxes:
                self.axes[4].axvline(learning_idx, color='y')

            self.fig.suptitle('{} \nEpoch:{}/{} e={:.2f}'.format(self.title, epoch_str, num_epoches, epsilon))
            self.fig.tight_layout()
            self.fig.subplots_adjust(top=0.85)
    
    def clear(self, xlim):
        with lock:
            _axes = self.axes.tolist()
            for ax in _axes[1:]:
                ax.cla()
                ax.relim()
                ax.autoscale()
            
            self.axes[1].set_ylabel('Agent')
            self.axes[2].set_ylabel('V')
            self.axes[3].set_ylabel('P')
            self.axes[4].set_ylabel('PV')
            for ax in _axes:
                ax.set_xlim(xlim)
                ax.get_xaxis().get_major_formatter().set_scientific(False)
                ax.get_yaxis().get_major_formatter().set_scientific(False)
                ax.ticklabel_format(useOffset=False)
    
    def save(self, path):
        with lock:
            self.fig.savefig(path)