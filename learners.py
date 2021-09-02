import os
import logging
import abc
import collections
import threading
import time
import numpy as np
from utils import sigmoid
from environment import Environment
from agent import Agent
from networks import Network, DNN, LSTMNetwork, CNN
from visualizer import Visualizer

class ReinforcementLearner:
    __metaclass__=abc.ABCMeta
    lock = threading.Lock()

    def __init__(self, rl_method='rl', stock_code=None,
                chart_data=None, training_data=None,
                min_trading_unit=1, max_trading_unit=2, delayed_reward_threshold=.05, 
                net='dnn', num_steps=1, lr=0.001,
                value_network=None, policy_network=None,
                output_path='', reuse_models=True):

        assert min_trading_unit > 0
        assert max_trading_unit > 0
        assert max_trading_unit >= min_trading_unit
        assert num_steps
        assert lr > 0
        self.rl_method = rl_method
        self.stock_code = stock_code
        self.chart_data = chart_data
        self.environment = Environment(chart_data)
        self.agent = Agent(self.environment, 
                            min_trading_unit=min_trading_unit, 
                            max_trading_unit=max_trading_unit, 
                            delayed_reward_threshold=delayed_reward_threshold)
        
        self.training_data = training_data
        self.sample = None
        self.training_data_idx = -1
        self.num_features = self.agent.STATE_DIM
        if self.training_data is not None:
            self.num_features += self.training_data.shape[1]
        self.net = net
        self.num_steps = num_steps
        self.lr = lr
        self.value_network = value_network
        self.policy_network = policy_network
        self.reuse_models = reuse_models

        self.visualizer = Visualizer()

        self.memory_sample = []
        self.memory_action = []
        self.memory_reward = []
        self.memory_value = []
        self.memory_policy = []
        self.memory_pv = []
        self.memory_num_stocks = []
        self.memory_exp_idx = []
        self.memory_learning_idx = []

        self.loss = 0.
        self.itr_cnt = 0
        self.exploration_cnt = 0
        self.batch_size = 0
        self.learning_cnt = 0

        self.output_path = output_path

    def init_value_network(self, shared_network=None, activation='linear', loss='mse'):
        if self.net == 'dnn':
            self.value_network = DNN(input_dim=self.num_features, output_dim=self.agent.NUM_ACTIONS, 
                                    lr=self.lr, shared_network=shared_network, 
                                    activation=activation, loss=loss)

        elif self.net == 'lstm':
            self.value_network = LSTMNetwork(input_dim=self.num_features, output_dim=self.agent.NUM_ACTIONS, 
                                    lr=self.lr, num_steps=self.num_steps, shared_network=shared_network, 
                                    activation=activation, loss=loss)

        elif self.net == 'cnn':
            self.value_network = CNN(input_dim = self.num_features, output_dim=self.agent.NUM_ACTIONS,
                                    lr=self.lr, num_steps=self.num_steps, shared_network=shared_network,
                                    activation=activation, loss=loss)

        if self.reuse_models and os.path.exists(self.value_network_path):
            self.value_network.load_model(model_path=self.value_network_path)
        
    def init_policy_network(self, shared_network=None, activation='sigmoid', loss='mse'):
        if self.net == 'dnn':
            self.policy_network = DNN(input_dim=self.num_features, 
                                    output_dim=self.agent.NUM_ACTIONS, 
                                    lr=self.lr, shared_network=shared_network,
                                    activation=activation, loss=loss)

        elif self.net == 'lstm':
            self.policy_network = LSTMNetwork(input_dim=self.num_features, 
                                    output_dim=self.agent.NUM_ACTIONS, 
                                    lr=self.lr, num_steps=self.num_steps,
                                    shared_network=shared_network,
                                    activation=activation, loss=loss)
        
        elif self.net == 'cnn':
            self.policy_network = CNN(input_dim=self.num_features, 
                                    output_dim=self.agent.NUM_ACTIONS, 
                                    lr=self.lr, num_steps=self.num_steps,
                                    shared_network=shared_network,
                                    activation=activation, loss=loss)

        if self.reuse_models and os.path.exists(self.policy_network_path):
            self.value_network.load_model(model_path=self.policy_network_path)

    def reset(self):
        self.sample = None
        self.training_data_idx = -1
        self.environment.reset()
        self.agent.reset()
        self.visualizer.clear([0, len(self.chart_data)])
        self.memory_sample = []
        self.memory_action = []
        self.memory_reward = []
        self.memory_value = []
        self.memory_policy = []
        self.memory_pv = []
        self.memory_num_stocks = []
        self.memory_exp_idx = []
        self.memory_learning_idx = []

        self.loss = 0.
        self.itr_cnt = 0
        self.exploration_cnt = 0
        self.batch_size = 0
        self.learning_cnt = 0
    
    def build_sample(self):
        self.environment.observe()
        if len(self.training_data) > self.training_data_idx + 1:
            self.training_data_idx += 1
            self.sample = self.training_data.iloc[self.training_data_idx].tolist()
            self.sample.extend(self.agent.get_states())
            return self.sample

        return None

    @abc.abstractclassmethod
    def get_batch(self, batch_size, delayed_reward, discount_factor):
        pass

    def update_networks(self, batch_size, delayed_reward, discount_factor):
        x, y_value, y_policy = self.get_batch(batch_size, delayed_reward, discount_factor)

        if len(x) > 0:
            loss = 0
            if y_value is not None:
                loss += self.value_network.train_on_batch(x, y_value)
            if y_policy is not None:
                loss += self.policy_network.train_on_batch(x, y_policy)
            return loss
        return None
    
    def fit(self, delayed_reward, discount_factor):
        if self.batch_size > 0:
            _loss = self.update_networks(self.batch_size, delayed_reward, discount_factor)
            if _loss is not None:
                self.loss += abs(_loss)
                self.learning_cnt += 1
                self.memory_learning_idx.append(self.training_data_idx)
            self.batch_size = 0
    
    def visualize(self, epoch_str, num_epoches, epsilon):
        self.memory_action = [Agent.ACTION_HOLD] * (self.num_steps - 1) + self.memory_action
        self.memory_num_stocks = [0] * (self.num_steps - 1) + self.memory_num_stocks
        if self.value_network is not None:
            self.memory_value = [np.array([np.nan] * len(Agent.ACTIONS))] * (self.num_steps - 1) + self.memory_value
        if self.value_network is not None:
            self.memory_policy = [np.array([np.nan] * len(Agent.ACTIONS))] * (self.num_steps - 1) + self.memory_policy
        self.memory_pv = [self.agent.initial_balance] * (self.num_steps - 1) + self.memory_pv

        self.visualizer.plot(
            epoch_str=epoch_str, num_epoches=num_epoches,
            epsilon=epsilon, action_list=Agent.ACTIONS,
            actions=self.memory_action,
            num_stocks=self.memory_num_stocks,
            outvals_value=self.memory_value,
            outvals_policy=self.memory_policy,
            exps=self.memory_exp_idx,
            learning_idxes=self.memory_learning_idx,
            initial_balance=self.agent.initial_balance,
            pvs=self.memory_pv,
        )
        self.visualizer.save(os.path.join(self.epoch_summary_dir, 'epoch_summary_{}.png'.format(epoch_str)))

    def run(self, num_epoches=100, balance=1000000,
            discount_factor=0.9, start_epsilon=0.5, learning=True):
        info = "[{code}] RL:{rl} Net:{net} LR:{lr} " \
            "DF:{discount_factor} TU:[{min_trading_unit}," \
            "{max_trading_unit}] DRT:{delayed_reward_threshold}".format(
                code=self.stock_code, rl=self.rl_method, net=self.net,
                lr=self.lr, discount_factor=discount_factor,
                min_trading_unit=self.agent.min_trading_unit,
                max_trading_unit=self.agent.max_trading_unit,
                delayed_reward_threshold=self.agent.delayed_reward_threshold
            )
        with self.lock:
            logging.info(info)

        time_start = time.time()

        self.visualizer.prepare(self.environment.chart_data, info)

        self.epoch_summary_dir = os.path.join(self.output_path, 'epoch_summary_{}'.format(self.stock_code))
        if not os.path.isdir(self.epoch_summary_dir):
            os.makedirs(self.epoch_summary_dir)
        else:
            for f in os.listdir(self.epoch_summary_dir):
                os.remove(os.path.join(self.epoch_summary_dir, f))
        
        self.agent.set_balance(balance)

        max_portfolio_value = 0
        epoch_win_cnt = 0

        for epoch in range(num_epoches):
            time_start_epoch = time.time()

            q_sample=collections.deque(maxlen=self.num_steps)
        
            self.reset()

            if learning:
                epsilon = start_epsilon * (1. - float(epoch) / (num_epoches - 1))
                self.agent.reset_exploration()
            else:
                epsilon = start_epsilon

            while True:
                next_sample = self.build_sample()
                if next_sample is None:
                    break

                q_sample.append(next_sample)
                if len(q_sample) < self.num_steps:
                    continue
                
                pred_value = None
                pred_policy = None
                if self.value_network is not None:
                    pred_value = self.value_network.predict(list(q_sample))
                if self.policy_network is not None:
                    pred_policy = self.policy_network.predict(list(q_sample))
                
                action , confidence, exploration = self.agent.decide_action(pred_value, pred_policy, epsilon)

                immediate_reward, delayed_reward = self.agent.act(action, confidence)

                self.memory_sample.append(list(q_sample))
                self.memory_action.append(action)
                self.memory_reward.append(immediate_reward)
                if self.value_network is not None:
                    self.memory_value.append(pred_value)
                if self.policy_network is not None:
                    self.memory_policy.append(pred_policy)
                self.memory_pv.append(self.agent.portfolio_value)
                self.memory_num_stocks.append(self.agent.num_stocks)
                if exploration:
                    self.memory_exp_idx.append(self.training_data_idx)
                
                self.batch_size += 1
                self.itr_cnt += 1
                self.exploration_cnt += 1 if exploration else 0

                if learning and (delayed_reward != 0):
                    self.fit(delayed_reward, discount_factor)
                
            if learning:
                self.fit(self.agent.profitloss, discount_factor)

            num_epoches_digit = len(str(num_epoches))
            epoch_str = str(epoch + 1).rjust(num_epoches_digit, '0')
            time_end_epoch = time.time()
            elapsed_time_epoch = time_end_epoch - time_start_epoch
            if self.learning_cnt > 0:
                self.loss /= self.learning_cnt
            logging.info("[{}][Epoch {}/{}] Epsilon:{:.4f} "
                "#Expl.:{}/{} #Buy:{} #Sell:{} #Hold:{} "
                "#Stocks:{} PV:{:,.0f} "
                "LC:{} Loss:{:.6f} ET:{:.4f}".format(
                    self.stock_code, epoch_str, num_epoches, epsilon,
                    self.exploration_cnt, self.itr_cnt,
                    self.agent.num_buy, self.agent.num_sell,
                    self.agent.num_hold, self.agent.num_stocks,
                    self.agent.portfolio_value, self.learning_cnt,
                    self.loss, elapsed_time_epoch))
            
            self.visualize(epoch_str, num_epoches, epsilon)

            max_portfolio_value = max(max_portfolio_value, self.agent.portfolio_value)
            if self.agent.portfolio_value > self.agent.initial_balance:
                epoch_win_cnt += 1
            
            time_end = time.time()
            elapsed_time = time_end - time_start

            with self.lock:
                logging.info("[{code}] Elapsed Time:{elapsed_time:.4f} "
                    "Max PV:{max_pv:,.0f} #Win:{cnt_win}".format(
                    code=self.stock_code, elapsed_time=elapsed_time,
                    max_pv=max_portfolio_value, cnt_win=epoch_win_cnt))

    def save_models(self):
        if self.value_network is not None and self.value_network_path is not None:
            self.value_network.save_model(self.value_network_path)
        if self.policy_network is not None and self.policy_network_path is not None:
            self.policy_network.save_model(self.policy_network_path)

class DQNLearner(ReinforcementLearner):
    def __init__(self, *args, value_network_path=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.value_network_path = value_network_path
        self.init_value_network()
    
    def get_batch(self, batch_size, delayed_reward, discount_factor):
        memory = zip(
            reversed(self.memory_sample[-batch_size:]),
            reversed(self.memory_action[-batch_size:]),
            reversed(self.memory_value[-batch_size:]),
            reversed(self.memory_reward[-batch_size:]),
        )
        x = np.zeros((batch_size, self.num_steps, self.num_features))
        y_value = np.zeros((batch_size, self.agent.NUM_ACTIONS))
        value_max_next = 0
        reward_next = self.memory_reward[-1]
        for i, (sample, action, value, reward) in enumerate(memory):
            x[i] = sample
            y_value[i] = value
            r = (delayed_reward + reward_next - reward * 2) * 100
            y_value[i, action] = r + discount_factor * value_max_next
            value_max_next = value.max()
            reward_next = reward
        return x, y_value, None

class PolicyGradientLearner(ReinforcementLearner):
    def __init__(self, *args, policy_network_path=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.policy_network_path = policy_network_path
        self.init_value_network()
    
    def get_batch(self, batch_size, delayed_reward, discount_factor):
        memory = zip(
            reversed(self.memory_sample[-batch_size:]),
            reversed(self.memory_action[-batch_size:]),
            reversed(self.memory_policy[-batch_size:]),
            reversed(self.memory_reward[-batch_size:]),
        )
        x = np.zeros((batch_size, self.num_steps, self.num_features))
        y_policy = np.full((batch_size, self.agent.NUM_ACTIONS), .5)
        reward_next = self.memory_reward[-1]
        for i, (sample, action, policy, reward) in enumerate(memory):
            x[i] = sample
            y_policy[i] = policy
            r = (delayed_reward + reward_next - reward * 2) * 100
            y_policy[i, action] = sigmoid(r)
            reward_next = reward
        return x, None, y_policy

class ActorCriticLearner(ReinforcementLearner):
    def __init__(self, *args, shared_network=None, value_network_path=None, policy_network_path=None, **kwargs):
        super().__init__(*args, **kwargs)
        if shared_network is None:
            self.shared_network = Network.get_shared_network(net=self.net, num_steps=self.num_steps, input_dim=self.num_features)
        else:
            self.shared_network = shared_network
        self.value_network_path = value_network_path
        self.policy_network_path = policy_network_path
        if self.value_network is None:
            self.init_value_network(shared_network=shared_network)
        if self.policy_network is None:
            self.init_policy_network(shared_network=shared_network)
    
    def get_batch(self, batch_size, delayed_reward, discount_factor):
        memory = zip(
            reversed(self.memory_sample[-batch_size:]),
            reversed(self.memory_action[-batch_size:]),
            reversed(self.memory_value[-batch_size:]),
            reversed(self.memory_policy[-batch_size:]),
            reversed(self.memory_reward[-batch_size:]),
        )
        x = np.zeros((batch_size, self.num_steps, self.num_features))
        y_value = np.zeros((batch_size, self.agent.NUM_ACTIONS))
        y_policy = np.full((batch_size, self.agent.NUM_ACTIONS), .5)
        value_max_next = 0
        reward_next = self.memory_reward[-1]
        for i, (sample, action, value, policy, reward) in enumerate(memory):
            x[i] = sample
            y_value[i] = value
            y_policy[i] = policy
            r = (delayed_reward + reward_next - reward * 2) * 100
            y_value[i, action] = r + discount_factor * value_max_next
            y_policy[i, action] = sigmoid(value[action])
            value_max_next = value.max()
            reward_next = reward
        return x, y_value, y_policy

class A2CLearner(ActorCriticLearner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def get_batch(self, batch_size, delayed_reward, discount_factor):
        memory = zip(
            reversed(self.memory_sample[-batch_size:]),
            reversed(self.memory_action[-batch_size:]),
            reversed(self.memory_value[-batch_size:]),
            reversed(self.memory_policy[-batch_size:]),
            reversed(self.memory_reward[-batch_size:]),
        )
        x = np.zeros((batch_size, self.num_steps, self.num_features))
        y_value = np.zeros((batch_size, self.agent.NUM_ACTIONS))
        y_policy = np.full((batch_size, self.agent.NUM_ACTIONS), .5)
        value_max_next = 0
        reward_next = self.memory_reward[-1]
        for i, (sample, action, value, policy, reward) in enumerate(memory):
            x[i] = sample
            y_value[i] = value
            y_policy[i] = policy
            r = (delayed_reward + reward_next - reward * 2) * 100
            y_value[i, action] = r + discount_factor * value_max_next
            advantage = value[action] - value.mean()
            y_policy[i, action] = sigmoid(advantage)
            value_max_next = value.max()
            reward_next = reward
        return x, y_value, y_policy

class A3CLearner(ReinforcementLearner):
    def __init__(self, *args, list_stock_code=None,
        list_chart_data=None, list_training_data=None,
        list_min_trading_unit=None, list_max_trading_unit=None,
        value_network_path=None, policy_network_path=None,
        **kwars):
        assert len(list_training_data) > 0
        super().__init__(*args, **kwargs)
        self.num_features += list_training_data[0].shape[1]

        self.shared_network = Nextwork.get_shared_network(
            net=self.net, num_steps=self.num_steps,
            input_dim=self.num_features)
        self.value_network_path = value_network_path
        self.policy_network_path = policy_network_path
        if self.value_network is None:
            self.init_value_network(shared_network=self.shared_network)
        if self.policy_network is None:
            self.init_policy_network(shared_network=self.shared_network)

        self.learners = []
        for (stock_code, chart_data, training_data,
            min_trading_unit, max_trading_unit) in zip(
                list_stock_code, list_chart_data, list_training_data,
                list_min_trading_unit, list_max_trading_unit
            ):
            learner = A2CLearner(*args,
                stock_code=stock_code, chart_data=chart_data,
                training_data=training_data,
                min_trading_unit=min_trading_unit,
                max_trading_unit=max_trading_unit,
                shared_network=self.shared_network,
                value_network=self.value_network,
                policy_network=self.policy_network, **kwargs)
            self.learners.append(learner)
    
    def run(
        self, num_epoches=100, balance=1000000,
        discount_factor=0.9, start_epsilon=0.9, learning=True):
        threads = []
        for learner in self.learners:
            threads.append(threading.Thread(
                target=learner.fit, daemon=True, kwargs={
                'num_epoches': num_epoches, 'balance': balance,
                'discount_factor': discount_factor,
                'start_epsilon': start_epsilon,
                'learning': learning
            }))
        for thread in threads:
            thread.start()
            time.sleep(1)
        for thread in threads: thread.join()