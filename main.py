import os
import sys
import logging
import argparse
import json

import settings
import utils
import data_manager

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stock_code', nargs='+')
    parser.add_argument('--ver', choices=['v1', 'v2'], default='v1')
    parser.add_argument('--rl_method', choices=['dqn', 'pg', 'ac', 'a2c', 'a3c'])
    parser.add_argument('--net', choices=['dnn', 'lstm', 'cnn'], default='dnn')
    parser.add_argument('--num_steps', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--discount_factor', type=float, default=0)
    parser.add_argument('--start_epsilon', type=float, default=0)
    parser.add_argument('--balance', type=int, default=1000000)
    parser.add_argument('--num_epoches', type=int, default=100)
    parser.add_argument('--delayed_reward_threshold', type=float, default=0.05)
    parser.add_argument('--backend', choices=['tensroflow', 'pladiml'], default='tensorflow')
    parser.add_argument('--output_name', default=utils.get_time_str())
    parser.add_argument('--value_network_name')
    parser.add_argument('--policy_network_name')
    parser.add_argument('--reuse_models', action='store_true')
    parser.add_argument('--learning', action='store_true')
    parser.add_argument('--start_date', default='20180101')
    parser.add_argument('--end_date', default='20181231')
    args = parser.parse_args()

    if args.backend == 'tensorflow':
        os.environ['KERAS_BACKEND'] = 'tensorflow'
    elif args.backend == 'plaidml':
        os.environ['KERAS_BACKEND'] = 'plaidml.kers.backend'

    output_path = os.path.join(settings.BASE_DIR, 'output/{}_{}_{}'.format(args.output_name, args.rl_method, args.net))
    if not os.path.isdir(output_path):
        os.makedirs(output_path)
    
    with open(os.path.join(output_path, 'params.json'), 'w') as f:
        f.write(json.dumps(vars(args)))
    
    file_handler = logging.FileHandler(filename=os.path.join(output_path, "{}.log".format(args.output_name)), encoding='utf-8')
    stream_handler = logging.StreamHandler(sys.stdout)
    file_handler.setLevel(logging.DEBUG)
    stream_handler.setLevel(logging.INFO)
    logging.basicConfig(format="%(message)s", handlers=[file_handler, stream_handler], level=logging.DEBUG)

    from agent import Agent
    from learners import DQNLearner, PolicyGradientLearner, ActorCriticLearner, A2CLearner, A3CLearner

    value_network_path = ''
    policy_network_path = ''
    if args.value_network_name is not None:
        value_network_path = os.path.join(settings.BASE_DIR, 'models/{}.h5'.format(args.value_network_name))
    else:
        value_network_path = os.path.join(output_path, '{}_{}_value_{}.h5'.format(args.rl_method, args.net, args.output_name))
    if args.policy_network_name is not None:
        policy_network_path = os.path.join(settings.BASE_DIR, 'models/{}.h5'.format(args.policy_network_name))
    else:
        value_network_path = os.path.join(output_path, '{}_{}_policy_{}.h5'.format(args.rl_method, args.net, args.output_name))
    
    common_params = {}
    list_stock_code = []
    list_chart_data = []
    list_training_data = []
    list_min_trading_unit = []
    list_max_trading_unit = []
    
    for stock_code in args.stock_code:
        chart_data, training_data = data_manager.load_data(
            os.path.join(settings.BASE_DIR,
            'data/{}/{}.csv'.format(args.ver, stock_code)),
            args.start_date, args.end_date, ver=args.ver)
        
        min_trading_unit = max(int(100000 / chart_data.iloc[-1]['close']), 1)
        max_trading_unit = max(int(1000000 / chart_data.iloc[-1]['close']), 1)
        
        common_params = {'rl_method': args.rl_method,
            'delayed_reward_threshold': args.delayed_reward_threshold,
            'net': args.net, 'num_steps': args.num_steps, 'lr': args.lr,
            'output_path': output_path, 'reuse_models': args.reuse_models}
        
        learner = None
        if args.rl_method != 'a3c':
            common_params.update({'stock_code': stock_code, 'chart_data': chart_data, 
                                'training_data': training_data, 'min_trading_unit': min_trading_unit, 'max_trading_unit': max_trading_unit})
            if args.rl_method == 'dqn':
                learner = DQNLearner(**{**common_params, 'value_network_path': value_network_path})
            elif args.rl_method == 'pg':
                learner = PolicyGradientLearner(**{**common_params, 'policy_network_path': policy_network_path})
            elif args.rl_method == 'ac':
                learner = ActorCriticLearner(**{**common_params, 'value_network_path': value_network_path, 'policy_network_path': policy_network_path})
            elif args.rl_method == 'a2c':
                learner = A2CLearner(**{**common_params, 'value_network_path': value_network_path, 'policy_network_path': policy_network_path})
            if learner is not None:
                learner.run(balance=args.balance,
                    num_epoches=args.num_epoches,
                    discount_factor=args.discount_factor,
                    start_epsilon=args.start_epsilon,
                    learning=args.learning)
                learner.save_models()
        else:
            list_stock_code.append(stock_code)
            list_chart_data.append(chart_data)
            list_training_data.append(training_data)
            list_min_trading_unit.append(min_trading_unit)
            list_max_trading_unit.append(max_trading_unit)
        
    if args.rl_method == 'a3c':
        learner = A2CLearner(**{**common_params, 
        'list_stock_code': list_stock_code, 'list_chart_data': list_chart_data,
        'list_training_data': list_training_data, 'list_min_trading_unit': list_min_trading_unit,
        'list_max_trading_unit': list_max_trading_unit, 'value_network_path': value_network_path, 'policy_network_path': policy_network_path})

        learner.run(balance=args.balance,
                num_epoches=args.num_epoches,
                discount_factor=args.discount_factor,
                start_epsilon=args.start_epsilon,
                learning=args.learning)
        learner.save_models()