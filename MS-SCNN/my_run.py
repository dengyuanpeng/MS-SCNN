import argparse
import os
import torch
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
import random
import numpy as np
import ast


def parse_list(string):
    """Parse string representation of list"""
    try:
        return ast.literal_eval(string)
    except:
        return string


if __name__ == '__main__':
    fix_seed = 20

    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)
    parser = argparse.ArgumentParser(description='TimesNet')

    # basic config
    parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='Autoformer',
                        help='model name, options: [Autoformer, Transformer, TimesNet]')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--lookback_len', type=int, default=1, help='Lookback sequence length')
    parser.add_argument('--ext', type=int, default=1, help='enable extrapolation')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

    # inputation task
    parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')

    # anomaly detection task
    parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (%)')

    # model define
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--short_period_len', type=int, default=8, help='Short period length')
    parser.add_argument('--cycle_len', type=int, default=8, help='Cycle length')
    parser.add_argument('--kernel_size', type=int, default=2, help='Kernel size')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=8, help='dimension of model')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--long_term', type=int, default=1, help='enable long-term component')
    parser.add_argument('--seasonal', type=int, default=1, help='enable seasonal component')
    parser.add_argument('--short_term', type=int, default=1, help='enable short-term component')
    parser.add_argument('--spatial', type=int, default=1, help='enable spatial component')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')

    # MS-SCNN specific hyperparameters
    parser.add_argument('--reduction_ratio', type=int, default=4,
                        help='reduction ratio for channel attention, options: [2, 4, 8]')
    parser.add_argument('--season_dilations', type=str, default='[1,4,7]',
                        help='dilation rates for seasonal component, options: [[1,3,5], [1,4,7], [2,5,8]]')
    parser.add_argument('--long_dilations', type=str, default='[2,5,8,11]',
                        help='dilation rates for long-term component, options: [[1,3,5,7], [1,4,7,10], [2,5,8,11]]')
    parser.add_argument('--short_dilations', type=str, default='[1,2]',
                        help='dilation rates for short-term component, options: [[1,2], [1,3], [2,4]]')
    parser.add_argument('--long_kernel_size', type=int, default=5,
                        help='kernel size for long-term component, options: [5, 6, 7]')
    parser.add_argument('--season_kernel_size', type=int, default=3,
                        help='kernel size for seasonal component, options: [3, 4, 5]')
    parser.add_argument('--short_kernel_size', type=int, default=2,
                        help='kernel size for short-term component, options: [2, 3]')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epoch', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use   gpus', default=False)
    parser.add_argument('--devices', type=str, default='2,3', help='device ids of multile gpus')

    # de-stationary projector params
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

    args = parser.parse_args()

    # Parse list arguments
    args.season_dilations = parse_list(args.season_dilations)
    args.long_dilations = parse_list(args.long_dilations)
    args.short_dilations = parse_list(args.short_dilations)

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print(args)

    if args.task_name == 'long_term_forecast':
        Exp = Exp_Long_Term_Forecast
    elif args.task_name == 'short_term_forecast':
        Exp = Exp_Short_Term_Forecast
    elif args.task_name == 'imputation':
        Exp = Exp_Imputation
    elif args.task_name == 'anomaly_detection':
        Exp = Exp_Anomaly_Detection
    elif args.task_name == 'classification':
        Exp = Exp_Classification
    else:
        Exp = Exp_Long_Term_Forecast

    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_ext{}_bl{}_dm{}_el{}_dl{}_fc{}_eb{}_dt{}_{}_lt{}_se{}_st{}_si{}_rr{}_sd{}_ld{}_sd{}_lk{}_sk{}_shk{}_{}'.format(
                args.task_name,
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.ext,
                args.lookback_len,
                args.d_model,
                args.e_layers,
                args.d_layers,
                args.factor,
                args.embed,
                args.distil,
                args.des,
                args.long_term,
                args.seasonal,
                args.short_term,
                args.spatial,
                args.reduction_ratio,
                str(args.season_dilations).replace(' ', ''),
                str(args.long_dilations).replace(' ', ''),
                str(args.short_dilations).replace(' ', ''),
                args.long_kernel_size,
                args.season_kernel_size,
                args.short_kernel_size,
                args.dropout, ii, )

            exp = Exp(args)  # set experiments
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)
            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_ext{}_bl{}_dm{}_el{}_dl{}_fc{}_eb{}_dt{}_{}_lt{}_se{}_st{}_si{}_rr{}_sd{}_ld{}_sd{}_lk{}_sk{}_shk{}_{}'.format(
            args.task_name,
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.ext,
            args.lookback_len,
            args.d_model,
            args.e_layers,
            args.d_layers,
            args.factor,
            args.embed,
            args.distil,
            args.des,
            args.long_term,
            args.seasonal,
            args.short_term,
            args.spatial,
            args.reduction_ratio,
            str(args.season_dilations).replace(' ', ''),
            str(args.long_dilations).replace(' ', ''),
            str(args.short_dilations).replace(' ', ''),
            args.long_kernel_size,
            args.season_kernel_size,
            args.short_kernel_size, ii, )

        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()