import argparse
import datetime as dte
import os

import data_formatters.base
import expt_settings.configs
import libs.hyperparam_opt
import libs.tft_model
import libs.utils as utils
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import tensorflow as tf

import warnings
warnings.simplefilter(action='ignore')

ExperimentConfig = expt_settings.configs.ExperimentConfig
HyperparamOptManager = libs.hyperparam_opt.HyperparamOptManager
ModelClass = libs.tft_model.TemporalFusionTransformer

def main(expt_name, use_gpu, restart_opt, model_folder, hyperparam_iterations,
         data_csv_path, data_formatter):
    default_keras_session = tf.compat.v1.keras.backend.get_session()

    if use_gpu:
        tf_config = utils.get_default_tensorflow_config(tf_device="gpu", gpu_id=2)
    else:
        tf_config = utils.get_default_tensorflow_config(tf_device="cpu")

    print("Loading & splitting data")
    train, valid, test = data_formatter.split_data(data_csv_path)
    train_samples, valid_samples = data_formatter.get_num_samples_for_calibration()

    fixed_params = data_formatter.get_experiment_params()
    param_ranges = ModelClass.get_hyperparm_choices()
    fixed_params['model_folder'] = model_folder

    print("Loading hyperparam manager")
    opt_manager = HyperparamOptManager(param_ranges, fixed_params, model_folder)

    success = opt_manager.load_results()
    if success and not restart_opt:
        print("Loaded results from previous training")
    else:
        print("Creating new hyperparameter optimization")
        opt_manager.clear()

    tf.compat.v1.get_default_graph()
    with tf.Graph().as_default(), tf.compat.v1.Session(config=tf_config) as sess:
        tf.compat.v1.keras.backend.set_session(sess)
        tf.compat.v1.experimental.output_all_intermediates(True)
        params = opt_manager.get_next_parameters()
        # Create a TFT model
        model = ModelClass(params, use_cudnn=True)

        if not model.training_data_cached():
            model.cache_batched_data(train, "train", num_samples=train_samples)
            model.cache_batched_data(valid, "valid", num_samples=valid_samples)

        # Train and save model
        model.fit()

        val_loss = model.evaluate()
        if np.allclose(val_loss, 0.) or np.isnan(val_loss):
            # Set all invalid losses to infintiy.
            # N.b. val_loss only becomes 0. when the weights are nan.
            print("Skipping bad configuration....")
            val_loss = np.inf
        opt_manager.update_score(params, val_loss, model)
        tf.compat.v1.keras.backend.set_session(sess)
        model.save(model_folder)

    print("*** Running tests ***")
    tf.compat.v1.reset_default_graph()
    with tf.Graph().as_default(), tf.compat.v1.Session(config=tf_config) as sess:
        tf.compat.v1.keras.backend.set_session(sess)
        best_params = opt_manager.get_best_params()
        model = ModelClass(best_params, use_cudnn=use_gpu)

        model.load(opt_manager.hyperparam_folder)

        print("Computing best validation loss")
        val_loss = model.evaluate(valid)

        print("Computing test loss")
        output_map = model.predict(test, return_targets=True)
        targets = data_formatter.format_predictions(output_map["targets"])
        p50_forecast = data_formatter.format_predictions(output_map["p50"])
        p90_forecast = data_formatter.format_predictions(output_map["p90"])

        def extract_numerical_data(data):
            """Strips out forecast time and identifier columns."""
            return data[[
                col for col in data.columns
                if col not in {"forecast_time", "identifier"}
            ]]

        p50_loss = utils.numpy_normalised_quantile_loss(
            extract_numerical_data(targets), extract_numerical_data(p50_forecast),
            0.5)
        p90_loss = utils.numpy_normalised_quantile_loss(
            extract_numerical_data(targets), extract_numerical_data(p90_forecast),
            0.9)
        # tf.keras.backend.set_session(default_keras_session)

    print("Hyperparam optimisation completed @ {}".format(dte.datetime.now()))
    print("Best validation loss = {}".format(val_loss))
    print("Params:")

    for k in best_params:
        print(k, " = ", best_params[k])
    print()
    print("Normalised Quantile Loss for Test Data: P50={}, P90={}".format(
        p50_loss.mean(), p90_loss.mean()))

    plot_range = {
        'notch': [600, 1600],
        'dlr': [600, 1600]

    }
    plot_ticks = {
        'notch': [np.arange(0, 1001, 200), np.arange(0, 1.1, 0.5)]
    }

    plt.figure(figsize=(20, 10))
    plt.plot(output_map['targets']['t+0'][plot_range[expt_name][0]:plot_range[expt_name][1]], label='target', color='#E31A1C', linewidth=4)
    plt.plot(output_map['p10']['t+0'][plot_range[expt_name][0]:plot_range[expt_name][1]], label='p10', color='#009ADE', linestyple='--', linewidth=4)
    plt.plot(output_map['p50']['t+0'][plot_range[expt_name][0]:plot_range[expt_name][1]], label='p50', color='#009ADE', linestyple='--', linewidth=4)
    plt.plot(output_map['p90']['t+0'][plot_range[expt_name][0]:plot_range[expt_name][1]], label='p90', c='#AF588A', linestyple='--', linewidth=4)
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper right', fontsize=20, ncol=4)
    plt.xticks(plot_ticks[expt_name][0], fontsize=20)
    plt.yticks(plot_ticks[expt_name][1], fontsize=20)
    plt.xlabel('Time Step', fontsize=30)
    plt.ylabel('Value', fontsize=30)
    plt.savefig(model_folder + f'{expt_name}.png')

if __name__=="__main__":
    def get_args():
        """Returns settings from command line."""
        experiment_names = ExperimentConfig.default_experiments
        parser = argparse.ArgumentParser(description="Data download configs")
        parser.add_argument(
            "expt_name",
            metavar="e",
            type=str,
            nargs="?",
            default="mobiact",
            choices=experiment_names,
            help="Experiment Name. Default={}".format(",".join(experiment_names)))
        parser.add_argument(
            "output_folder",
            metavar="f",
            type=str,
            nargs="?",
            default=".",
            help="Path to folder for data download")
        parser.add_argument(
            "use_gpu",
            metavar="g",
            type=str,
            nargs="?",
            choices=["yes", "no"],
            default="yes",
            help="Whether to use gpu for training.")
        parser.add_argument(
            "restart_hyperparam_opt",
            metavar="o",
            type=str,
            nargs="?",
            choices=["yes", "no"],
            default="no",
            help="Whether to re-run hyperparameter optimisation from scratch.")
        args = parser.parse_known_args()[0]

        root_folder = None if args.output_folder == "." else args.output_folder

        return args.expt_name, root_folder, args.use_gpu == "yes", args.restart_hyperparam_opt


    name, folder, use_tensorflow_with_gpu, restart = get_args()
    print("Using output folder {}".format(folder))

    config = ExperimentConfig(name, folder)
    formatter = config.make_data_formatter()
    main(
          expt_name=name,
          use_gpu=use_tensorflow_with_gpu,
          restart_opt=restart,
          model_folder=os.path.join(config.model_folder, "main"),
          hyperparam_iterations=config.hyperparam_iterations,
          data_csv_path=config.data_csv_path,
          data_formatter=formatter)
