import logging
import sys
import os
import traceback
import json
from datetime import datetime
import string
import random
from collections import OrderedDict
import time
import pickle
from functools import partial
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
import pandas as pd
from utils import utils
import torch
from torch.utils.data import DataLoader
import numpy as np
from models.loss import l2_reg_loss
from datasets.dataset import ImputationDataset, TransductionDataset, ClassiregressionDataset, collate_unsuperv, collate_superv


logger = logging.getLogger('__main__')

NEG_METRICS = {'loss'}  # metrics for which "better" is less

val_times = {"total_time": 0, "count": 0}


def pipeline_factory(config):
    """For the task specified in the configuration returns the corresponding combination of
    Dataset class, collate function and Runner class."""

    task = config['task']

    if task == "imputation":
        return partial(ImputationDataset, mean_mask_length=config['mean_mask_length'],
                       masking_ratio=config['masking_ratio'], mode=config['mask_mode'],
                       distribution=config['mask_distribution'], exclude_feats=config['exclude_feats']),\
            collate_unsuperv, UnsupervisedRunner
    if task == "transduction":
        return partial(TransductionDataset, mask_feats=config['mask_feats'],
                       start_hint=config['start_hint'], end_hint=config['end_hint']), collate_unsuperv, UnsupervisedRunner
    if (task == "classification") or (task == "regression"):
        return ClassiregressionDataset, collate_superv, SupervisedRunner
    else:
        raise NotImplementedError("Task '{}' not implemented".format(task))


def setup(args):
    """Prepare training session: read configuration from file (takes precedence), create directories.
    Input:
        args: arguments object from argparse
    Returns:
        config: configuration dictionary
    """

    config = args.__dict__  # configuration dictionary

    if args.config_filepath is not None:
        logger.info("Reading configuration ...")
        try:  # dictionary containing the entire configuration settings in a hierarchical fashion
            config.update(utils.load_config(args.config_filepath))
        except:
            logger.critical("Failed to load configuration file. Check JSON syntax and verify that files exist")
            traceback.print_exc()
            sys.exit(1)

    # Create output directory
    initial_timestamp = datetime.now()
    output_dir = config['output_dir']
    if not os.path.isdir(output_dir):
        raise IOError(
            "Root directory '{}', where the directory of the experiment will be created, must exist".format(output_dir))

    output_dir = os.path.join(output_dir, config['experiment_name'])

    formatted_timestamp = initial_timestamp.strftime("%Y-%m-%d_%H-%M-%S")
    config['initial_timestamp'] = formatted_timestamp
    if (not config['no_timestamp']) or (len(config['experiment_name']) == 0):
        rand_suffix = "".join(random.choices(string.ascii_letters + string.digits, k=3))
        output_dir += "_" + formatted_timestamp + "_" + rand_suffix
    config['output_dir'] = output_dir
    config['save_dir'] = os.path.join(output_dir, 'checkpoints')
    config['pred_dir'] = os.path.join(output_dir, 'predictions')
    config['tensorboard_dir'] = os.path.join(output_dir, 'tb_summaries')
    utils.create_dirs([config['save_dir'], config['pred_dir'], config['tensorboard_dir']])

    # Save configuration as a (pretty) json file
    with open(os.path.join(output_dir, 'configuration.json'), 'w') as fp:
        json.dump(config, fp, indent=4, sort_keys=True)

    logger.info("Stored configuration file in '{}'".format(output_dir))

    return config


def fold_evaluate(dataset, model, device, loss_module, target_feats, config, dataset_name):

    allfolds = {'target_feats': target_feats,  # list of len(num_folds), each element: list of target feature integer indices
                'predictions': [],  # list of len(num_folds), each element: (num_samples, seq_len, feat_dim) prediction per sample
                'targets': [],  # list of len(num_folds), each element: (num_samples, seq_len, feat_dim) target/original input per sample
                'target_masks': [],  # list of len(num_folds), each element: (num_samples, seq_len, feat_dim) boolean mask per sample
                'metrics': [],  # list of len(num_folds), each element: (num_samples, num_metrics) metric per sample
                'IDs': []}  # list of len(num_folds), each element: (num_samples,) ID per sample

    for i, tgt_feats in enumerate(target_feats):

        dataset.mask_feats = tgt_feats  # set the transduction target features

        loader = DataLoader(dataset=dataset,
                            batch_size=config['batch_size'],
                            shuffle=False,
                            num_workers=config['num_workers'],
                            pin_memory=True,
                            collate_fn=lambda x: collate_unsuperv(x, max_len=config['max_seq_len']))

        evaluator = UnsupervisedRunner(model, loader, device, loss_module,
                                       print_interval=config['print_interval'], console=config['console'])

        logger.info("Evaluating {} set, fold: {}, target features: {}".format(dataset_name, i, tgt_feats))
        aggr_metrics, per_batch = evaluate(evaluator)

        metrics_array = convert_metrics_per_batch_to_per_sample(per_batch['metrics'], per_batch['target_masks'])
        metrics_array = np.concatenate(metrics_array, axis=0)
        allfolds['metrics'].append(metrics_array)
        allfolds['predictions'].append(np.concatenate(per_batch['predictions'], axis=0))
        allfolds['targets'].append(np.concatenate(per_batch['targets'], axis=0))
        allfolds['target_masks'].append(np.concatenate(per_batch['target_masks'], axis=0))
        allfolds['IDs'].append(np.concatenate(per_batch['IDs'], axis=0))

        metrics_mean = np.mean(metrics_array, axis=0)
        metrics_std = np.std(metrics_array, axis=0)
        for m, metric_name in enumerate(list(aggr_metrics.items())[1:]):
            logger.info("{}:: Mean: {:.3f}, std: {:.3f}".format(metric_name, metrics_mean[m], metrics_std[m]))

    pred_filepath = os.path.join(config['pred_dir'], dataset_name + '_fold_transduction_predictions.pickle')
    logger.info("Serializing predictions into {} ... ".format(pred_filepath))
    with open(pred_filepath, 'wb') as f:
        pickle.dump(allfolds, f, pickle.HIGHEST_PROTOCOL)


def convert_metrics_per_batch_to_per_sample(metrics, target_masks):
    """
    Args:
        metrics: list of len(num_batches), each element: list of len(num_metrics), each element: (num_active_in_batch,) metric per element
        target_masks: list of len(num_batches), each element: (batch_size, seq_len, feat_dim) boolean mask: 1s active, 0s ignore
    Returns:
        metrics_array = list of len(num_batches), each element: (batch_size, num_metrics) metric per sample
    """
    metrics_array = []
    for b, batch_target_masks in enumerate(target_masks):
        num_active_per_sample = np.sum(batch_target_masks, axis=(1, 2))
        batch_metrics = np.stack(metrics[b], axis=1)  # (num_active_in_batch, num_metrics)
        ind = 0
        metrics_per_sample = np.zeros((len(num_active_per_sample), batch_metrics.shape[1]))  # (batch_size, num_metrics)
        for n, num_active in enumerate(num_active_per_sample):
            new_ind = ind + num_active
            metrics_per_sample[n, :] = np.sum(batch_metrics[ind:new_ind, :], axis=0)
            ind = new_ind
        metrics_array.append(metrics_per_sample)
    return metrics_array


def evaluate(evaluator):
    """Perform a single, one-off evaluation on an evaluator object (initialized with a dataset)"""

    eval_start_time = time.time()
    with torch.no_grad():
        aggr_metrics, per_batch = evaluator.evaluate(epoch_num=None, keep_all=True)
    eval_runtime = time.time() - eval_start_time
    print()
    print_str = 'Evaluation Summary: '
    for k, v in aggr_metrics.items():
        if v is not None:
            print_str += '{}: {:8f} | '.format(k, v)
    logger.info(print_str)
    logger.info("Evaluation runtime: {} hours, {} minutes, {} seconds\n".format(*utils.readable_time(eval_runtime)))

    return aggr_metrics, per_batch


def validate(val_evaluator, tensorboard_writer, config, best_metrics, best_value, epoch):
    """Run an evaluation on the validation set while logging metrics, and handle outcome"""

    logger.info("Evaluating on validation set ...")
    eval_start_time = time.time()
    with torch.no_grad():
        aggr_metrics, per_batch = val_evaluator.evaluate(epoch, keep_all=True)
    eval_runtime = time.time() - eval_start_time
    logger.info("Validation runtime: {} hours, {} minutes, {} seconds\n".format(*utils.readable_time(eval_runtime)))

    global val_times
    val_times["total_time"] += eval_runtime
    val_times["count"] += 1
    avg_val_time = val_times["total_time"] / val_times["count"]
    avg_val_batch_time = avg_val_time / len(val_evaluator.dataloader)
    avg_val_sample_time = avg_val_time / len(val_evaluator.dataloader.dataset)
    logger.info("Avg val. time: {} hours, {} minutes, {} seconds".format(*utils.readable_time(avg_val_time)))
    logger.info("Avg batch val. time: {} seconds".format(avg_val_batch_time))
    logger.info("Avg sample val. time: {} seconds".format(avg_val_sample_time))

    print()
    print_str = 'Epoch {} Validation Summary: '.format(epoch)
    for k, v in aggr_metrics.items():
        tensorboard_writer.add_scalar('{}/val'.format(k), v, epoch)
        print_str += '{}: {:8f} | '.format(k, v)
    logger.info(print_str)

    if config['key_metric'] in NEG_METRICS:
        condition = (aggr_metrics[config['key_metric']] < best_value)
    else:
        condition = (aggr_metrics[config['key_metric']] > best_value)
    if condition:
        best_value = aggr_metrics[config['key_metric']]
        utils.save_model(os.path.join(config['save_dir'], 'model_best.pth'), epoch, val_evaluator.model)
        best_metrics = aggr_metrics.copy()

        pred_filepath = os.path.join(config['pred_dir'], 'best_predictions')
        
        save_dict = {}
        for k, v_list in per_batch.items():
            if len(v_list) > 0:
                try:
                    # Flatten lists of arrays (e.g., [array(4,1), array(1,1)]) into single array (5,1)
                    save_dict[k] = np.concatenate(v_list, axis=0)
                except Exception as e:
                    # Fallback for unexpected shapes
                    save_dict[k] = np.array(v_list, dtype=object) 
            else:
                save_dict[k] = v_list
                
        np.savez(pred_filepath, **save_dict)

        try:

            if 'targets' in per_batch and 'predictions' in per_batch:
                logger.info("Saving best predictions to CSV...")

                all_predictions = np.concatenate(per_batch['predictions'], axis=0)
                all_targets = np.concatenate(per_batch['targets'], axis=0)

                df = None

                if all_targets.ndim > 1 and all_targets.shape[1] > 1:

                    target_cols = {f'actual_{i}': all_targets[:, i] for i in range(all_targets.shape[1])}
                    pred_cols = {f'predicted_{i}': all_predictions[:, i] for i in range(all_predictions.shape[1])}
                    df = pd.DataFrame({**target_cols, **pred_cols})
                else:

                    df = pd.DataFrame({
                        'actual': all_targets.flatten(),
                        'predicted': all_predictions.flatten()
                    })

                csv_filepath = os.path.join(config['pred_dir'], 'best_predictions.csv')
                df.to_csv(csv_filepath, index=False)
                logger.info(f"Successfully saved predictions to {csv_filepath}")

        except Exception as e:
            logger.error(f"Failed to save predictions to CSV: {e}")

    return aggr_metrics, best_metrics, best_value


def check_progress(epoch):

    if epoch in [100, 140, 160, 220, 280, 340]:
        return True
    else:
        return False


class BaseRunner(object):

    def __init__(self, model, dataloader, device, loss_module, optimizer=None, l2_reg=None, print_interval=10, console=True):

        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.optimizer = optimizer
        self.loss_module = loss_module
        self.l2_reg = l2_reg
        self.print_interval = print_interval
        self.printer = utils.Printer(console=console)

        self.epoch_metrics = OrderedDict()

    def train_epoch(self, epoch_num=None):
        raise NotImplementedError('Please override in child class')

    def evaluate(self, epoch_num=None, keep_all=True):
        raise NotImplementedError('Please override in child class')

    def print_callback(self, i_batch, metrics, prefix=''):

        total_batches = len(self.dataloader)

        template = "{:5.1f}% | batch: {:9d} of {:9d}"
        content = [100 * (i_batch / total_batches), i_batch, total_batches]
        for met_name, met_value in metrics.items():
            template += "\t|\t{}".format(met_name) + ": {:g}"
            content.append(met_value)

        dyn_string = template.format(*content)
        dyn_string = prefix + dyn_string
        self.printer.print(dyn_string)


class UnsupervisedRunner(BaseRunner):

    def train_epoch(self, epoch_num=None):

        self.model = self.model.train()

        epoch_loss = 0  # total loss of epoch
        total_active_elements = 0  # total unmasked elements in epoch
        for i, batch in enumerate(self.dataloader):

            X, targets, target_masks, padding_masks, IDs = batch
            targets = targets.to(self.device)
            target_masks = target_masks.to(self.device)  # 1s: mask and predict, 0s: unaffected input (ignore)
            padding_masks = padding_masks.to(self.device)  # 0s: ignore

            predictions = self.model(X.to(self.device), padding_masks)  # (batch_size, padded_length, feat_dim)

            # Cascade noise masks (batch_size, padded_length, feat_dim) and padding masks (batch_size, padded_length)
            target_masks = target_masks * padding_masks.unsqueeze(-1)
            loss = self.loss_module(predictions, targets, target_masks)  # (num_active,) individual loss (square error per element) for each active value in batch
            batch_loss = torch.sum(loss)
            mean_loss = batch_loss / len(loss)  # mean loss (over active elements) used for optimization

            if self.l2_reg:
                total_loss = mean_loss + self.l2_reg * l2_reg_loss(self.model)
            else:
                total_loss = mean_loss

            # Zero gradients, perform a backward pass, and update the weights.
            self.optimizer.zero_grad()
            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
            self.optimizer.step()

            metrics = {"loss": mean_loss.item()}
            if i % self.print_interval == 0:
                ending = "" if epoch_num is None else 'Epoch {} '.format(epoch_num)
                self.print_callback(i, metrics, prefix='Training ' + ending)

            with torch.no_grad():
                total_active_elements += len(loss)
                epoch_loss += batch_loss.item()  # add total loss of batch

        epoch_loss = epoch_loss / total_active_elements  # average loss per element for whole epoch
        self.epoch_metrics['epoch'] = epoch_num
        self.epoch_metrics['loss'] = epoch_loss
        return self.epoch_metrics

    def evaluate(self, epoch_num=None, keep_all=True):

        self.model = self.model.eval()

        epoch_loss = 0  # total loss of epoch
        total_active_elements = 0  # total unmasked elements in epoch

        if keep_all:
            per_batch = {'target_masks': [], 'targets': [], 'predictions': [], 'metrics': [], 'IDs': []}
        for i, batch in enumerate(self.dataloader):

            X, targets, target_masks, padding_masks, IDs = batch
            targets = targets.to(self.device)
            target_masks = target_masks.to(self.device)  # 1s: mask and predict, 0s: unaffected input (ignore)
            padding_masks = padding_masks.to(self.device)  # 0s: ignore

            predictions = self.model(X.to(self.device), padding_masks)  # (batch_size, padded_length, feat_dim)

            # Cascade noise masks (batch_size, padded_length, feat_dim) and padding masks (batch_size, padded_length)
            target_masks = target_masks * padding_masks.unsqueeze(-1)
            loss = self.loss_module(predictions, targets, target_masks)  # (num_active,) individual loss (square error per element) for each active value in batch
            batch_loss = torch.sum(loss).cpu().item()
            mean_loss = batch_loss / len(loss)  # mean loss (over active elements) used for optimization the batch

            if keep_all:
                per_batch['target_masks'].append(target_masks.cpu().numpy())
                per_batch['targets'].append(targets.cpu().numpy())
                per_batch['predictions'].append(predictions.cpu().numpy())
                per_batch['metrics'].append([loss.cpu().numpy()])
                per_batch['IDs'].append(IDs)

            metrics = {"loss": mean_loss}
            if i % self.print_interval == 0:
                ending = "" if epoch_num is None else 'Epoch {} '.format(epoch_num)
                self.print_callback(i, metrics, prefix='Evaluating ' + ending)

            total_active_elements += len(loss)
            epoch_loss += batch_loss  # add total loss of batch

        epoch_loss = epoch_loss / total_active_elements  # average loss per element for whole epoch
        self.epoch_metrics['epoch'] = epoch_num
        self.epoch_metrics['loss'] = epoch_loss

        if keep_all:
            return self.epoch_metrics, per_batch
        else:
            return self.epoch_metrics


class AdaptiveLossCoefficient:
    """
    Implements your 3-Case adaptive lambda strategy:
    - Case 1 (Early): Main loss is high -> Keep lambda low.
    - Case 2 (Mid): Main loss is low, Var loss is high -> INCREASE lambda.
    - Case 3 (Converged): Main loss is low, Var loss is low -> DECREASE lambda.
    """

    def __init__(self, base_lambda=0.05, min_lambda=0.001, max_lambda=0.2,
                 increase_factor=1.5, decrease_factor=0.99,
                 patience=20, main_loss_threshold=100.0, var_loss_threshold=0.1):
        self.current_lambda = base_lambda
        self.min_lambda = min_lambda
        self.max_lambda = max_lambda
        self.increase_factor = increase_factor
        self.decrease_factor = decrease_factor
        self.patience = patience

        # Thresholds to define "low" loss
        self.main_loss_threshold = main_loss_threshold
        self.var_loss_threshold = var_loss_threshold

        self.patience_counter = 0

        logger.info(f"Initialized AdaptiveLambda (3-Case Logic): "
                    f"start_lambda={base_lambda}, range=[{min_lambda}, {max_lambda}], "
                    f"factors=[inc:{increase_factor}, dec:{decrease_factor}], "
                    f"patience={patience} batches, "
                    f"thresholds=[main:{main_loss_threshold}, var:{var_loss_threshold}]")

    def get_coeff(self):
        return self.current_lambda

    def step(self, current_main_loss, current_var_loss):
        """
        Updates the coefficient based on the main loss.
        Call this every batch.
        """
        self.patience_counter += 1
        if self.patience_counter >= self.patience:
            new_lambda = self.current_lambda * self.decrease_factor
            self.current_lambda = max(new_lambda, self.min_lambda)
            logger.info(f"DECREASING lambda to {self.current_lambda:.6f}")
            self.patience_counter = 0
        return

class SupervisedRunner(BaseRunner):

    def __init__(self, model, dataloader, device, loss_module, optimizer=None, l2_reg=None,
                 print_interval=10, console=True):
        super(SupervisedRunner, self).__init__(model, dataloader, device, loss_module,
                                               optimizer, l2_reg, print_interval, console)

        # Adaptive Loss Coefficient
        # self.lambda_adapter = AdaptiveLossCoefficient(
        #     base_lambda=0.04,  
        #     min_lambda=1e-6,  
        #     max_lambda=0.04,  
        #     increase_factor=1.01,  
        #     decrease_factor=0.9,  
        #     patience=10,  
        #     main_loss_threshold=100.0,  
        #     var_loss_threshold=3  
        # )

        # Fixed Loss Coefficient
        self.lambda_adapter = AdaptiveLossCoefficient(
            base_lambda=0.05,  
            min_lambda=0.05,  
            max_lambda=0.05,  
            increase_factor=1.01,  
            decrease_factor=0.9,  
            patience=10,  
            main_loss_threshold=100.0,  
            var_loss_threshold=3  
        )

        self.current_lambda = self.lambda_adapter.get_coeff()
        logger.info(f"Initialized 3-Case adaptive lambda. Starting_lambda={self.current_lambda}")


    def train_epoch(self, epoch_num=None):
        self.model = self.model.train()
        
        epoch_loss = 0
        total_groups = 0
        
        for i, batch in enumerate(self.dataloader):
            X, targets, padding_masks, IDs = batch
            
            # Ensure batch_size is a multiple of 5
            batch_size = X.shape[0]
            if batch_size % 5 != 0:
                logger.warning(f"Batch size {batch_size} is not multiple of 5, skipping")
                continue
                
            targets = targets.to(self.device)
            padding_masks = padding_masks.to(self.device)
            
            # Forward pass
            predictions = self.model(X.to(self.device), padding_masks)
            
            # Group processing
            group_size = 5
            num_groups = batch_size // group_size
            
            # Reshape to (num_groups, group_size, ...)
            predictions_reshaped = predictions.reshape(num_groups, group_size, -1)
            targets_reshaped = targets.reshape(num_groups, group_size, -1)
            
            # Calculate the sum of predictions for each group
            group_predictions_sum = predictions_reshaped.sum(dim=1)  # (num_groups, ...)
            
            # Use the 5th sample's label as the group target (take the label of the last sample in each group)
            group_targets = targets_reshaped[:, -1, :]  
            
            # Print detailed logs only for the first batch of each epoch to prevent console spam
            if (i == 0) and (epoch_num is not None) and (epoch_num > 0):
                # Move data to CPU and convert to numpy for printing
                with torch.no_grad():
                    preds_np = predictions_reshaped.cpu().numpy()
                    targets_np = targets_reshaped.cpu().numpy()
                    sum_preds_np = group_predictions_sum.cpu().numpy()
                    sum_targets_np = group_targets.cpu().numpy()

                    # Loop through and print each group in this batch (usually just the first group as an example)
                    for g_idx in range(min(num_groups, 2)):  # Print only the first 2 groups
                        logger.info(f"--- Epoch {epoch_num}, Batch {i}, Group {g_idx} Check ---")
                        
                        # Print 5 individual labels
                        logger.info(f"  Individual Labels: [" +
                                    ", ".join([f"{t[0]:.2f}" for t in targets_np[g_idx]]) +
                                    "]")
                        
                        # Print 5 individual predictions
                        logger.info(f"  Individual Preds:  [" + 
                                    ", ".join([f"{p[0]:.4f}" for p in preds_np[g_idx]]) + 
                                    "]")

                        # Print values used for loss calculation
                        logger.info(f"  ==> Summed Prediction: {sum_preds_np[g_idx][0]:.4f}")
                        logger.info(f"  ==> Target Label (5th): {sum_targets_np[g_idx][0]:.2f} (This label is used for loss)")

            # Calculate loss
            loss = self.loss_module(group_predictions_sum, group_targets)
            
            # Process loss tensor (following the original pipeline)
            if loss.dim() > 0 and loss.numel() > 1:
                batch_loss = torch.sum(loss)
                mean_loss = batch_loss / len(loss)
            else:
                mean_loss = loss

            # Calculate Variance Loss
            current_lambda = self.lambda_adapter.get_coeff()

            if current_lambda > 0:
                variance_per_group = torch.var(predictions_reshaped, dim=1, unbiased=False)
                mean_variance_loss = torch.mean(variance_per_group)

            total_loss = mean_loss + (current_lambda * mean_variance_loss)

            # Optimization step
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
            self.optimizer.step()

            # Update Adaptive Lambda
            self.lambda_adapter.step(mean_loss.item(), mean_variance_loss.item())
            logger.info(f"Update lambda={self.lambda_adapter.get_coeff()}")
            
            # Record metrics
            loss_value = mean_loss.item() if hasattr(mean_loss, 'item') else mean_loss
            metrics = {"loss": loss_value}
            if i % self.print_interval == 0:
                ending = "" if epoch_num is None else 'Epoch {} '.format(epoch_num)
                self.print_callback(i, metrics, prefix='Training ' + ending)
            
            epoch_loss += loss_value
            total_groups += num_groups
        
        avg_epoch_loss = epoch_loss / total_groups if total_groups > 0 else 0
        self.epoch_metrics['epoch'] = epoch_num
        self.epoch_metrics['loss'] = avg_epoch_loss
        return self.epoch_metrics
    
    def evaluate(self, epoch_num=None, keep_all=True):
        self.model = self.model.eval()

        epoch_loss = 0
        total_groups = 0

        per_batch = {'targets': [], 'predictions': [], 'metrics': [], 'IDs': []}

        for i, batch in enumerate(self.dataloader):
            X, targets, padding_masks, IDs = batch
            
            # Ensure batch_size is a multiple of 5
            batch_size = X.shape[0]
            if batch_size % 5 != 0:
                continue
                
            targets = targets.to(self.device)
            padding_masks = padding_masks.to(self.device)

            with torch.no_grad():
                predictions = self.model(X.to(self.device), padding_masks)
                
                # Group processing
                group_size = 5
                num_groups = batch_size // group_size
                
                # Reshape to (num_groups, group_size, ...)
                predictions_reshaped = predictions.reshape(num_groups, group_size, -1)
                targets_reshaped = targets.reshape(num_groups, group_size, -1)
                
                # Calculate the sum of predictions for each group
                group_predictions_sum = predictions_reshaped.sum(dim=1)  # (num_groups, ...)
                
                # Use the 5th sample's label as the group target (take the label of the last sample in each group)
                group_targets = targets_reshaped[:, -1, :]  
                
                # Calculate loss - following the original pipeline
                # Use the same loss calculation method as in training
                loss = self.loss_module(group_predictions_sum, group_targets)
                
                # Check loss shape and process accordingly
                if loss.dim() > 0 and loss.numel() > 1:
                    # If loss is a tensor of multiple values, calculate the mean
                    batch_loss_tensor = torch.sum(loss)
                    mean_loss = batch_loss_tensor / len(loss)
                    loss_value = mean_loss.item()
                    
                    # Store the loss value for each sample (following the original pipeline)
                    per_batch['metrics'].append(loss.cpu().numpy())
                else:
                    # If loss is already a scalar
                    loss_value = loss.item()
                    batch_loss_tensor = loss # Keep as tensor for later use
                    # Store scalar loss (convert to numpy array for consistency)
                    per_batch['metrics'].append(np.array([loss_value]))
                
                # Store results
                per_batch['predictions'].append(group_predictions_sum.cpu().numpy())
                per_batch['targets'].append(group_targets.cpu().numpy())
                
                # Store the ID of the 5th sample
                group_ids = []
                for group_idx in range(num_groups):
                    fifth_sample_idx = group_idx * group_size + 4
                    if fifth_sample_idx < len(IDs):
                        group_ids.append(IDs[fifth_sample_idx])
                per_batch['IDs'].append(group_ids)
                
                metrics = {"loss": loss_value}
                if i % self.print_interval == 0:
                    ending = "" if epoch_num is None else 'Epoch {} '.format(epoch_num)
                    self.print_callback(i, metrics, prefix='Evaluating ' + ending)
                
                # Use .item() to accumulate pure Python numbers
                epoch_loss += batch_loss_tensor.item()
                total_groups += num_groups

        # Handle the end of the epoch
        if total_groups == 0:
            logger.warning("No complete groups found in evaluation")
            avg_epoch_loss = 0
        else:
            avg_epoch_loss = epoch_loss / total_groups # Loss was per-batch, now it's per-group avg

        self.epoch_metrics['epoch'] = epoch_num
        self.epoch_metrics['loss'] = avg_epoch_loss

        # Calculate other metrics
        if total_groups > 0 and keep_all and len(per_batch['predictions']) > 0:
            all_predictions = np.concatenate(per_batch['predictions'], axis=0)
            all_targets = np.concatenate(per_batch['targets'], axis=0)
            
            try:
                # Calculate regression metrics
                mse = mean_squared_error(all_targets, all_predictions)
                self.epoch_metrics['mse'] = float(mse)
                self.epoch_metrics['rmse'] = float(np.sqrt(mse))
                
                # Avoid division by zero in MAPE calculation
                # (Create a mask for non-zero targets)
                non_zero_mask = all_targets != 0
                if np.any(non_zero_mask):
                    mape = mean_absolute_percentage_error(all_targets[non_zero_mask], all_predictions[non_zero_mask])
                    self.epoch_metrics['mape'] = float(mape)
                else:
                    self.epoch_metrics['mape'] = float('inf') 
                    
                self.epoch_metrics['r2'] = float(r2_score(all_targets, all_predictions))
                
            except Exception as e:
                logger.error(f"Failed to calculate regression metrics: {e}")

        if keep_all:
            return self.epoch_metrics, per_batch
        else:
            return self.epoch_metrics