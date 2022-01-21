from tqdm import tqdm
import torch
import numpy as np
import pruners
from generator import masked_parameters

def prune_loop(model, loss, pruner_name, dataloader, num_classes,
               device, sparsity, schedule, scope, epochs,
               reinitialize=False, 
               train_mode=False,
               classifier_sparsity=-1.0,
               prune_bias=False,
               prune_batchnorm=False,
               prune_residual=False,
               prune_verbose=False):
    r"""Applies score mask loop iteratively to a final sparsity level.
    """
    # Set model to train or eval mode
    model.train()
    if not train_mode:
        model.eval()

    # Convert model to double precision to avoid overflow in SnyFlow
    model.double()

    # Create pruner
    pruner = pruners.__dict__[pruner_name](
        masked_parameters(model, prune_bias, prune_batchnorm, prune_residual),
        num_classes
    )

    # Prune model
    for epoch in tqdm(range(epochs)):
        pruner.score(model, loss, dataloader, device)
        if schedule == 'exponential':
            sparse = sparsity**((epoch + 1) / epochs)
            if classifier_sparsity >= 0.0:
                classifier_sparse = classifier_sparsity**((epoch+1) / epochs)
            else:
                classifier_sparse = -1.0
        elif schedule == 'linear':
            sparse = 1.0 - (1.0 - sparsity)*((epoch + 1) / epochs)
            if classifier_sparsity >= 0.0:
                classifier_sparse = (
                    1.0
                    - (1.0 - classifier_sparsity) * (epoch + 1) / epochs
                )
            else:
                classifier_sparse = -1.0
        pruner.mask(sparse, scope, classifier_sparse)

        # Print pruning information
        if prune_verbose and (device == 0 or device == 'cpu'):
            print('Iteration {}'.format(epoch+1))
            for i, (mask, param) in enumerate(pruner.masked_parameters):
                print(f'Layer: {i}\t Shape: {param.shape}\t '
                      'Total Params: {mask.numel()}\t '
                      'Remaining params: {mask.detach().cpu().numpy().sum()}')

    # Reainitialize weights
    if reinitialize:
        model._initialize_weights()

    # Confirm sparsity level. Print warning messages if the resulting sparsity
    #   is different from the expected by a large margin.
    remaining_params, total_params = pruner.stats()
    print(f'Finished pruning: {remaining_params/total_params}% sparsity')
    if np.abs(remaining_params - total_params * sparsity) >= 5:
        print(f'WARNING: {remaining_params} prunable parameters remaining,'
              ' expected {total_params*sparsity}')
        print(f'WARNING: {remaining_params} prunable parameters remaining,'
              ' expected {total_params*sparsity}')
        print(f'WARNING: {remaining_params} prunable parameters remaining,'
              ' expected {total_params*sparsity}')

    # Set the model back to single precision
    model.float()

