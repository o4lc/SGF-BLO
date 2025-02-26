import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from setup import myNN

def add_loss(W, train_accuracy, val_accuracy, test_accuracy, train_loss, val_loss, test_loss, pars, dimY, arch):
    A_tr, B_tr, A_val, B_val, A_test, B_test = pars
    train_accuracy.append(calculate_accuracy(A_tr, B_tr, W.reshape(dimY, -1),arch=arch).reshape(-1))
    val_accuracy.append(calculate_accuracy(A_val, B_val, W.reshape(dimY, -1),arch=arch).reshape(-1))
    test_accuracy.append(calculate_accuracy(A_test, B_test, W.reshape(dimY, -1),arch=arch).reshape(-1))

    train_loss.append(calculate_loss(A_tr, B_tr, W.reshape(dimY, -1), arch=arch).reshape(-1))
    val_loss.append(calculate_loss(A_val, B_val, W.reshape(dimY, -1), arch=arch).reshape(-1))
    test_loss.append(calculate_loss(A_test, B_test, W.reshape(dimY, -1), arch=arch).reshape(-1))
    return train_accuracy, val_accuracy, test_accuracy, train_loss, val_loss, test_loss

def calculate_accuracy(A, B, W, arch=None):
    if arch is not None: isNN = True
    with torch.no_grad():
        if isNN:
            predictions = myNN(arch, A, W)
        else:
            predictions = A @ W  # (n_samples, num_classes)
        predicted_labels = torch.argmax(predictions, dim=1)
        true_labels = torch.argmax(B, dim=1)  # Convert one-hot to class indices
        correct_predictions = (predicted_labels == true_labels).float().sum()
        return (correct_predictions / B.size(0)).detach().cpu().numpy() 

# Helper function to calculate loss
def calculate_loss(A, B, W, arch=None):
    if arch is not None: isNN = True
    with torch.no_grad():
        if isNN:
            logits = myNN(arch, A, W)
        else:
            logits = A @ W  # (n_samples, num_classes)
        true_labels = torch.argmax(B, dim=1)  # Convert one-hot to class indices
        loss = torch.nn.functional.cross_entropy(logits, true_labels, reduction='mean')
    return loss.unsqueeze(0).unsqueeze(0).detach().cpu().numpy()


def calculate_losses(sol, f, sizeX, sizeY, calc_derivatives, testID=None, deltaX=None):
    # Calculate derivatives
    x, y = sol[:sizeX], sol[sizeX:]
    if testID not in [0, 3, 4]:
        matrixVectorProduct = True
        non_convex = True
    else:
        matrixVectorProduct = False
        non_convex = False
    dfdx, dfdy, dgdx, dgdy, dgdyy, dgdyx = calc_derivatives(x, y, matrixVectorProduct)

    with torch.no_grad():
        # Calculate lossf directly as a tensor, avoid unnecessary reshaping
        lossf = f(x, y).reshape(-1, )
        # Compute norms as tensors
        lossG = torch.linalg.norm(dgdy)
        if non_convex:
            lossF = torch.Tensor(torch.linalg.norm(deltaX, 2))
        else:
            lossF = torch.linalg.norm(dfdx - dgdyx.T @ dgdyy.inverse() @ dfdy)
        # Detach and convert to NumPy arrays for storage
        return (
            lossf.item(), 
            lossG.item(), 
            lossF.item()
        )


def conjugate_gradient(A, b, x0, N):
    # Determine if inputs are numpy or torch tensors
    is_numpy = isinstance(A, np.ndarray)

    # Define dot product and norm operations based on type
    dot = np.dot if is_numpy else torch.matmul
    norm = np.linalg.norm if is_numpy else torch.norm

    # Copy inputs correctly
    r = b - dot(A, x0)
    p = r.copy() if is_numpy else r.clone()
    x = x0.copy() if is_numpy else x0.clone()
    rs_old = dot(r.T, r)

    for i in range(N):
        Ap = dot(A, p)
        alpha = rs_old / dot(p.T, Ap)
        x += alpha * p
        r -= alpha * Ap
        rs_new = dot(r.T, r)

        if norm(r) < 1e-10:  # Convergence criterion
            break

        p = r + (rs_new / rs_old) * p
        rs_old = rs_new

    return torch.Tensor(x)


