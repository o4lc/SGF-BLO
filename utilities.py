import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

def get_axs(toy_example=False):
    fig1, ax1 = plt.subplots(1, 1, figsize=(7, 5))
    fig11, ax11 = plt.subplots(1, 1, figsize=(7, 5))
    fig2, ax2 = plt.subplots(1, 1, figsize=(7, 5))
    if not toy_example:
        fig3, ax3 = plt.subplots(1, 1, figsize=(7, 5))
        fig4, ax4 = plt.subplots(1, 1, figsize=(7, 5))
        return fig1, ax1, fig11, ax11, fig2, ax2, fig3, ax3, fig4, ax4
    else:
        return fig1, ax1, fig11, ax11, fig2, ax2

def scenario_setup(id):
    if id == 0: #scenarioAlpha
        return [('InversionFree', 0.01, 0.1, None), ('InversionFree', 0.05, 0.1, None), 
                 ('InversionFree', 0.1, 0.1, None), ('InversionFree', 0.5, 0.1, None), 
                 ('InversionFree', 1, 0.1, None)]
    elif id == 1: #scenarioEps
        return [('InversionFree', 0.1, 0.05, None), ('InversionFree', 0.1, 0.1, None),
                ('InversionFree', 0.1, 0.5, None), ('InversionFree', 0.1, 1, None)]
    elif id == 2: #scenario2ndOrder
        return [('SecondOrder', 0.1, None, None), ('STABLE', 0.1, None, None)]
    elif id == 3: #scenarioOthers
        return [('InversionFree', 0.1, 0.1, 0.25), ('AITBio', 0.1, 0.1, 0.25), ('TTSA', 0.1, 0.1, 0.25),
                ('InversionFree', 0.1, 0.1, 0.5), ('AITBio', 0.1, 0.1, 0.5), ('TTSA', 0.1, 0.1, 0.5)]
    else:
         return [('InversionFree', 0.01, 0.1, None)]

    
def load_setup(toy_example=False, p=None):
    if toy_example:
        c = torch.load('data/c.pt', weights_only=True)
        d = torch.load('data/d.pt', weights_only=True)
        A = torch.load('data/A.pt', weights_only=True)
        H = torch.load('data/H.pt', weights_only=True)

        dimX = (A.shape[0], 1);dimY = (A.shape[1], 1);

        def f(x, y):
            x = x.reshape(dimX); y = y.reshape(dimY)
            return torch.sin(c.T @ x + d.T @ y) + torch.log(torch.linalg.norm(x+y)**2 + 1)

        def g(x, y):
            x = x.reshape(dimX); y = y.reshape(dimY)
            return 0.5 * torch.linalg.norm(H@y - x)**2
        
        print('dim X:', dimX, 'dim Y:' ,dimY)
        return f, g, c, d, A, H, dimX, dimY
    
    else:
        string = 'p' + str(p)
        A_tr = torch.load('data/A_tr' + string + '.pt', weights_only=True).to(torch.float32)
        B_tr = torch.load('data/B_tr' + string + '.pt', weights_only=True).to(torch.float32)
        
        A_val = torch.load('data/A_val' + string + '.pt', weights_only=True).to(torch.float32)
        B_val = torch.load('data/B_val' + string + '.pt', weights_only=True).to(torch.float32)

        A_test = torch.load('data/A_test' + string + '.pt', weights_only=True).to(torch.float32)
        B_test = torch.load('data/B_test' + string + '.pt', weights_only=True).to(torch.float32)

        lam = 0.001      # Regularization parameter
        dimX = (A_tr.shape[0], 1); dimY = (A_tr.shape[1], B_tr.shape[1]);

        def f(x, y):
            x = x.reshape(dimX); y = y.reshape(dimY)
            loss = F.cross_entropy(A_val @ y, B_val)
            return loss

        def g(x, y):
            x = x.reshape(dimX); y = y.reshape(dimY)
            loss = F.cross_entropy(A_tr @ y, B_tr, reduction='none')
            return torch.mean(torch.mul(loss, torch.sigmoid(x))) + lam * torch.pow(torch.norm(y, 'fro'), 2)

        print('dim X:', dimX, 'dim Y:' ,dimY)
        return f, g, A_tr, B_tr, A_val, B_val, A_test, B_test, dimX, dimY
    
# def calc_derivatives(x, y):
#     f_val = f(x,y)
#     g_val = g(x,y)

#     # 1st derivatives
#     dfdy = torch.autograd.grad(f_val, y, create_graph=True, allow_unused=True, materialize_grads=True)[0]
#     dfdx = torch.autograd.grad(f_val, x, create_graph=True, allow_unused=True, materialize_grads=True)[0]

#     dgdy = torch.autograd.grad(g_val, y, create_graph=True, allow_unused=True, materialize_grads=True)[0]
#     dgdx = torch.autograd.grad(g_val, x, create_graph=True, allow_unused=True, materialize_grads=True)[0]

#     # Initialize tensors for 2nd derivatives
#     dgdyy = torch.zeros((sizeY, sizeY))
#     dgdyx = torch.zeros((sizeY, sizeX))
#     # Compute 2nd derivatives element-wise
#     for i in range(dgdy.shape[0]):
#         dgdyy[i, :] = torch.autograd.grad(dgdy[i], y, retain_graph=True, create_graph=True, allow_unused=True, materialize_grads=True)[0][:, 0]
#         dgdyx[i, :] = torch.autograd.grad(dgdy[i], x, retain_graph=True, create_graph=True, allow_unused=True, materialize_grads=True)[0][:, 0]
    
#     return dfdx, dfdy, dgdx, dgdy, dgdyy, dgdyx
#     # now = time.time()
#     # dfdx, dfdy, dgdx, dgdy, dgdyy, dgdyx = calc_derivatives(x, y)
#     # print('Time elapsed:', time.time() - now)
#     # now = time.time()
#     # dfdx2, dfdy2, dgdx2, dgdy2, dgdyy2, dgdyx2 = calc_derivatives_analytic(x, y)
#     # print('Time elapsed:', time.time() - now)

#     # assert torch.allclose(dfdx, dfdx2)
#     # assert torch.allclose(dfdy, dfdy2)
#     # assert torch.allclose(dgdx, dgdx2)
#     # assert torch.allclose(dgdy, dgdy2, atol=1e-2)
#     # assert torch.allclose(dgdyx, dgdyx2)
#     # assert torch.allclose(dgdyy, dgdyy2)
    

def calculate_accuracy(A, B, W):
    predictions = A @ W  # (n_samples, num_classes)
    predicted_labels = torch.argmax(predictions, dim=1)
    true_labels = torch.argmax(B, dim=1)  # Convert one-hot to class indices
    correct_predictions = (predicted_labels == true_labels).float().sum()
    return correct_predictions / B.size(0)

# Helper function to calculate loss
def calculate_loss(A, B, W):
    logits = A @ W  # (n_samples, num_classes)
    true_labels = torch.argmax(B, dim=1)  # Convert one-hot to class indices
    loss = torch.nn.functional.cross_entropy(logits, true_labels, reduction='mean')
    return loss.unsqueeze(0).unsqueeze(0)


def calculate_losses(sol, f, sizeX, sizeY, calc_derivatives):
    # Calculate lossf directly as a tensor, avoid unnecessary reshaping
    lossf = f(sol[:sizeX], sol[sizeX:]).reshape(-1, )

    # Calculate derivatives
    dfdx, dfdy, dgdx, dgdy, dgdyy, dgdyx = calc_derivatives(sol[:sizeX], sol[sizeX:])

    # Compute norms as tensors
    lossG = torch.linalg.norm(dgdy)
    lossF = torch.linalg.norm(dfdx - dgdyx.T @ dgdyy.inverse() @ dfdy)
    # Detach and convert to NumPy arrays for storage
    return (
        lossf.detach().numpy(), 
        lossG.detach().numpy(), 
        lossF.detach().numpy()
    )

def conjugate_gradient(A, b, x0, N):
    r = b - np.dot(A, x0)
    p = r.copy()
    x = x0.copy()
    rs_old = np.dot(r.T, r)

    for i in range(N):
        Ap = np.dot(A, p)
        alpha = rs_old / np.dot(p.T, Ap)
        x += alpha * p
        r -= alpha * Ap
        rs_new = np.dot(r.T, r)
        if np.sqrt(rs_new) < 1e-10:  # Convergence criterion
            break
        p = r + (rs_new / rs_old) * p
        rs_old = rs_new

    return torch.Tensor(x)

# def AIDBio(x, y0, alpha=0.1, beta=0.01, K=10, D=10):
#     y = y0
#     lossF, lossG, lossF2 = [], [], []
#     for k in range(K):
#         term = 0
#         termxy = []
#         termyy = []
#         for t in range(D):
#             dfdx, dfdy, dgdx, dgdy, dgdyy, dgdyx = calc_derivatives(x, y)
#             termxy.append(dgdyx)
#             termyy.append(torch.eye(sizeY) - alpha * dgdyy)
#             y = y - alpha * dgdy
            
#             term1, term2, term3 = calculate_losses(torch.cat((x, y), 0))
#             lossF.append(term1); lossG.append(term2); lossF2.append(term3)

#         for t in range(D):
#             tmp = torch.eye(sizeY)
#             for j in range(t+1, D):
#                 tmp = termyy[j] @ tmp
#             term += termxy[t].T @ tmp
#         dfdx, dfdy, dgdx, dgdy, dgdyy, dgdyx = calc_derivatives(x, y)
#         x = x - beta * (dfdx - alpha * term @ dfdy)
        
#         term1, term2, term3 = calculate_losses(torch.cat((x, y), 0))
#         lossF.append(term1); lossG.append(term2); lossF2.append(term3)
        
#     return np.array(lossF), np.array(lossG), np.array(lossF2)
