import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

def get_axs(toy_example=False, toy_CS=False):
    fig1, ax1 = plt.subplots(1, 1, figsize=(8, 6))
    fig11, ax11 = plt.subplots(1, 1, figsize=(8, 6))
    fig2, ax2 = plt.subplots(1, 1, figsize=(8, 6))
    if not toy_example and not toy_CS:
        fig3, ax3 = plt.subplots(1, 1, figsize=(8, 6))
        fig4, ax4 = plt.subplots(1, 1, figsize=(8, 6))
        return fig1, ax1, fig11, ax11, fig2, ax2, fig3, ax3, fig4, ax4
    else:
        return fig1, ax1, fig11, ax11, fig2, ax2

def scenario_setup(id):
    '''
    (name of the method, alpha, epsilon, corropution rate, mode, beta/w)
    '''
    # mode = ['RXGD', 'QCQP', 'QP1', 'QP2', 'MO-GD', 'NN']
    if id == -1: #testingScenario
        return [('IFDT', 0.1, 0.1, 0, 'QP1', None), ('BOME', 0.1, 0.1, 0, None, None)]
    elif id == 0: #scenarioAlpha
        return [('IFCT', 0.01, 0.1, None, None, None), ('IFCT', 0.05, 0.1, None, None, None), 
                 ('IFCT', 0.1, 0.1, None, None, None), ('IFCT', 0.5, 0.1, None, None, None), 
                 ('IFCT', 1, 0.1, None, None, None)]
    elif id == 1: #scenarioEps
        return [('IFCT', 0.1, 0.05, None, None, None), ('IFCT', 0.1, 0.1, None, None, None),
                ('IFCT', 0.1, 0.2, None, None, None), ('IFCT', 0.1, 0.5, None, None, None)]
    elif id == 2: #scenarioOthers
        return [('IFCT', 0.1, 0.1, 0.25, None, None), ('AIDBio', 0.1, 0.1, 0.25, None, None),
                ('IFCT', 0.1, 0.1, 0.4, None, None), ('AIDBio', 0.1, 0.1, 0.4, None, None)]
    # ----------------------------------------
    elif id == 3: #scenario2ndOrder
        return [('SecondOrder', 0.1, None, None, None, None, None), ('STABLE', 0.1, None, None, None, None, None)]
    # ----------------------------------------
    elif id == 4: #scenarioIFDT-K ablation
        return [('IFDT', 0.1, 0.1, 0, 'QP1', None), ('IFDT', 0.1, 0.1, -1, 'QP1', None), ('IFDT', 0.1, 0.1, -2, 'QP1', None)]
    elif id == 5: #scenarioIFDT-K ablation
        return [('IFDT', 0.1, 0.1, 0, 'QP2', None), ('IFDT', 0.1, 0.1, -1, 'QP2', None), ('IFDT', 0.1, 0.1, -2, 'QP2', None)]
    elif id == 6: #scenarioIFDT SOTA
        return [('IFDT', 0.1, 0.1, 0, 'QP1', None), ('IFDT', 0.1, 0.1, 0, 'QP2', None), ('BOME', 0.1, 0.1, 0, None, None)]
    elif id == 7: #scenarioIFDT SOTA 2
        return [('IFDT', 0.1, 0.1, 0.25, 'QP1', None), ('IFDT', 0.1, 0.1, 0.25, 'QP2', None), \
                ('BOME', 0.1, 0.1, 0.25, None, None), ('VPBGD', 0.1, 0.1, 0.25, None, None)]
    elif id == 8: #scenarioIFDT SOTA 3
        return [('IFDT', 0.1, 0.1, 0.25, 'QP1', None), ('IFDT', 0.1, 0.1, 0.25, 'QP2', None), \
                ('BOME', 0.1, 0.1, 0.25, None, None), ('AIDBio', 0.1, 0.1, 0.25, None, None), ('VPBGD', 0.1, 0.1, 0.25, None, None)]
    # ----------------------------------------
    elif id == 9: #scenarioQCQP w ablation
        return [('IFDT', 0.1, 0.1, 0.25, 'QCQP', 0.001), ('IFDT', 0.1, 0.1, 0.25, 'QCQP', 0.01), ('IFDT', 0.1, 0.1, 0.25, 'QCQP', 0.1)]
    elif id == 10: #scenarioQCQP comparison for convex
        return [('IFDT', 0.1, 0.1, 0.25, 'QCQP', 0.01), ('AIDBio', 0.1, 0.1, 0.25, ' ', 0.01)]
    elif id == 11: #scenarioQCQP comparison for Large Scale
        p = 0.25
        return [('IFDT', 0.1, 0.5, p, 'QCQP', 0.001), ('BOME', 0.1, 0.5, p, ' ', 0.001), ('VPBGD', 0.1, 0.5, p, ' ', 0.001)]
    # ----------------------------------------
    elif id == 12: #scenarioQCQP log barrier
        return [('IFDT', 0.1, 0.1, 0.25, 'QCQP', 0.001), ('IFDT', 0.1, 0.1, 0.25, 'QCQP', 0.01), ('IFDT', 0.1, 0.1, 0.25, 'QCQP', 0.1)]
    # elif id == 10: #scenarioTest
    #     return [('IFDT', 0.01, 0.1, 0.25, 'QCQP', -1), ('IFDT', 0.01, 0.1, 0.25, 'QP1', -1), ('VPBGD', 0.01, 0.1, 0.25, ' ', -1)] 
    # elif id == 10: #scenarioTest
    #     return [('IFDT', 0.1, 0.001, 0.25, 'MOGD', 0.1), ('IFDT', 0.1, 0.001, 0.25, 'QP1', 0.1)] 
    # [('NLSolver', 0.01, 0.1, 0.25, ' ', 0.01)]
    # , ('VPBGD', 0.01, 0.01, 0.25, ' ', 0.01)] 
                # ('IFDT', 0.01, 50, 0.25, 'QP1'), ('BOME', 0.01, 50, 0.25, ' ')]
    # , ('IFDT', 0.5, 0.1, 0.25, 'QP1')]
    else:
         return [('IFCT', 0.01, 0.1, None)]

    
def load_setup(testID=0, p=None, device=None):
    # TODO Handle the experiments better!
    if device is None:
        raise ValueError('Device not specified')
    if testID in [0, 1, 2]:
        # Toy example
        c = torch.load('data/c.pt', weights_only=True).to(device)
        d = torch.load('data/d.pt', weights_only=True).to(device)
        A = torch.load('data/A.pt', weights_only=True).to(device)
        H = torch.load('data/H.pt', weights_only=True).to(device)

        dimX = (A.shape[0], 1);dimY = (A.shape[1], 1);

        def f(x, y):
            x = x.reshape(dimX).to(device); y = y.reshape(dimY).to(device)
            return torch.sin(c.T @ x + d.T @ y) + torch.log(torch.linalg.norm(x+y)**2 + 1)

        if testID == 0:
            def g(x, y):
                x = x.reshape(dimX); y = y.reshape(dimY)
                return 0.5 * torch.linalg.norm(H@y - x)**2
        elif testID == 1:
            def g(x, y):
                x = x.reshape(dimX); y = y.reshape(dimY)
                return torch.cos(0.5 * torch.linalg.norm(H@y - x)**2)
        elif testID == 2:
            def g(x, y):
                x = x.reshape(dimX); y = y.reshape(dimY)
                return 0.5 * torch.linalg.norm(H@y - x)**2 - torch.log(y).sum().reshape(1, 1) / 10
        else:
            raise ValueError('Invalid test case ID')
    
        return f, g, c, d, A, H, dimX, dimY
    
    elif testID == 3:
        # Toy Coreset selection
        y_tilde = torch.Tensor([[3], [-2]]).to(device)
        X = torch.Tensor([[1, 3], [3, 1], [-2, 2], [-3, 2]]).to(device)

        dimX = (4, 1); dimY = (2, 1);
        def f(x, y):
            x = x.reshape(dimX); y = y.reshape(dimY)
            return 0.5 * torch.linalg.norm(y - y_tilde)**2
        
        def g(x, y):
            x = x.reshape(dimX); y = y.reshape(dimY)
            return 0.5 * torch.linalg.norm(y - X.T @ torch.softmax(x,  dim=0))**2
        
        return f, g, y_tilde, X, dimX, dimY

    
    elif testID in [4, 5, 6]:
        arch = None
        # DHC with PCA and without PCA
        if testID == 4:
            string = 'p' + str(p)
        else:
            string = 'p' + str(p) + 'Full'
        A_tr = torch.load('data/A_tr' + string + '.pt', weights_only=True).to(torch.float32).to(device)
        B_tr = torch.load('data/B_tr' + string + '.pt', weights_only=True).to(torch.float32).to(device)
        
        A_val = torch.load('data/A_val' + string + '.pt', weights_only=True).to(torch.float32).to(device)
        B_val = torch.load('data/B_val' + string + '.pt', weights_only=True).to(torch.float32).to(device)

        A_test = torch.load('data/A_test' + string + '.pt', weights_only=True).to(torch.float32).to(device)
        B_test = torch.load('data/B_test' + string + '.pt', weights_only=True).to(torch.float32).to(device)

        lam = 0.001      # Regularization parameter
        if testID == 4 or testID == 5:
            dimX = (A_tr.shape[0], 1); dimY = (A_tr.shape[1], B_tr.shape[1]);
            def f(x, y):
                x = x.reshape(dimX); y = y.reshape(dimY)
                loss = F.cross_entropy(A_val @ y, B_val)
                return loss

            def g(x, y):
                x = x.reshape(dimX); y = y.reshape(dimY)
                loss = F.cross_entropy(A_tr @ y, B_tr, reduction='none')
                return torch.mean(torch.mul(loss, torch.sigmoid(x))) + lam * torch.pow(torch.norm(y, 'fro'), 2)
        else:
            # dim_1 = 50; dim_2 = 25
            arch = [A_tr.shape[1], 50, B_tr.shape[1]]
            model = SimpleNN(arch)
            # print(model, A_val.shape, B_val.shape)
            num_params = sum(p.numel() for p in model.parameters())
            del model
            torch.cuda.empty_cache() 

            dimX = (A_tr.shape[0], 1); dimY = (num_params, 1);
            def f(x, y, iflogits=False):
                x = x.reshape(dimX); y = y.reshape(dimY)
                # model = SimpleNN(A_tr.shape[1], dim_1, dim_2, B_tr.shape[1])
                # load_weights(model, y)
                # logits = model(A_val)
                # NN implementation to keep the gradients in the graph
                logits = myNN(arch, A_val, y)


                loss = F.cross_entropy(logits, B_val)
                if iflogits:
                    return loss, logits
                return loss

            def g(x, y, iflogits=False):
                x = x.reshape(dimX); y = y.reshape(dimY)

                logits = myNN(arch, A_tr, y)

                loss = F.cross_entropy(logits, B_tr, reduction='none')
                if iflogits:
                    return torch.mean(torch.mul(loss, torch.sigmoid(x))) + lam * torch.pow(torch.norm(y, 'fro'), 2), logits
                return torch.mean(torch.mul(loss, torch.sigmoid(x))) + lam * torch.pow(torch.norm(y, 'fro'), 2)

        print('dim X:', dimX, 'dim Y:' ,dimY)
        return f, g, A_tr, B_tr, A_val, B_val, A_test, B_test, dimX, dimY, arch

    else:
        raise ValueError('Invalid test case ID')
    

def myNN(arch, A, y):
    parNum = 0
    for l in range(len(arch) - 1):
        Wsize = arch[l] * arch[l + 1]; Bsize = arch[l + 1]
        if l == 0:
            xx = F.relu(A @ y[parNum: parNum + Wsize].reshape(A.shape[1], arch[1]) +
                            y[parNum + Wsize: parNum + Wsize + Bsize].reshape(1, arch[1]))
        elif l == len(arch) - 2:
            logits = xx @ y[parNum: parNum + Wsize].reshape(arch[1], arch[2]) +\
                    y[parNum + Wsize:].reshape(1, arch[2])
        else:
            xx = F.relu(xx @ y[parNum: parNum + Wsize].reshape(A.shape[1], arch[1]) +
                            y[parNum + Wsize: parNum + Wsize + Bsize].reshape(1, arch[1]))
        parNum += Wsize + Bsize
    return logits

class SimpleNN(nn.Module):
    def __init__(self, arch):
        super(SimpleNN, self).__init__()
        
        layers = []
        for i in range(len(arch) - 1):
            layers.append(nn.Linear(arch[i], arch[i + 1]))
            if i < len(arch) - 2:  # Apply ReLU to all but the last layer
                layers.append(nn.ReLU())
        
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

def load_weights(model, y):
    """Load weights from vector y into the model with autograd tracking."""
    start = 0
    for param in model.parameters():
        num_params = param.numel()
        with torch.no_grad():
            param.data = y[start:start + num_params].view(param.shape)  # Keeps tracking
        start += num_params



