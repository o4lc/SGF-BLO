import numpy as np
import torch
import torchdiffeq
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import argparse
import time
import torch.nn.functional as F
from tqdm import tqdm
from torchviz import make_dot


from utilities import conjugate_gradient, add_loss, calculate_losses, cvxpy_QCQP, cvxpy_MOGD
from setup import load_setup, scenario_setup, get_axs

def LineSearch_merit(deltaX, x, y, tt, lam, beta, alpha, k=-1):
    assert beta >= lam
    eta = 0.05
    t = tt
    # if k > 100:
    #     # print(k)
    #     t = 0.1
    #     x_temp = x + t * deltaX[:sizeX]; y_temp = y + t * deltaX[sizeX:]
    #     return t, x_temp, y_temp
    dfdx_old, dfdy_old, _, dgdy_old = calc_derivatives(x, y, matrixVectorProduct=True, first_order=True)
    h_old = torch.linalg.norm(dgdy_old, 2)**2 
    E_old =  f(x, y) + beta * h_old
    while True:
        x_temp = x + t * deltaX[:sizeX]; y_temp = y + t * deltaX[sizeX:]
        # dfdx, dfdy, dgdx, dgdy, dgdyy, dgdyx = calc_derivatives(x_temp, y_temp, matrixVectorProduct=False, first_order=False)
        # dh = 2 * torch.cat((dgdyx.T @ dgdy, dgdyy.T @ dgdy), 0) 
        dfdx, dfdy, dgdx, dgdy = calc_derivatives(x_temp, y_temp, matrixVectorProduct=False, first_order=True)
        h_temp = torch.linalg.norm(dgdy, 2)**2  
        E_new = f(x_temp, y_temp) + beta * h_temp
        # if E_new > E_old - 0.1 * t * (torch.linalg.norm(deltaX, 2)**2 - (beta - lam) * dh.T @ deltaX):
        if E_new > E_old + eta * t * (-torch.linalg.norm(deltaX, 2)**2 - (beta - lam) * alpha * h_old):
            t *= 0.5
        else:
            break
    if k % 100 == 0:
        print('t=', t, E_new.item(), E_old.item(), '--', h_old.item(), h_temp.item())
    return t, x_temp, y_temp

def LineSearch(deltaX, x, y, tt, feasible=True, armijo=True):
    t = 10 * tt
    gamma = 0.01
    dfdx_old, dfdy_old, _, dgdy_old = calc_derivatives(x, y, matrixVectorProduct=True, first_order=True)
    h_old = torch.linalg.norm(dgdy_old, 2)**2 - epsilon**2
    # Feasibility check
    if feasible:
        while True:
            x_temp = x + t * deltaX[:sizeX]; y_temp = y + t * deltaX[sizeX:]
            dfdx, dfdy, dgdx, dgdy = calc_derivatives(x_temp, y_temp, matrixVectorProduct=True, first_order=True)
            if torch.linalg.norm(dgdy, 2)**2 - epsilon**2 > (1 - gamma) * h_old:
            # if torch.linalg.norm(dgdy, 2)**2 > epsilon**2:
                t *= 0.5
            else:
                break 
    if armijo:
        while True:
            x_temp = x + t * deltaX[:sizeX]; y_temp = y + t * deltaX[sizeX:]
            dfdx, dfdy, dgdx, dgdy = calc_derivatives(x_temp, y_temp, matrixVectorProduct=True, first_order=True)
            # if f(x_temp, y_temp) > f(x, y):
            if f(x_temp, y_temp) > f(x, y) + 0.1 * t * torch.cat((dfdx_old, dfdy_old), 0).T @ deltaX:
            # if f(x_temp, y_temp) > f(x, y) - 0.1 * t * torch.linalg.norm(deltaX, 2)**2:
                t *= 0.5
            else:
                break
    # print('--', t)
    return t, x_temp, y_temp


def calc_derivatives(x, y, matrixVectorProduct=False, first_order=False):
    # Redefine x and y with fresh computation graph
    x = x.clone().detach().requires_grad_(True)
    y = y.clone().detach().requires_grad_(True)

    if toy_example or toy_example_nc:
        dfdx = torch.cos(c.T @ x + d.T @ y) * c + 2 *(x+y) / (torch.linalg.norm(x+y)**2 + 1)
        dfdy = torch.cos(c.T @ x + d.T @ y) * d + 2 * (x+y) / (torch.linalg.norm(x+y)**2 + 1)

        if toy_example:
            dgdx = - (H @ y - x)
            dgdy = H.T @ (H @ y - x)
            if first_order:
                return dfdx, dfdy, dgdx, dgdy

            dgdyy = H.T @ H
            dgdyx = -H

        else:
            diff = (H @ y - x).reshape(-1, )
            norm2 = torch.sum(diff**2)
            # 
            dgdx = -torch.sin(0.5 * norm2) * (x - H @ y) 
            dgdy = -torch.sin(0.5 * norm2) * H.T @ (H @ y - x)
            if first_order:
                return dfdx, dfdy, dgdx, dgdy
            # 
            dgdyy = -torch.sin(0.5 * norm2) * H.T @ H - H.T @ torch.outer(diff, diff) @ H * torch.cos(0.5 * norm2)
            dgdyx = torch.sin(0.5 * norm2)  * H.T + H.T @ torch.outer(diff, diff) * torch.cos(0.5 * norm2)
        return dfdx, dfdy, dgdx, dgdy, dgdyy, dgdyx
    

    elif toy_CS:
        dfdx = torch.zeros_like(x)
        dfdy = y - y_tilde

        s = torch.softmax(x, dim=0)
        s_reshaped = s.reshape(-1, )
        dgdx = - ((X @ ( y - X.T @ s)).T @ (torch.diag(s_reshaped) - torch.outer(s_reshaped, s_reshaped))).T
        dgdy = y - X.T @ torch.softmax(x, dim=0)

        if first_order:
            return dfdx, dfdy, dgdx, dgdy

        dgdyy = torch.eye(dimY[0]).to(y.device)
        dgdyx = -X.T @ (torch.diag(s_reshaped) - torch.outer(s_reshaped, s_reshaped))

        return dfdx, dfdy, dgdx, dgdy, dgdyy, dgdyx

    elif DHC or DHC_LS:
        x = x.reshape(dimX); y = y.reshape(dimY)
        logits_val = A_val @ y
        logist_tr = A_tr @ y

        dfdx  = torch.zeros_like(x)
        dfdy = 1 / B_val.shape[0] * (A_val.T @ (torch.softmax(logits_val, dim=1) - B_val)).reshape(-1, 1)

        loss = F.cross_entropy(logist_tr, B_tr)
        sigmoid_x = torch.sigmoid(x)
        softmax_y = torch.softmax(logist_tr, dim=1)
        
        dgdx = 1 / B_tr.shape[0] * loss * sigmoid_x * (1 - sigmoid_x)
        # dgdy =  (1 / n_train * A_tr.T @ ((softmax_y - B_tr) * sigmoid_x) + 2 * lam * y).reshape(-1, 1)

        dgdyx = 1 / B_tr.shape[0]**2 * (sigmoid_x * (1 - sigmoid_x)).T * (A_tr.T @ (softmax_y - B_tr)).reshape(-1, 1)
        # dgdyy = 1 / n_train * A_tr.T @ (softmax_y * (1 - softmax_y) * A_tr) \
        #             + 2 * lam * torch.eye(sizeY)
        y = y.reshape((sizeY, 1))
        g_val = g(x, y)
        dgdy = torch.autograd.grad(g_val, y, create_graph=True, allow_unused=True, materialize_grads=True)[0]
        if first_order:
            return dfdx, dfdy, dgdx, dgdy
        if matrixVectorProduct:
            hessian_vector_product_yy = torch.autograd.grad(dgdy, y, grad_outputs=dgdy, create_graph=True, allow_unused=True)[0]
            return dfdx, dfdy, dgdx, dgdy, hessian_vector_product_yy, dgdyx.T @ dgdy

        else:
            dgdyy = torch.zeros((sizeY, sizeY)).to(device)
            for i in range(dgdy.shape[0]):
                dgdyy[i, :] = torch.autograd.grad(dgdy[i], y, retain_graph=True, create_graph=True,
                                                allow_unused=True, materialize_grads=True)[0][:, 0]
            return dfdx, dfdy, dgdx, dgdy, dgdyy, dgdyx

    else:
        x = x.reshape(dimX); y = y.reshape(dimY)
        f_val = f(x,y)
        g_val = g(x,y)
        # make_dot(g_val.mean(), params={'y':y, 'x':x}).render("./Debug/g", format="png")

        dfdx  = torch.zeros_like(x)
        dfdy = torch.autograd.grad(f_val, y, create_graph=True, allow_unused=True, materialize_grads=True)[0]

        dgdy, dgdx = torch.autograd.grad(g_val, [y,x], create_graph=True, retain_graph=True, allow_unused=False)
        if first_order:
            return dfdx, dfdy, dgdx, dgdy
        if matrixVectorProduct:
            hessian_vector_product_yy, hessian_vector_product_yx = torch.autograd.grad(dgdy, [y, x], grad_outputs=dgdy, create_graph=True, allow_unused=False)
            return dfdx, dfdy, dgdx, dgdy, hessian_vector_product_yy, hessian_vector_product_yx
        else:
            # Initialize tensors for 2nd derivatives
            dgdyy = torch.zeros((sizeY, sizeY))
            dgdyx = torch.zeros((sizeY, sizeX))
            # Compute 2nd derivatives element-wise
            for i in range(dgdy.shape[0]):
                dgdyy[i, :] = torch.autograd.grad(dgdy[i], y, retain_graph=True, create_graph=True, allow_unused=True, materialize_grads=True)[0][:, 0]
                dgdyx[i, :] = torch.autograd.grad(dgdy[i], x, retain_graph=True, create_graph=True, allow_unused=True, materialize_grads=True)[0][:, 0]
            return dfdx, dfdy, dgdx, dgdy, dgdyy, dgdyx

def solveLL(x, device=None, threshold=1e-3):
    t0 = time.time()
    y = torch.randn((sizeY, 1), requires_grad=True, dtype=torch.float32).to(device)
    lr = 1e-1
    while True:
        if toy_example or toy_example_nc or toy_CS:
            dfdx, dfdy, dgdx, dgdy, dgdyy, dgdyx = calc_derivatives(x, y)
            with torch.no_grad():
                HessianInv = dgdyy.inverse()
                y -= lr * HessianInv @ dgdy
        else:
            dfdx, dfdy, dgdx, dgdy = calc_derivatives(x, y, matrixVectorProduct=False, first_order=True)
            if torch.linalg.norm(dgdy, 2) > 5e-2:
                lr = 1
            else:
                lr = 1e-1
                print(torch.linalg.norm(dgdy, 2))
            with torch.no_grad():
                y -= lr * dgdy
        if torch.linalg.norm(dgdy, 2) < threshold:
                    break
        
    print('LL error: ', torch.linalg.norm(dgdy, 2), 'Time elapsed:', time.time() - t0, '\n')
    return y, dgdy

# Define the system of ODEs
def system(t, variables):
    x, y = variables[:sizeX], variables[sizeX:]
    global dxdt #Because its previous value is required in ProjectMethod 1
    progress_bar.update(1)

    if (method == 'InversionFree') and not toy_example:
        dfdx, dfdy, dgdx, dgdy, hessian_vector_product, dgdyx = calc_derivatives(x, y, matrixVectorProduct=True)
    else:
        dfdx, dfdy, dgdx, dgdy, dgdyy, dgdyx = calc_derivatives(x, y, matrixVectorProduct=False)
        hessian_vector_product = dgdyy.T @ dgdy

    with torch.no_grad():    
        if method == 'InversionFree':
            a = 2 * dgdyx.T @ dgdy
            b = 2 * hessian_vector_product
            c = -alpha * (torch.linalg.norm(dgdy, 2)**2 - epsilon**2)
            ab = torch.cat((a, b), 0)

            tot = torch.cat((dfdx, dfdy), 0)
            d = ab * torch.maximum(torch.Tensor([0]).to(device), -ab.T @ tot - c) / (torch.linalg.norm(a, 2)**2 + torch.linalg.norm(b, 2)**2)
            dtotdt = -tot - d
            dxdt = dtotdt[:sizeX]; dydt = dtotdt[sizeX:]
            
            # if torch.linalg.norm(dgdy, 2) > epsilon and not torch.allclose(torch.linalg.norm(dgdy, 2), torch.Tensor([epsilon])):
            #     print('t=',t, '-', torch.linalg.norm(dgdy, 2), epsilon)

        elif method == 'NewSecondOrder':
            a = dgdyx @ dgdyx.T
            b = dgdyy @ dgdyy.T
            c = -alpha * dgdy + dgdyx @ dfdx + dgdyy @ dfdy
            lam = -torch.inverse(a + b) @ c

            dxdt = -dfdx - dgdyx.T @ lam
            dydt = -dfdy - dgdyy.T @ lam

        elif method == 'SecondOrder':
            mu = 0.05
            gHessianInv = dgdyy.inverse()
            dxdt = -dfdx + dgdyx.T @ gHessianInv @ dfdy
            dydt = -gHessianInv @ (mu * dgdy + dgdyx @ dxdt)  
        # 
        elif method == 'STABLE':
            mu1 = 1
            mu2 = 0.5
            gHessianInv = dgdyy.inverse()
            dxdt = mu1 * (-dfdx + dgdyx.T @ gHessianInv @ dfdy)
            dydt = -mu2 * dgdy - gHessianInv @ dgdyx @ dxdt 
        
        else:
            raise ValueError('Invalid method')
    
    return torch.cat((dxdt, dydt), 0)

def IFDT(x, y, alpha, alpha_step=0.1, K=100, beta=1, mode='RXGD', lossS=None, device=None):
    global A_tr, B_tr, A_val, B_val, A_test, B_test, calc_derivatives, epsilon
    zero = torch.Tensor([0]).to(device)

    lossF, lossG, lossF2 = lossS
    train_accuracy, val_accuracy, test_accuracy = [], [], []
    train_loss, val_loss, test_loss = [], [], []
    tt = 1
    beta = torch.Tensor([beta]).to(device)
    for k in tqdm(range(K)):
        # Calculate derivatives for current x and y
        if toy_example or toy_example_nc or toy_CS:
            dfdx, dfdy, dgdx, dgdy, dgdyy, dgdyx = calc_derivatives(x, y)
            hvp_yy = dgdyy.T @ dgdy
            hvp_yx = dgdyx.T @ dgdy
        else:
            dfdx, dfdy, dgdx, dgdy, hvp_yy, hvp_yx = calc_derivatives(x, y, matrixVectorProduct=True)
        
        a = 2 * hvp_yx
        b = 2 * hvp_yy
        c = -alpha * (torch.linalg.norm(dgdy, 2)**2 - epsilon**2)
        tot = torch.cat((dfdx, dfdy), 0)
        dh = torch.cat((a, b), 0)    
        if mode  == 'QCQP':
            w = 0.01
            alpha = K**(-1/3) // 100
            c = -alpha * -(torch.linalg.norm(dgdy, 2)**2 - epsilon**2)**2
            with torch.no_grad():
                dtotdt, lam = cvxpy_QCQP(tot, dh, c, w)
                dtotdt = torch.Tensor(dtotdt).to(device)
                lam = torch.Tensor(lam).to(device)
                # rad = torch.sqrt(torch.linalg.norm(dh / (2*w))**2 + c / w)
                # term = torch.linalg.norm(-tot + dh / (2 * w))
                # if term > rad:
                #     dtotdt = -dh/(2*w) + rad * (-tot + dh / (2 * w)) / term
                #     assert torch.allclose(torch.linalg.norm(dtotdt + dh/(2*w)), rad, atol=1e-6)
                # else:
                #     dtotdt = -tot

                # # Checking the step!
                # if False:
                #     dtotdt_cvxpy = cvxpy_QCQP(tot, dh, c, w)
                #     assert torch.allclose(dtotdt.detach().cpu().numpy(), dtotdt_cvxpy, atol=1e-5)
                # else:
                #     assert (dh.T @ dtotdt < c - w * torch.linalg.norm(dtotdt, 2)**2) or \
                #             torch.allclose(dh.T @ dtotdt, c - w * torch.linalg.norm(dtotdt, 2)**2, atol=1e-1)
                #     if torch.linalg.norm(dgdy, 2) < epsilon:
                #         assert tot.T @ dtotdt <= 0
                #     else:
                #         assert dh.T @ dtotdt <= 0
            tt, x, y = LineSearch(dtotdt, x, y, tt=0.1)
            
        elif mode == 'MOGD':
            if k % 1000 == 0:
                print('==', k, beta)
                beta *= 2
            # c = -alpha * torch.linalg.norm(dgdy, 2)**2
            c = -alpha * torch.linalg.norm(dgdy, 2)**2
            with torch.no_grad():
                dtotdt, lam = cvxpy_MOGD(tot, dh, c, beta)
                # dtotdt, lam = cvxpy_QCQP(tot, dh, c, w)
                dtotdt = torch.Tensor(dtotdt).to(device)
                lam = torch.Tensor(lam).to(device)
                # 
                # dtotdt = -tot - beta * dh
                lam = 0
                if k % 100 == 0:
                    print(torch.linalg.norm(dtotdt, 2), lam)
                if beta < lam:
                    print('beta < lam', beta, lam)
                    beta = lam
            tt, x, y = LineSearch_merit(dtotdt, x, y, tt=10, lam=lam, beta=beta, alpha=alpha, k=k)



        elif mode == 'RXGD':
            with torch.no_grad():
                d = dh * torch.maximum(zero, -dh.T @ tot - c) / (torch.linalg.norm(dh, 2)**2)
                dtotdt = -tot - d
                dxdt = dtotdt[:sizeX]; dydt = dtotdt[sizeX:]
                with torch.no_grad():
                    x = x + alpha_step * dxdt
                    y = y + alpha_step * dydt


        elif 'Ours' in mode:
            with torch.no_grad():
                if mode[-1] == '1':
                    # K^-1/3 ~ 0.001
                    if toy_example or toy_example_nc: alpha_K = K**(-1/3); alpha_step_K = K**(-1/3)
                    elif toy_CS: alpha_K = 0.1 * K**(-1/3); alpha_step_K = 0.1 * K**(-1/3)
                    else: alpha_K = 5 * K**(-1/3); alpha_step_K = 5 * K**(-1/3)
                    cprime = alpha_K * (torch.linalg.norm(dh, 2)**2)
                elif mode[-1] == '2':
                    # K^-1/3 ~ 0.001, K^-2/3 ~ 0.0005
                    alpha_K = 10 * K**(-1/3); alpha_step_K = 10 * K**(-2/3)
                    cprime = alpha_K * (torch.linalg.norm(dh, 2) * torch.linalg.norm(dgdy, 2))
                else:
                    raise ValueError('Invalid mode')

                lam = torch.maximum(zero, -tot.T @ dh + cprime) / torch.linalg.norm(dh, 2)**2
                dtotdt = -tot - lam * dh
                dxdt = dtotdt[:sizeX]; dydt = dtotdt[sizeX:]
                # if p == 0:
                #     if beta < lam:
                #         print('beta < lam', beta, lam)
                #         beta = lam
                #     tt, x, y = LineSearch_merit(dtotdt, x, y, tt=0.5, lam=lam, beta=beta, alpha=alpha)
                # else:
                dxdt = dtotdt[:sizeX]; dydt = dtotdt[sizeX:]
                x = x + alpha_step_K * dxdt
                y = y + alpha_step_K * dydt


        else:
            raise NotImplementedError('Invalid mode')
            # with torch.no_grad():
            #     cprime = -alpha * (torch.linalg.norm(dh, 2) * torch.linalg.norm(dgdy, 2) - epsilon**2)
            #     lam = (-tot.T @ dh + torch.linalg.norm(dh)**2) / (torch.linalg.norm(tot)**2 + torch.linalg.norm(dh)**2 - 2 * tot.T @ dh); feas=False
            #     # lam = (-tot.T @ dh + torch.linalg.norm(dh)**2 + c) / (torch.linalg.norm(tot)**2 + torch.linalg.norm(dh)**2 - 2 * tot.T @ dh); feas=True
            #     # lam = (-tot.T @ dh + torch.linalg.norm(dh)**2 + cprime) / (torch.linalg.norm(tot)**2 + torch.linalg.norm(dh)**2 - 2 * tot.T @ dh); feas=True
            #     # 
            #     # print(lam)
            #     if lam < 0.0:
            #         deltaX = -dh
            #     elif lam > 1:
            #         deltaX = -tot
            #     else:
            #         deltaX = - lam * tot - (1 - lam) * dh
            #         # print(lam, torch.allclose(tot.T @ deltaX, dh.T @ deltaX), torch.linalg.norm(deltaX))
            # t, x, y = LineSearch(deltaX, x, y, dfdx_old=dfdx, dfdy_old=dfdy, feasible=feas, armijo=True)

        # Compute and store losses  
        term1, term2, term3 = calculate_losses(torch.cat((x, y), 0), f, sizeX, sizeY, calc_derivatives, testID=args.testID, deltaX=dtotdt)
        lossF.append(term1); lossG.append(term2); lossF2.append(term3)

        if DHC or DHC_LS or NN:
            pars = (A_tr, B_tr, A_val, B_val, A_test, B_test)
            train_accuracy, val_accuracy, test_accuracy, train_loss, val_loss, test_loss = add_loss(y, train_accuracy, val_accuracy, test_accuracy, 
                                                                                                    train_loss, val_loss, test_loss, pars, dimY, arch)
        

    # Convert lists of losses to tensors for easy analysis
    return np.array(lossF), np.array(lossG), np.array(lossF2), (train_accuracy, val_accuracy, test_accuracy), (train_loss, val_loss, test_loss)





def TTSA(x, y, alpha=0.1, beta_step=0.1, K=100):
    global A_tr, B_tr, A_val, B_val, A_test, B_test, toy_example, calc_derivatives

    lossF, lossG, lossF2 = [], [], []
    train_accuracy, val_accuracy, test_accuracy = [], [], []
    train_loss, val_loss, test_loss = [], [], []
    # x.requires_grad = False
    for k in tqdm(range(K)):
        # Calculate derivatives for current x and y
        dfdx, dfdy, dgdx, dgdy, dgdyy, dgdyx = calc_derivatives(x, y)

        # Update y
        y = y - beta_step / (1+k)**(3/5) * dgdy

        # Recalculate derivatives after updating y
        dfdx, dfdy, dgdx, dgdy, dgdyy, dgdyx = calc_derivatives(x, y)

        # Update x
        with torch.no_grad():
            x = x - alpha / (1+k)**(2/5) * (dfdx - dgdyx.T @ dgdyy.inverse() @ dfdy)

        # Compute and store losses (no gradient tracking)
        term1, term2, term3 = calculate_losses(torch.cat((x, y), 0), f, sizeX, sizeY, calc_derivatives, testID=args.testID)
        lossF.append(term1); lossG.append(term2); lossF2.append(term3)
        if DHC or DHC_LS or NN:
            pars = (A_tr, B_tr, A_val, B_val, A_test, B_test)
            train_accuracy, val_accuracy, test_accuracy, train_loss, val_loss, test_loss = add_loss(y, train_accuracy, val_accuracy, test_accuracy, 
                                                                                                    train_loss, val_loss, test_loss, pars, dimY, arch)

    # Convert lists of losses to tensors for easy analysis
    return np.array(lossF), np.array(lossG), np.array(lossF2), (train_accuracy, val_accuracy, test_accuracy), (train_loss, val_loss, test_loss)



def AIDBio(x, y0, alpha_step=0.1, beta_step=0.1, K=10, D=10, device=None):
    global A_tr, B_tr, A_val, B_val, A_test, B_test, toy_example
    y = y0
    lossF, lossG, lossF2 = [], [], []
    train_accuracy, val_accuracy, test_accuracy = [], [], []
    train_loss, val_loss, test_loss = [], [], []
    nu = torch.zeros_like(y0)
    # x.requires_grad = False
    for k in tqdm(range(K)):
        for t in range(D):
            dfdx, dfdy, dgdx, dgdy, dgdyy, dgdyx = calc_derivatives(x, y)
            y = y - alpha_step * dgdy
            
            term1, term2, term3 = calculate_losses(torch.cat((x, y), 0), f, sizeX, sizeY, calc_derivatives, testID=args.testID, deltaX=dgdy)
            if toy_example_nc or toy_CS or NN:
                lossF.append(term1); lossG.append(term2); lossF2.append(lossF2[-1] if len(lossF2) > 0 else 0)
            else:
                lossF.append(term1); lossG.append(term2); lossF2.append(term3)
            if DHC or DHC_LS or NN:
                pars = (A_tr, B_tr, A_val, B_val, A_test, B_test)
                train_accuracy, val_accuracy, test_accuracy, train_loss, val_loss, test_loss = add_loss(y, train_accuracy, val_accuracy, test_accuracy, 
                                                                                                        train_loss, val_loss, test_loss, pars, dimY, arch)

        dfdx, dfdy, dgdx, dgdy, dgdyy, dgdyx = calc_derivatives(x, y)
        with torch.no_grad():
            if False:
                nu =  dgdyy.inverse() @ dfdy
            else:
                nu = conjugate_gradient(dgdyy.detach(), dfdy.detach(), nu.detach(), 10)
            x = x - beta_step * (dfdx - dgdyx.T @ nu)
        
        term1, term2, term3 = calculate_losses(torch.cat((x, y), 0), f, sizeX, sizeY, calc_derivatives, testID=args.testID,
                                                deltaX=(dfdx - dgdyx.T @ nu))
        lossF.append(term1); lossG.append(term2); lossF2.append(term3)
        if DHC or DHC_LS or NN:
            pars = (A_tr, B_tr, A_val, B_val, A_test, B_test)
            train_accuracy, val_accuracy, test_accuracy, train_loss, val_loss, test_loss = add_loss(y, train_accuracy, val_accuracy, test_accuracy, 
                                                                                                        train_loss, val_loss, test_loss, pars, dimY, arch)
        
        
    return np.array(lossF), np.array(lossG), np.array(lossF2), (train_accuracy, val_accuracy, test_accuracy), (train_loss, val_loss, test_loss)

def BOME(x, y0, alpha_step, K, T, device=None):
    global A_tr, B_tr, A_val, B_val, A_test, B_test
    y = y0
    lossF, lossG, lossF2 = [], [], []
    train_accuracy, val_accuracy, test_accuracy = [], [], []
    train_loss, val_loss, test_loss = [], [], []

    for k in tqdm(range(K)):
        y_gd = y
        for t in range(T):
            # inner loop
            dfdx, dfdy, dgdx, dgdy = calc_derivatives(x, y_gd, first_order=True)
            y_gd = y_gd - alpha_step * dgdy

            term1, term2, term3 = calculate_losses(torch.cat((x, y), 0), f, sizeX, sizeY, calc_derivatives, testID=args.testID, deltaX=dgdy)
            if toy_example_nc or toy_CS or NN:
                lossF.append(term1); lossG.append(term2); lossF2.append(lossF2[-1] if len(lossF2) > 0 else 0)
            else:
                lossF.append(term1); lossG.append(term2); lossF2.append(term3)
            if DHC or DHC_LS or NN:
                pars = (A_tr, B_tr, A_val, B_val, A_test, B_test)
                train_accuracy, val_accuracy, test_accuracy, train_loss, val_loss, test_loss = add_loss(y, train_accuracy, val_accuracy, test_accuracy, 
                                                                                                        train_loss, val_loss, test_loss, pars, dimY, arch)
                
        # outer-loop
        dfdx, dfdy, dgdx, dgdy = calc_derivatives(x, y, first_order=True)
        _, _, dgdx2, dgdy2 = calc_derivatives(x, y_gd, first_order=True)
        dqdx = dgdx - dgdx2
        dqdy = dgdy
        with torch.no_grad():
            phi = torch.sum(torch.cat((dqdx, dqdy), 0)**2)
            lam = torch.max(torch.Tensor([0]).to(device), 0.1 * phi - (dqdx.T @ dfdx + dqdy.T @ dfdy)) / phi
            x = x - alpha_step * (dfdx + lam * dqdx)
            y = y - alpha_step * (dfdy + lam * dqdy)
        # y.requires_grad = True
        term1, term2, term3 = calculate_losses(torch.cat((x, y), 0), f, sizeX, sizeY, calc_derivatives, testID=args.testID, 
                                               deltaX=torch.cat(((dfdx + lam * dqdx), (dfdy + lam * dqdy)), 0))
        lossF.append(term1); lossG.append(term2); lossF2.append(term3)
        if DHC or DHC_LS or NN:
            pars = (A_tr, B_tr, A_val, B_val, A_test, B_test)
            train_accuracy, val_accuracy, test_accuracy, train_loss, val_loss, test_loss = add_loss(y, train_accuracy, val_accuracy, test_accuracy, 
                                                                                                    train_loss, val_loss, test_loss, pars, dimY, arch)
    
    return np.array(lossF), np.array(lossG), np.array(lossF2), (train_accuracy, val_accuracy, test_accuracy), (train_loss, val_loss, test_loss)

def VPBGD(x, y0, alpha_step, K, T, device=None):
    global A_tr, B_tr, A_val, B_val, A_test, B_test
    y = y0
    lossF, lossG, lossF2 = [], [], []
    train_accuracy, val_accuracy, test_accuracy = [], [], []
    train_loss, val_loss, test_loss = [], [], []

    gamma_init = 0 
    gamma_max = 0.2
    gamma_steps = K * 3 // 4
    gamma = gamma_init
    for k in tqdm(range(K)):
        gamma = min(gamma_max, gamma + gamma_max / gamma_steps)
        y_gd = y
        for t in range(T):
            dfdx, dfdy, dgdx, dgdy = calc_derivatives(x, y_gd, first_order=True)
            y_gd = y_gd - alpha_step * dgdy

            term1, term2, term3 = calculate_losses(torch.cat((x, y_gd), 0), f, sizeX, sizeY, calc_derivatives, testID=args.testID, deltaX=dgdy)
            if toy_example_nc or toy_CS or NN:
                lossF.append(term1); lossG.append(term2); lossF2.append(lossF2[-1] if len(lossF2) > 0 else 0)
            else:
                lossF.append(term1); lossG.append(term2); lossF2.append(term3)
            if DHC or DHC_LS or NN:
                pars = (A_tr, B_tr, A_val, B_val, A_test, B_test)
                train_accuracy, val_accuracy, test_accuracy, train_loss, val_loss, test_loss = add_loss(y, train_accuracy, val_accuracy, test_accuracy, 
                                                                                                        train_loss, val_loss, test_loss, pars, dimY, arch)
        dfdx, dfdy, dgdx, dgdy = calc_derivatives(x, y, first_order=True)
        _, _, dgdx2, dgdy2 = calc_derivatives(x, y_gd, first_order=True)
        with torch.no_grad():
            x = x - alpha_step * (dfdx + gamma * (dgdx - dgdx2))
            y = y - alpha_step * (dfdy + gamma * dgdy)

        term1, term2, term3 = calculate_losses(torch.cat((x, y), 0), f, sizeX, sizeY, calc_derivatives, testID=args.testID,
                                                deltaX=torch.cat(((dfdx + gamma * (dgdx - dgdx2)), (dfdy + gamma * dgdy)), 0))
        lossF.append(term1); lossG.append(term2); lossF2.append(term3)
        if DHC or DHC_LS or NN:
            pars = (A_tr, B_tr, A_val, B_val, A_test, B_test)
            train_accuracy, val_accuracy, test_accuracy, train_loss, val_loss, test_loss = add_loss(y, train_accuracy, val_accuracy, test_accuracy, 
                                                                                                    train_loss, val_loss, test_loss, pars, dimY, arch)
        
    return np.array(lossF), np.array(lossG), np.array(lossF2), (train_accuracy, val_accuracy, test_accuracy), (train_loss, val_loss, test_loss)
 


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='.',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--testID', type=int, default=0)
    parser.add_argument('--scenarioID', type=int, default=0)
    args = parser.parse_args()
    # 
    toy_example = (args.testID == 0)
    toy_example_nc = (args.testID == 1)
    toy_CS = (args.testID == 2)
    DHC = (args.testID == 3)
    DHC_LS = (args.testID == 4)
    NN = (args.testID == 5)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    plt.rcParams.update({
    'font.size': 16,          # General font size
    'xtick.labelsize': 16,    # Tick label size for x-axis
    'ytick.labelsize': 16,    # Tick label size for y-axis
    'axes.labelsize': 16,      # Font size for axis labels,
    'pdf.fonttype': 42,
    'ps.fonttype': 42
})


    scenarios = scenario_setup(args.scenarioID)

    if toy_example or toy_example_nc:
        f, g, c, d, A, H, dimX, dimY = load_setup(args.testID, device=device)
        fig1, ax1, fig11, ax11, fig2, ax2 = get_axs(toy_example or toy_example_nc)
    elif toy_CS:
        f, g, y_tilde, X, dimX, dimY = load_setup(args.testID, device=device)
        fig1, ax1, fig11, ax11, fig2, ax2 = get_axs(toy_CS=toy_CS)
    else:
        f, g, A_tr, B_tr, A_val, B_val, A_test, B_test, dimX, dimY, arch = load_setup(args.testID, p=scenarios[0][3], device=device)
        fig1, ax1, fig11, ax11, fig2, ax2, fig3, ax3, fig4, ax4 = get_axs(toy_example)
    
    sizeX = dimX[0] * dimX[1]; sizeY = dimY[0] * dimY[1]

    torch.manual_seed(0); np.random.seed(0)
    if toy_example or toy_example_nc:
        x0 = torch.randn((sizeX, 1), requires_grad=False, dtype=torch.float32).to(device)
        t = torch.linspace(0, 200, 5000)
    elif toy_CS:
        x0 = torch.randn((sizeX, 1), requires_grad=False, dtype=torch.float32).to(device)
        t = torch.linspace(0, 20, 25000)
    elif DHC or DHC_LS:
        x0 = torch.zeros((sizeX, 1), requires_grad=False, dtype=torch.float32).to(device)
        t = torch.linspace(0, 50, 10)
    elif NN:
        x0 = torch.randn((sizeX, 1), requires_grad=True, dtype=torch.float32).to(device)
        t = torch.linspace(0, 0, 500)


    if toy_example or toy_example_nc or toy_CS:# and 'InversionFree' in [method for method, _, _, _ in scenarios]:
        y0, dgdy = solveLL(x0, device=device)
    else:
        y0 = torch.randn((sizeY, 1), requires_grad=True, dtype=torch.float32).to(device)


    for (method, alpha, epsilon, p, mode, beta) in scenarios:  
        if DHC or DHC_LS or NN: f, g, A_tr, B_tr, A_val, B_val, A_test, B_test, dimX, dimY, arch = load_setup(args.testID, p=p, device=device)
        lossF, lossG, lossF2 = [], [], []
        acc, loss = None, None

        if args.scenarioID == 4 or args.scenarioID == 5:
            if p == 0: t = torch.linspace(0, 200, 20000)
            elif p == -1: t = torch.linspace(0, 20, 15000)
            else: t = torch.linspace(0, 20, 10000)  
        
        # -----------------------------------------------------------------
        print('-- Method:', method, 'Alpha:', alpha, 'Epsilon:', epsilon, 'P:', p, 'mode:', mode)
        if toy_example or toy_example_nc: alpha_step = 0.01
        elif toy_CS: alpha_step = 0.5
        else: alpha_step = 1

        t1 = time.time()
        if method in ['InversionFree', 'NewSecondOrder', 'SecondOrder', 'STABLE']:
            initial_conditions = torch.cat((x0, y0), 0)
            progress_bar = tqdm(total= 4 * len(t))
            solution = torchdiffeq.odeint(system, initial_conditions, t, method='rk4')
            progress_bar.close()
            tt = t
            train_accuracy, val_accuracy, test_accuracy = [], [], []
            train_loss, val_loss, test_loss = [], [], []
            for i in range(len(solution)):
                dfdx, dfdy, dgdx, dgdy, dgdyy, dgdyx = calc_derivatives(solution[i, :sizeX], solution[i, sizeX:])
                with torch.no_grad():
                    lossF.append(f(solution[i, :sizeX], solution[i, sizeX:]).detach().cpu().numpy().reshape(-1))
                    lossG.append(torch.linalg.norm(dgdy).detach().cpu().numpy())
                    lossF2.append(torch.linalg.norm(dfdx - dgdyx.T @ dgdyy.inverse() @ dfdy).detach().cpu().numpy())
                if DHC or DHC_LS or NN:
                    pars = (A_tr, B_tr, A_val, B_val, A_test, B_test)
                    train_accuracy, val_accuracy, test_accuracy, train_loss, val_loss, test_loss =\
                          add_loss(solution[i, sizeX:], train_accuracy, val_accuracy, test_accuracy, train_loss, val_loss, test_loss, pars, dimY, arch)
                                                                      
            acc = (train_accuracy, val_accuracy, test_accuracy); loss = (train_loss, val_loss, test_loss)
        elif method == 'AIDBio':
            lossF, lossG, lossF2, acc, loss = AIDBio(x0, y0, alpha_step=alpha_step, beta_step=0.01, K=np.maximum(1, int(len(t) * 4 / 11)), D=10
                                                    , device=device)
            tt = torch.linspace(0, t[-1], lossF.shape[0])
        elif method == 'BOME':
            if toy_example_nc: alpha_step *= 0.5
            elif toy_CS: alpha_step *= 0.1
            lossF, lossG, lossF2, acc, loss = BOME(x0, y0, alpha_step=alpha_step, K=np.maximum(1, int(len(t) * 4 / 11)), T=10, device=device)
            tt = torch.linspace(0, t[-1], lossF.shape[0])
        elif method == 'IFDT':
            lossF, lossG, lossF2, acc, loss = IFDT(x0, y0, alpha, alpha_step=alpha_step, K=np.maximum(1, int(len(t) * 4)), mode=mode,
                                                    lossS=(lossF, lossG, lossF2), beta=beta, device=device)
            tt = torch.linspace(0, t[-1], lossF.shape[0])
        elif method == 'VPBGD':
            lossF, lossG, lossF2, acc, loss = VPBGD(x0, y0, alpha_step=alpha_step / 10, K=np.maximum(1, int(len(t) * 4) // 11), T=10, device=device)
            tt = torch.linspace(0, t[-1], lossF.shape[0])
        # elif method == 'TTSA':
        #     lossF, lossG, lossF2, acc, loss = TTSA(x, y0, K=np.maximum(1, int(len(t) * 2)))
        #     tt = torch.linspace(0, t[-1], lossF.shape[0])
        else:
            raise ValueError('Invalid method')
        print('Time taken:', time.time() - t1)

        
        with torch.no_grad():
            flag_method, flag_alpha, flag_epsilon, flag_p = 1, 1, 1, 1
            try:
                if args.scenarioID == 4 or args.scenarioID == 5: pass
                if scenarios[0][0] == scenarios[1][0]: flag_method = 0
                if scenarios[0][1] == scenarios[1][1]: flag_alpha = 0
                if scenarios[0][2] == scenarios[1][2]: flag_epsilon = 0
                if scenarios[0][3] == scenarios[1][3]: flag_p = 0
            except:
                flag_alpha, flag_epsilon, flag_p = 0, 0, 0

            if flag_alpha: strLabel = method + r': $\alpha$= ' + str(alpha)
            elif flag_epsilon: strLabel = method + r': $\varepsilon$= ' + str(epsilon)
            elif DHC or DHC_LS or NN: 
                if method == 'IFDT': 
                    if mode == 'Ours1': strLabel = 'Theorem 4.1'
                    elif mode == 'Ours2': strLabel = 'Theorem 4.5'
                    strLabel +=  r': p= ' + str(p)
                else: strLabel = method + r': p= ' + str(p)
            elif args.scenarioID == 4 or args.scenarioID == 5: strLabel = r'K = ' + str(len(tt) // 10**3) + r" $\times 10^3$"

            else:
                if method != 'IFDT': strLabel = method
                else:
                    if mode == 'Ours1': strLabel = 'Theorem 4.1'
                    elif mode == 'Ours2': strLabel = 'Theorem 4.5'
                    else: strLabel = mode

            print('Number of Gradient Calculations:', len(tt), '\n')
            if 'InversionFree' not in [method for method, _, _, _, _, _ in scenarios] and \
                'SecondOrder' not in [method for method, _, _, _, _, _ in scenarios]:
                tt = range(len(lossF))
                t_label = 'iterations'
            else:
                t_label = 'time'
            
            if 'MOGD' in [mode for _, _, _, _, mode, _ in scenarios]:
                try:
                    axm  # Try accessing axm
                except NameError:
                    fign, axn = plt.subplots(1, 1, figsize=(8, 6)) 
                    fign, axm = plt.subplots(1, 1, figsize=(8, 6)) 

                axn.plot(tt, np.cumsum(lossF2) / np.arange(1, len(lossF2) + 1), label='Average Step: '+ method)
                axn.set_yscale('log')
                axn.legend()
                axm.plot(tt, (lossF + 1) + beta * lossG**2, label='Merit Function: ' + method)
                axm.set_yscale('log')
                axm.legend()
            ax1.plot(tt, lossF, label= (strLabel))
            ax11.plot(tt, lossF2, label=(strLabel))

            
            # -----------------------------------------------------
            if len(tt) > 10**4:
                ax11.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x / 1e3:.0f}"))
                ax2.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x / 1e3:.0f}"))  
                t_label += r" $\times 10^3$"

            ax2.plot(tt, lossG, label=(strLabel))
            # if not (toy_example_nc or toy_CS):
            # ax2.plot(tt, [epsilon] * len(tt), 'r--')

            if DHC or DHC_LS or NN: 
                # Plotting accuracy
                print('Train Accuracy:', acc[0][-1].item(), 'Validation Accuracy:', acc[1][-1].item(), 'Test Accuracy:', acc[2][-1].item(), '\n')
                ax3.plot(tt, acc[2], label=(strLabel))
                ax3.set_xlabel(t_label)
                ax3.set_ylabel('Test Accuracy')
                ax3.legend()

                if False:
                    # Plotting loss
                    try:
                        axtr  # Try accessing axm
                    except NameError:
                        fign, axtr = plt.subplots(1, 1, figsize=(8, 6))
                    axtr.plot(tt, loss[0], label=(strLabel))
                    axtr.set_xlabel(t_label)
                    axtr.set_ylabel('Training Loss')
                    axtr.legend() 
                    fign.savefig('Result/' + ('toy_example/' if toy_example or toy_example_nc or toy_CS else 'DHC/') + 'Loss_tr' + '.pdf', dpi=300,
                                bbox_inches='tight', pad_inches=0.1)

                ax4.plot(tt, loss[1], label=(strLabel))
                ax4.set_xlabel(t_label)
                ax4.set_ylabel('Validation Loss')
                ax4.legend()

                fig3.savefig('Result/' + ('toy_example/' if toy_example or toy_example_nc or toy_CS else 'DHC/') + 'Acc' + '.pdf', dpi=300,
                             bbox_inches='tight', pad_inches=0.1)
                fig4.savefig('Result/' + ('toy_example/' if toy_example or toy_example_nc or toy_CS else 'DHC/') + 'Loss' + '.pdf', dpi=300,
                             bbox_inches='tight', pad_inches=0.1)
                
            ax1.legend()
            ax1.set_xlabel(t_label)
            ax1.set_ylabel('f(x,y)')

            ax11.legend()
            ax11.set_xlabel(t_label)
            if toy_example_nc or NN:
                ax11.set_ylabel(r'$\|\Delta x\|$')
            else:
                ax11.set_ylabel(r'$\|F(x,y)\|$')
            ax11.set_yscale('log')

            ax2.legend()
            ax2.set_xlabel(t_label)
            ax2.set_ylabel(r'$\|\nabla g(x,y)\|$')


            # plt.tight_layout()
            scenarioItems = ['method', 'alpha', 'epsilon']
            item = 2 * flag_epsilon + 1 * flag_alpha + 0 * flag_method
            fig1.savefig('Result/' + ('toy_example/' if toy_example or toy_example_nc or toy_CS else 'DHC/') + scenarioItems[item] + ':' + str(scenarios[0][item]) + '-up1' + '.pdf',
                          dpi=300, bbox_inches='tight', pad_inches=0.1)
            fig11.savefig('Result/' + ('toy_example/' if toy_example or toy_example_nc or toy_CS else 'DHC/') + scenarioItems[item] + ':' + str(scenarios[0][item]) + '-up2' + '.pdf',
                           dpi=300, bbox_inches='tight', pad_inches=0.1)
            fig2.savefig('Result/' + ('toy_example/' if toy_example or toy_example_nc or toy_CS else 'DHC/') + scenarioItems[item] + ':' + str(scenarios[0][item]) + '-low' + '.pdf',
                          dpi=300, bbox_inches='tight', pad_inches=0.1)


    plt.close(fig1)
    fig11.show()
    # plt.pause(0)
    plt.show()
    plt.close('all')