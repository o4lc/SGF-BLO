import numpy as np
import torch
import torch.nn.functional as F
import time
from tqdm import tqdm

from setup import load_setup
from utilities import add_loss, calculate_losses, cvxpy_QCQP, cvxpy_MOGD, conjugate_gradient

class BilevelSolver:
    def __init__(self, testID, scenarioID, use_time=False, device='cpu'):
        self.testID = testID
        self.toy_example = (testID == 0)
        self.toy_example_nc = (testID == 1)
        self.toy_example_cons = (testID == 2)
        self.toy_CS = (testID == 3)
        self.DHC = (testID == 4)
        self.DHC_LS = (testID == 5)
        self.NN = (testID == 6)

        self.device = device
        self.scenarioID = scenarioID

        self.num_grad_calc = None
        self.use_time = use_time

        if self.use_time:
            if self.toy_example: self.time_limit = 60  # seconds
            elif self.DHC: self.time_limit = 250  # seconds
            elif self.NN : self.time_limit = 1000  # seconds
            else: raise NotImplementedError('Time limit not set for this problem')

        # scenarios = scenario_setup(scenarioID)
        

    def load_setup(self, p=-1):
        if self.toy_example or self.toy_example_nc or self.toy_example_cons:
            self.f, self.g, self.c, self.d, self.A, self.H, self.dimX, self.dimY = load_setup(self.testID, device=self.device)
        elif self.toy_CS:
            self.f, self.g, self.y_tilde, self.X, self.dimX, self.dimY = load_setup(self.testID, device=self.device)
        else:
            self.f, self.g, self.A_tr, self.B_tr, self.A_val, self.B_val, self.A_test,\
                  self.B_test, self.dimX, self.dimY, self.arch = load_setup(self.testID, p=p, device=self.device)
        
        self.sizeX = self.dimX[0] * self.dimX[1];
        self.sizeY = self.dimY[0] * self.dimY[1]

    def solveLL(self, x):
        t0 = time.time()
        y = torch.randn((self.sizeY, 1), dtype=torch.float32, device=self.device)
        y.requires_grad_(True)  # Explicitly ensure y is a leaf tensor with requires_grad
        # Initialize Adam optimizer
        optimizer = torch.optim.Adam([y])
        while True:
            if self.toy_example or self.toy_example_nc or self.toy_CS:
                dfdx, dfdy, dgdx, dgdy, dgdyy, dgdyx = self.calc_derivatives(x, y)
                with torch.no_grad():
                    HessianInv = dgdyy.inverse()
                    y -= 0.1 * HessianInv @ dgdy
                if torch.linalg.norm(dgdy, 2) < 1e-3:
                    break
            else:
                dfdx, dfdy, dgdx, dgdy = self.calc_derivatives(x, y, matrixVectorProduct=False, first_order=True)
                with torch.no_grad():
                    optimizer.zero_grad()  # Zero previous gradients
                    y.grad = dgdy.reshape(y.shape)  # Set the gradient manually for Adam
                    optimizer.step()  # Perform an optimization step
                if torch.linalg.norm(dgdy, 2) < 5e-2:
                    break
                # print('LL error: ', torch.linalg.norm(dgdy, 2), 'Time elapsed:', time.time() - t0)
            
        print('LL error: ', torch.linalg.norm(dgdy, 2), 'Time elapsed:', time.time() - t0, '\n')
        return y, dgdy
    
    def solveLL_constrained(self, x):
        t0 = time.time()
        y = torch.randn((self.sizeY, 1), dtype=torch.float32, device=self.device)
        y = torch.clamp(y, min=0.01)  # Apply the constraint
        y.requires_grad_(True)  # Explicitly ensure y is a leaf tensor with requires_grad
        # Initialize Adam optimizer
        optimizer = torch.optim.Adam([y])
        while True:
            _, _, _, dgdy = self.calc_derivatives(x, y, matrixVectorProduct=False, first_order=True)
            # dgdy = self.H.T @ (self.H @ y - x)
            with torch.no_grad():
                optimizer.zero_grad()  # Zero previous gradients
                y.grad = dgdy.reshape(y.shape)  # Set the gradient manually for Adam
                # print(y.grad)
                optimizer.step()  # Perform an optimization step
                # print(y)
                # y = torch.clamp(y, min=0.01)  # Apply the constraint
                # print(torch.linalg.norm(dgdy2, 2), dgdy, '\n-----')
            if torch.linalg.norm(dgdy, 2) < 5e-2:
                break
            # print('LL error: ', torch.linalg.norm(dgdy, 2), 'Time elapsed:', time.time() - t0)
            
        print('LL error: ', torch.linalg.norm(dgdy, 2), 'Time elapsed:', time.time() - t0, '\n')
        return y, dgdy
    
    def setup_solver(self):
        if self.toy_example or self.toy_example_nc or self.toy_example_cons:
            x0 = torch.randn((self.sizeX, 1), requires_grad=False, dtype=torch.float32).to(self.device)
            t = torch.linspace(0, 200, 20000)
        elif self.toy_CS:
            x0 = torch.randn((self.sizeX, 1), requires_grad=False, dtype=torch.float32).to(self.device)
            t = torch.linspace(0, 20, 25000)
        elif self.DHC or self.DHC_LS:
            x0 = torch.zeros((self.sizeX, 1), requires_grad=False, dtype=torch.float32).to(self.device)
            t = torch.linspace(0, 50, 100)
        elif self.NN:
            x0 = torch.randn((self.sizeX, 1), requires_grad=True, dtype=torch.float32).to(self.device)
            t = torch.linspace(0, 0, 200)

        if self.toy_example_cons:
            y0, dgdy = self.solveLL_constrained(x0)
        else:
            y0, dgdy = self.solveLL(x0)
        # if self.toy_example or self.toy_example_nc or self.toy_CS:# or self.DHC:
        #     y0, dgdy = self.solveLL(x0)
        # else:
        #     y0 = torch.randn((self.sizeY, 1), requires_grad=True, dtype=torch.float32).to(self.device)

        return x0, y0, t
    


    def calc_derivatives(self, x, y, matrixVectorProduct=False, first_order=False):
        # Redefine x and y with fresh computation graph
        x = x.clone().detach().requires_grad_(True)
        y = y.clone().detach().requires_grad_(True)

        if (self.toy_example or self.toy_example_nc):
            dfdx = torch.cos(self.c.T @ x + self.d.T @ y) * self.c + 2 *(x+y) / (torch.linalg.norm(x+y)**2 + 1)
            dfdy = torch.cos(self.c.T @ x + self.d.T @ y) * self.d + 2 *(x+y) / (torch.linalg.norm(x+y)**2 + 1)

            if self.toy_example:
                dgdx = - (self.H @ y - x)
                dgdy = self.H.T @ (self.H @ y - x)
                if first_order:
                    return dfdx, dfdy, dgdx, dgdy

                dgdyy = self.H.T @ self.H
                dgdyx = -self.H
            
            # elif self.toy_example_cons:
            #     dgdx = - (self.H @ y - x)
            #     dgdy = self.H.T @ (self.H @ y - x) - (1 / y)
            #     if first_order:
            #         return dfdx, dfdy, dgdx, dgdy

            #     dgdyy = self.H.T @ self.H + torch.diag(1 / y**2)
            #     dgdyx = -self.H

            elif self.toy_example_nc:
                diff = (self.H @ y - x).reshape(-1, )
                norm2 = torch.sum(diff**2)
                # 
                dgdx = torch.sin(0.5 * norm2) * (self.H @ y - x) 
                dgdy = -torch.sin(0.5 * norm2) * self.H.T @ (self.H @ y - x)
                if first_order:
                    return dfdx, dfdy, dgdx, dgdy
                # 
                dgdyy = -torch.sin(0.5 * norm2) * self.H.T @ self.H - self.H.T @ torch.outer(diff, diff) @ self.H * torch.cos(0.5 * norm2)
                dgdyx = torch.sin(0.5 * norm2)  * self.H.T + self.H.T @ torch.outer(diff, diff) * torch.cos(0.5 * norm2)
            return dfdx, dfdy, dgdx, dgdy, dgdyy, dgdyx
        

        elif self.toy_CS:
            dfdx = torch.zeros_like(x)
            dfdy = y - self.y_tilde

            s = torch.softmax(x, dim=0)
            s_reshaped = s.reshape(-1, )
            dgdx = - ((self.X @ ( y - self.X.T @ s)).T @ (torch.diag(s_reshaped) - torch.outer(s_reshaped, s_reshaped))).T
            dgdy = y - self.X.T @ torch.softmax(x, dim=0)

            if first_order:
                return dfdx, dfdy, dgdx, dgdy

            dgdyy = torch.eye(self.dimY[0]).to(y.device)
            dgdyx = -self.X.T @ (torch.diag(s_reshaped) - torch.outer(s_reshaped, s_reshaped))

            return dfdx, dfdy, dgdx, dgdy, dgdyy, dgdyx

        elif self.DHC or self.DHC_LS:
            x = x.reshape(self.dimX); y = y.reshape(self.dimY)
            logits_val = self.A_val @ y
            logist_tr = self.A_tr @ y

            dfdx  = torch.zeros_like(x)
            dfdy = 1 / self.B_val.shape[0] * (self.A_val.T @ (torch.softmax(logits_val, dim=1) - self.B_val)).reshape(-1, 1)

            loss = F.cross_entropy(logist_tr, self.B_tr)
            sigmoid_x = torch.sigmoid(x)
            softmax_y = torch.softmax(logist_tr, dim=1)
            
            dgdx = 1 / self.B_tr.shape[0] * loss * sigmoid_x * (1 - sigmoid_x)
            # dgdy =  (1 / n_train * A_tr.T @ ((softmax_y - B_tr) * sigmoid_x) + 2 * lam * y).reshape(-1, 1)

            dgdyx = 1 / self.B_tr.shape[0]**2 * (sigmoid_x * (1 - sigmoid_x)).T * (self.A_tr.T @ (softmax_y - self.B_tr)).reshape(-1, 1)
            # dgdyy = 1 / n_train * A_tr.T @ (softmax_y * (1 - softmax_y) * A_tr) \
            #             + 2 * lam * torch.eye(sizeY)
            y = y.reshape((self.sizeY, 1))
            g_val = self.g(x, y)
            dgdy = torch.autograd.grad(g_val, y, create_graph=True, allow_unused=True, materialize_grads=True)[0]
            if first_order:
                return dfdx, dfdy, dgdx, dgdy
            if matrixVectorProduct:
                hessian_vector_product_yy = torch.autograd.grad(dgdy, y, grad_outputs=dgdy, create_graph=True, allow_unused=True)[0]
                return dfdx, dfdy, dgdx, dgdy, hessian_vector_product_yy, dgdyx.T @ dgdy

            else:
                dgdyy = torch.zeros((self.sizeY, self.sizeY)).to(self.device)
                for i in range(dgdy.shape[0]):
                    dgdyy[i, :] = torch.autograd.grad(dgdy[i], y, retain_graph=True, create_graph=True,
                                                    allow_unused=True, materialize_grads=True)[0][:, 0]
                return dfdx, dfdy, dgdx, dgdy, dgdyy, dgdyx

        else:
            # Calculate using PyTorch
            x = x.reshape(self.dimX); y = y.reshape(self.dimY)
            f_val = self.f(x,y)
            g_val = self.g(x,y)
            # make_dot(g_val.mean(), params={'y':y, 'x':x}).render("./Debug/g", format="png")

            dfdx  = torch.autograd.grad(f_val, x, create_graph=True, allow_unused=True, materialize_grads=True)[0]
            dfdy = torch.autograd.grad(f_val, y, create_graph=True, allow_unused=True, materialize_grads=True)[0]

            dgdy, dgdx = torch.autograd.grad(g_val, [y,x], create_graph=True, retain_graph=True, allow_unused=False)
            if first_order:
                return dfdx, dfdy, dgdx, dgdy
            if matrixVectorProduct:
                hessian_vector_product_yy, hessian_vector_product_yx = torch.autograd.grad(dgdy, [y, x], grad_outputs=dgdy, create_graph=True, allow_unused=False)
                return dfdx, dfdy, dgdx, dgdy, hessian_vector_product_yy, hessian_vector_product_yx
            else:
                # Initialize tensors for 2nd derivatives
                dgdyy = torch.zeros((self.sizeY, self.sizeY))
                dgdyx = torch.zeros((self.sizeY, self.sizeX))
                # Compute 2nd derivatives element-wise
                for i in range(dgdy.shape[0]):
                    dgdyy[i, :] = torch.autograd.grad(dgdy[i], y, retain_graph=True, create_graph=True, allow_unused=True, materialize_grads=True)[0][:, 0]
                    dgdyx[i, :] = torch.autograd.grad(dgdy[i], x, retain_graph=True, create_graph=True, allow_unused=True, materialize_grads=True)[0][:, 0]
                return dfdx, dfdy, dgdx, dgdy, dgdyy.to(self.device), dgdyx.to(self.device)
            

    def LineSearch_merit(self, deltaX, x, y, tt, lam, beta, alpha, k=-1, dh_old=None):
        assert beta >= lam
        zero = torch.Tensor([0]).to(self.device)

        def merit_func(x, y, grad_g):
            h = torch.linalg.norm(grad_g, 2)**2 - self.epsilon**2 
            merit =  self.f(x, y) + beta * torch.maximum(zero, h)**2
            return merit, h
        
        def grad_merit(deltaX, df, dh, h):
            if h > 0:
                return (df.T @ deltaX + 2 * beta * dh.T @ deltaX * h)
            else:
                return (df.T @ deltaX)

        eta = 0.10
        t = tt

        dfdx_old, dfdy_old, _, dgdy_old = self.calc_derivatives(x, y, matrixVectorProduct=True, first_order=True)
        df_old = torch.cat((dfdx_old, dfdy_old), 0)
        E_old, h_old =  merit_func(x, y, dgdy_old)
        
        while True:
            x_temp = x + t * deltaX[:self.sizeX]; y_temp = y + t * deltaX[self.sizeX:]
            dfdx, dfdy, dgdx, dgdy = self.calc_derivatives(x_temp, y_temp, matrixVectorProduct=False, first_order=True)
            E_new, h_new = merit_func(x_temp, y_temp, dgdy)
            # if torch.linalg.norm(dgdy_old)**2 > epsilon**2:
            grad = grad_merit(deltaX, df_old, dh_old, h_old)
            try: assert grad <= 0
            except: 
                print(grad, torch.linalg.norm(dgdy_old)**2)
                raise
            if E_new > E_old + eta * t * grad:
                t *= 0.5
            else:
                break
        return t, x_temp, y_temp


    def LineSearch(self, deltaX, x, y, tt, feasible=True, armijo=True, grads=[]):
        t = 10 * tt
        gamma = 0.1
        dfdx_old, dfdy_old, _, dgdy_old = grads
        h_old = torch.linalg.norm(dgdy_old, 2)**2 - self.epsilon**2
        # Feasibility check
        if feasible:
            while True:
                x_temp = x + t * deltaX[:self.sizeX]; y_temp = y + t * deltaX[self.sizeX:]
                dfdx, dfdy, dgdx, dgdy = self.calc_derivatives(x_temp, y_temp, matrixVectorProduct=True, first_order=True)
                self.num_grad_calc += 1
                # print('--', 't: ', t, 'h- : ', h_old.item(), 
                    #   'h+ : ', (torch.linalg.norm(dgdy, 2)**2).item() - epsilon**2)
                if torch.linalg.norm(dgdy, 2)**2 - self.epsilon**2 > (1 - gamma) * h_old:
                    t *= 0.5
                else:
                    break 
        if armijo:
            while True:
                x_temp = x + t * deltaX[:self.sizeX]; y_temp = y + t * deltaX[self.sizeX:]
                dfdx, dfdy, dgdx, dgdy = self.calc_derivatives(x_temp, y_temp, matrixVectorProduct=True, first_order=True)
                self.num_grad_calc += 1
                # if f(x_temp, y_temp) > f(x, y):
                try: assert torch.cat((dfdx_old, dfdy_old), 0).T @ deltaX <= 0
                except: 
                    print(torch.cat((dfdx_old, dfdy_old), 0).T @ deltaX)
                    print(h_old)
                    raise
                if self.f(x_temp, y_temp) > self.f(x, y) + 0.1 * t * torch.cat((dfdx_old, dfdy_old), 0).T @ deltaX:
                # if f(x_temp, y_temp) > f(x, y) - 0.1 * t * torch.linalg.norm(deltaX, 2)**2:
                    t *= 0.5
                else:
                    break
        # t *= 0.5
        x_temp = x + t * deltaX[:self.sizeX]; y_temp = y + t * deltaX[self.sizeX:]
        return t, x_temp, y_temp
    

    def IFDT(self, x, y, alpha, alpha_step=0.1, K=100, beta=1, mode='RXGD', lossS=None):
        zero = torch.Tensor([0]).to(self.device)

        lossF, lossG, lossF2 = lossS
        train_accuracy, val_accuracy, test_accuracy = [], [], []
        train_loss, val_loss, test_loss = [], [], []
        t_list = []
        tt = 1

        if self.use_time: 
            start_time = time.time()

        if beta is not None:
            beta = torch.Tensor([beta]).to(self.device)
        for k in tqdm(range(K if not self.use_time else 1000000)):
            if self.use_time and (time.time() - start_time) > self.time_limit: break
            # Calculate derivatives for current x and y
            if self.toy_example or self.toy_example_nc or self.toy_CS:
                dfdx, dfdy, dgdx, dgdy, dgdyy, dgdyx = self.calc_derivatives(x, y)
                hvp_yy = dgdyy.T @ dgdy
                hvp_yx = dgdyx.T @ dgdy
            else:
                dfdx, dfdy, dgdx, dgdy, hvp_yy, hvp_yx = self.calc_derivatives(x, y, matrixVectorProduct=True)
            self.num_grad_calc += 1

            a = 2 * hvp_yx
            b = 2 * hvp_yy
            c = -alpha * (torch.linalg.norm(dgdy, 2)**2 - self.epsilon**2)
            tot = torch.cat((dfdx, dfdy), 0)
            dh = torch.cat((a, b), 0)    
            if mode  == 'QCQP':
                w = beta
                with torch.no_grad():
                    if False:
                        dtotdt, lam = cvxpy_QCQP(tot, dh, c, w)
                        dtotdt = torch.Tensor(dtotdt).to(self.device)
                        lam = torch.Tensor(lam).to(self.device)
                    else:
                        rad = torch.sqrt(torch.linalg.norm(dh / (2*w))**2 + c / w)
                        term = torch.linalg.norm(-tot + dh / (2 * w))
                        if term > rad:
                            dtotdt = -dh/(2*w) + rad * (-tot + dh / (2 * w)) / term
                            assert torch.allclose(torch.linalg.norm(dtotdt + dh/(2*w)), rad, atol=1e-6)
                        else:
                            dtotdt = -tot
                    
                    # if not self.toy_example_cons:
                    try: assert (torch.allclose(dh.T @ dtotdt, c - w * torch.linalg.norm(dtotdt, 2)**2, atol=1e-3) or \
                                    dh.T @ dtotdt < c - w * torch.linalg.norm(dtotdt, 2)**2)    
                    except: print((dh.T @ dtotdt).item(), (c - w * torch.linalg.norm(dtotdt, 2)**2).item(), c); raise

                    # # Checking the step!
                    if False:
                        dtotdt_cvxpy, lam = cvxpy_QCQP(tot, dh, c, w)
                        dtotdt_cvxpy = torch.Tensor(dtotdt_cvxpy).to(self.device)
                        try: assert torch.allclose(dtotdt, dtotdt_cvxpy, atol=1e-4)
                        except: print(torch.linalg.norm(dtotdt - dtotdt_cvxpy, 2)); raise

                # If the QCQP is infeasible, or we are on the boundary, we use costant step size
                if torch.linalg.norm(dgdy, 2)**2 - self.epsilon**2 >= -1e-5 or \
                                    (-0.25/w * torch.linalg.norm(dh, 2)**2) > c:
                    print('constant step activated')
                    if self.toy_example_cons:
                        alpha_step = 5e-3
                    x = x + alpha_step * dtotdt[:self.sizeX]; y = y + alpha_step * dtotdt[self.sizeX:]
                else:
                    tt, x, y = self.LineSearch(dtotdt, x, y, tt=0.1, grads=(dfdx, dfdy, dgdx, dgdy))
                    t_list.append([tt])
                
            elif mode == 'MOGD':
                if k % 500 == 0:
                    beta *= 2
                    print('==', k, beta)
                # c = -alpha * torch.linalg.norm(dgdy, 2)**2
                c = -alpha * (torch.linalg.norm(dgdy, 2)**2 - self.epsilon**2)
                with torch.no_grad():
                    dtotdt, lam = cvxpy_MOGD(tot, dh, c, beta)
                    dtotdt = torch.Tensor(dtotdt).to(self.device)
                    lam = torch.Tensor(lam).to(self.device)
                    # lam = ...
                    # dtotdt = -tot - beta * dh

                    if beta < lam:
                        print('beta < lam', beta, lam)
                        beta = lam
                tt, x, y = self.LineSearch_merit(dtotdt, x, y, tt=1, lam=lam, beta=beta, alpha=alpha, k=k, dh_old=dh)



            elif mode == 'RXGD':
                with torch.no_grad():
                    d = dh * torch.maximum(zero, -dh.T @ tot - c) / (torch.linalg.norm(dh, 2)**2)
                    dtotdt = -tot - d
                    dxdt = dtotdt[:self.sizeX]; dydt = dtotdt[self.sizeX:]
                    with torch.no_grad():
                        x = x + alpha_step * dxdt
                        y = y + alpha_step * dydt


            elif mode in ['QP1', 'QP2']:
                with torch.no_grad():
                    if mode == 'QP1':
                        # K^-1/3 ~ 0.001
                        if self.toy_example: alpha_K = 1.5 * K**(-1/3); alpha_step_K = 1.5 * K**(-1/3)
                        elif self.toy_example_nc: alpha_K = 1.5 * K**(-1/3); alpha_step_K = 2 * K**(-1/3)
                        elif self.toy_CS: alpha_K = 0.1 * K**(-1/3); alpha_step_K = 0.1 * K**(-1/3)
                        else: alpha_K = 0.01 * K**(-1/3); alpha_step_K = 1 * K**(-1/3)
                        cprime = alpha_K * (torch.linalg.norm(dh, 2)**2)
                    elif mode == 'QP2':
                        # K^-1/3 ~ 0.001, K^-2/3 ~ 0.0005
                        if self.toy_example: alpha_K = 10 * K**(-1/3); alpha_step_K = 10 * K**(-2/3)
                        elif self.toy_example_nc: alpha_K = 10 * K**(-1/3); alpha_step_K = 0.5 * K**(-2/3)
                        else: alpha_K = 10 * K**(-1/3); alpha_step_K = 10 * K**(-2/3)
                        cprime = alpha_K * (torch.linalg.norm(dh, 2) * torch.linalg.norm(dgdy, 2))
                    else:
                        raise ValueError('Invalid mode')

                    lam = torch.maximum(zero, -tot.T @ dh + cprime) / torch.linalg.norm(dh, 2)**2
                    dtotdt = -tot - lam * dh
                    dxdt = dtotdt[:self.sizeX]; dydt = dtotdt[self.sizeX:]
                    x = x + alpha_step_K * dxdt
                    y = y + alpha_step_K * dydt
            else:
                raise NotImplementedError('Invalid mode')
            
            # Compute and store losses  
            term1, term2, term3 = calculate_losses(torch.cat((x, y), 0), self.f, self.sizeX, self.sizeY,\
                                                    self.calc_derivatives, testID=self.testID, deltaX=dtotdt)
            lossF.append(term1); lossG.append(term2); lossF2.append(term3)

            if self.DHC or self.DHC_LS or self.NN:
                pars = (self.A_tr, self.B_tr, self.A_val, self.B_val, self.A_test, self.B_test)
                train_accuracy, val_accuracy, test_accuracy, train_loss, val_loss, test_loss = add_loss(y, train_accuracy, val_accuracy, test_accuracy, 
                                                                                                train_loss, val_loss, test_loss, pars, self.dimY, self.arch)
            if False:
                # when comparing based on number of gradient calculations
                if mode == 'QCQP' and self.num_grad_calc > K: break
        if False:
            plt.figure()
            plt.plot(t_list)
            plt.xlabel('Iterations')
            plt.ylabel('Step size')
            plt.yscale('log')
        # Convert lists of losses to tensors for easy analysis
        return np.array(lossF), np.array(lossG), np.array(lossF2), (train_accuracy, val_accuracy, test_accuracy), (train_loss, val_loss, test_loss)
    
    def BOME(self, x, y0, alpha_step, K, T):
        y = y0
        lossF, lossG, lossF2 = [], [], []
        train_accuracy, val_accuracy, test_accuracy = [], [], []
        train_loss, val_loss, test_loss = [], [], []

        if self.use_time: 
            K = 1000000
            start_time = time.time()

        for k in tqdm(range(K)):
            if self.use_time and (time.time() - start_time) > self.time_limit: break
            y_gd = y.clone().detach()
            for t in range(T):
                # inner loop
                dfdx, dfdy, dgdx, dgdy = self.calc_derivatives(x, y_gd, first_order=True)
                with torch.no_grad():
                    y_gd = y_gd - alpha_step * dgdy

                term1, term2, term3 = calculate_losses(torch.cat((x, y), 0), self.f, self.sizeX, self.sizeY,\
                                                        self.calc_derivatives, testID=self.testID, deltaX=dgdy)
                if self.toy_example_nc or self.toy_CS or self.NN:
                    lossF.append(term1); lossG.append(term2); lossF2.append(lossF2[-1] if len(lossF2) > 0 else 0)
                else:
                    lossF.append(term1); lossG.append(term2); lossF2.append(term3)
                if self.DHC or self.DHC_LS or self.NN:
                    pars = (self.A_tr, self.B_tr, self.A_val, self.B_val, self.A_test, self.B_test)
                    train_accuracy, val_accuracy, test_accuracy, train_loss, val_loss, test_loss = add_loss(y, train_accuracy, val_accuracy, test_accuracy, 
                                                                                                train_loss, val_loss, test_loss, pars, self.dimY, self.arch)
            # outer-loop
            dfdx, dfdy, dgdx, dgdy = self.calc_derivatives(x, y, first_order=True)
            _, _, dgdx2, dgdy2 = self.calc_derivatives(x, y_gd, first_order=True)
            dqdx = dgdx - dgdx2
            dqdy = dgdy
            with torch.no_grad():
                # phi = 0.5 * torch.linalg.norm(torch.cat((dqdx, dqdy), 0), 2)**2
                phi = 0.5 * (self.g(x, y) - self.g(x, y_gd))
                term = torch.linalg.norm(torch.cat((dqdx, dqdy), 0), 2)**2
                lam = torch.max(torch.Tensor([0]).to(self.device), phi - (dqdx.T @ dfdx + dqdy.T @ dfdy)) / term
                x = x - alpha_step * (dfdx + lam * dqdx)
                y = y - alpha_step * (dfdy + lam * dqdy)
            # y.requires_grad = True
            term1, term2, term3 = calculate_losses(torch.cat((x, y), 0), self.f, self.sizeX, self.sizeY, self.calc_derivatives, testID=self.testID, 
                                                deltaX=torch.cat(((dfdx + lam * dqdx), (dfdy + lam * dqdy)), 0))
            lossF.append(term1); lossG.append(term2); lossF2.append(term3)
            if self.DHC or self.DHC_LS or self.NN:
                pars = (self.A_tr, self.B_tr, self.A_val, self.B_val, self.A_test, self.B_test)
                train_accuracy, val_accuracy, test_accuracy, train_loss, val_loss, test_loss = add_loss(y, train_accuracy, val_accuracy, test_accuracy, 
                                                                                                train_loss, val_loss, test_loss, pars, self.dimY, self.arch)
        
        return np.array(lossF), np.array(lossG), np.array(lossF2), (train_accuracy, val_accuracy, test_accuracy), (train_loss, val_loss, test_loss)
    

    def VPBGD(self, x, y0, alpha_step, K, T):
        y = y0
        lossF, lossG, lossF2 = [], [], []
        train_accuracy, val_accuracy, test_accuracy = [], [], []
        train_loss, val_loss, test_loss = [], [], []

        if self.use_time: 
            K = 1000000
            start_time = time.time()

        gamma_init = 0 
        gamma_max = 0.2
        gamma_steps = K * 3 // 4
        gamma = gamma_init
        for k in tqdm(range(K)):
            if self.use_time and (time.time() - start_time) > self.time_limit: break
            gamma = min(gamma_max, gamma + gamma_max / gamma_steps)
            y_gd = y
            for t in range(T):
                dfdx, dfdy, dgdx, dgdy = self.calc_derivatives(x, y_gd, first_order=True)
                y_gd = y_gd - alpha_step * dgdy

                term1, term2, term3 = calculate_losses(torch.cat((x, y_gd), 0), self.f, self.sizeX, self.sizeY, self.calc_derivatives,
                                                        testID=self.testID, deltaX=dgdy)
                if self.toy_example_nc or self.toy_CS or self.NN:
                    lossF.append(term1); lossG.append(term2); lossF2.append(lossF2[-1] if len(lossF2) > 0 else 0)
                else:
                    lossF.append(term1); lossG.append(term2); lossF2.append(term3)
                if self.DHC or self.DHC_LS or self.NN:
                    pars = (self.A_tr, self.B_tr, self.A_val, self.B_val, self.A_test, self.B_test)
                    train_accuracy, val_accuracy, test_accuracy, train_loss, val_loss, test_loss = add_loss(y, train_accuracy, val_accuracy, test_accuracy, 
                                                                                                    train_loss, val_loss, test_loss, pars, self.dimY, self.arch)
            dfdx, dfdy, dgdx, dgdy = self.calc_derivatives(x, y, first_order=True)
            _, _, dgdx2, dgdy2 = self.calc_derivatives(x, y_gd, first_order=True)
            with torch.no_grad():
                x = x - alpha_step * (dfdx + gamma * (dgdx - dgdx2))
                y = y - alpha_step * (dfdy + gamma * dgdy)

            term1, term2, term3 = calculate_losses(torch.cat((x, y), 0), self.f, self.sizeX, self.sizeY, self.calc_derivatives, testID=self.testID,
                                                    deltaX=torch.cat(((dfdx + gamma * (dgdx - dgdx2)), (dfdy + gamma * dgdy)), 0))
            lossF.append(term1); lossG.append(term2); lossF2.append(term3)
            if self.DHC or self.DHC_LS or self.NN:
                pars = (self.A_tr, self.B_tr, self.A_val, self.B_val, self.A_test, self.B_test)
                train_accuracy, val_accuracy, test_accuracy, train_loss, val_loss, test_loss = add_loss(y, train_accuracy, val_accuracy, test_accuracy, 
                                                                                            train_loss, val_loss, test_loss, pars, self.dimY, self.arch)
            
        return np.array(lossF), np.array(lossG), np.array(lossF2), (train_accuracy, val_accuracy, test_accuracy), (train_loss, val_loss, test_loss)
    

    def AIDBio(self, x, y0, alpha_step=0.1, beta_step=0.1, K=10, D=10):
        y = y0
        lossF, lossG, lossF2 = [], [], []
        train_accuracy, val_accuracy, test_accuracy = [], [], []
        train_loss, val_loss, test_loss = [], [], []
        nu = torch.zeros_like(y0)
        # x.requires_grad = False

        if self.use_time: 
            K = 1000000
            start_time = time.time()

        for k in tqdm(range(K)):
            if self.use_time and (time.time() - start_time) > self.time_limit: break
            for t in range(D):
                dfdx, dfdy, dgdx, dgdy, dgdyy, dgdyx = self.calc_derivatives(x, y)
                with torch.no_grad():
                    y = y - alpha_step * dgdy
                
                term1, term2, term3 = calculate_losses(torch.cat((x, y), 0), self.f, self.sizeX, self.sizeY, self.calc_derivatives,\
                                                        testID=self.testID, deltaX=dgdy)
                if self.toy_example_nc or self.toy_CS or self.NN:
                    lossF.append(term1); lossG.append(term2); lossF2.append(lossF2[-1] if len(lossF2) > 0 else 0)
                else:
                    lossF.append(term1); lossG.append(term2); lossF2.append(term3)
                if self.DHC or self.DHC_LS or self.NN:
                    pars = (self.A_tr, self.B_tr, self.A_val, self.B_val, self.A_test, self.B_test)
                    train_accuracy, val_accuracy, test_accuracy, train_loss, val_loss, test_loss = add_loss(y, train_accuracy, val_accuracy, test_accuracy, 
                                                                                                    train_loss, val_loss, test_loss, pars, self.dimY, self.arch)

            dfdx, dfdy, dgdx, dgdy, dgdyy, dgdyx = self.calc_derivatives(x, y)
            with torch.no_grad():
                if False:
                    nu =  dgdyy.inverse() @ dfdy
                else:
                    nu = conjugate_gradient(dgdyy.detach(), dfdy.detach(), nu.detach(), 10)
                x = x - beta_step * (dfdx - dgdyx.T @ nu)
            
            term1, term2, term3 = calculate_losses(torch.cat((x, y), 0), self.f, self.sizeX, self.sizeY, self.calc_derivatives, testID=self.testID,
                                                            deltaX=(dfdx - dgdyx.T @ nu))
            lossF.append(term1); lossG.append(term2); lossF2.append(term3)
            if self.DHC or self.DHC_LS or self.NN:
                pars = (self.A_tr, self.B_tr, self.A_val, self.B_val, self.A_test, self.B_test)
                train_accuracy, val_accuracy, test_accuracy, train_loss, val_loss, test_loss = add_loss(y, train_accuracy, val_accuracy, test_accuracy, 
                                                                                            train_loss, val_loss, test_loss, pars, self.dimY, self.arch)
            
            
        return np.array(lossF), np.array(lossG), np.array(lossF2), (train_accuracy, val_accuracy, test_accuracy), (train_loss, val_loss, test_loss)
    

    def TTSA(self, x, y, alpha=0.1, beta_step=0.1, K=100):
        lossF, lossG, lossF2 = [], [], []
        train_accuracy, val_accuracy, test_accuracy = [], [], []
        train_loss, val_loss, test_loss = [], [], []
        raise NotImplementedError('TTSA is not implemented yet')
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
    

    def NLSolver(self, x, y):
        raise NotImplementedError('NLSolver is not implemented yet')
        lossF, lossG, lossF2 = [], [], []
        train_accuracy, val_accuracy, test_accuracy = [], [], []
        train_loss, val_loss, test_loss = [], [], []

        def objective(z):
            x = torch.tensor(z[:sizeX], dtype=torch.float, device=device, requires_grad=True)
            y = torch.tensor(z[sizeX:], dtype=torch.float, device=device, requires_grad=True)
            return f(x, y).item()
        
        def constraint(z):
            x = torch.tensor(z[:sizeX], dtype=torch.float, device=device, requires_grad=True).unsqueeze(1)
            y = torch.tensor(z[sizeX:], dtype=torch.float, device=device, requires_grad=True).unsqueeze(1)
            dfdx, dfdy, dgdx, dgdy = calc_derivatives(x, y, matrixVectorProduct=False, first_order=True)
            return (torch.linalg.norm(dgdy, 2)**2).item()
        
        def gradient(z):
            x = torch.tensor(z[:sizeX], dtype=torch.float, device=device, requires_grad=True).unsqueeze(1)
            y = torch.tensor(z[sizeX:], dtype=torch.float, device=device, requires_grad=True).unsqueeze(1)
            dfdx, dfdy, dgdx, dgdy = calc_derivatives(x, y, matrixVectorProduct=False, first_order=True)
            # Flatten the gradients to 1D vectors before concatenation:
            grad_x = dfdx.detach().cpu().numpy().flatten()
            grad_y = dfdy.detach().cpu().numpy().flatten()
            concatenated = np.concatenate([grad_x, grad_y])
            return concatenated

        def constraint_grad(z):
            x = torch.tensor(z[:sizeX], dtype=torch.float, device=device, requires_grad=True).unsqueeze(1)
            y = torch.tensor(z[sizeX:], dtype=torch.float, device=device, requires_grad=True).unsqueeze(1)
            dfdx, dfdy, dgdx, dgdy, dgdyy, dgdyx = calc_derivatives(x, y, matrixVectorProduct=True, first_order=False)
            # Compute partial gradients for the constraint:
            grad_x_part = (2 * (dgdyy @ dgdy)).detach().cpu().numpy().flatten()
            grad_y_part = (2 * (dgdyx.T @ dgdy)).detach().cpu().numpy().flatten()
            concatenated = np.concatenate([grad_x_part, grad_y_part])
            return concatenated


        constraints = {'type': 'eq', 'fun': constraint, 'jac': constraint_grad}
        x0 = torch.cat((x, y), 0).detach().cpu().numpy()
        term1, term2, term3 = calculate_losses(torch.cat((x, y), 0), f, sizeX, sizeY, calc_derivatives, testID=args.testID)
        print('Inital vals', 'f(x, y): ', term1, 'LL loss: ', term2, 'UL loss: ', term3)

        # Solve
        # result = minimize(objective, x0, method='SLSQP')
        options = {'ftol': 1e-3, 'eps': 1e-3}
        result = minimize(objective, x0, method='SLSQP', jac=gradient, constraints=constraints, options=options)


        # print(np.linalg.norm(result.x - x0))
        x, y = torch.Tensor(result.x[:sizeX]).to(device), torch.Tensor(result.x[sizeX:]).to(device)
        # x, y = torch.Tensor(x0[:sizeX]).to(device), torch.Tensor(x0[sizeX:]).to(device)

        term1, term2, term3 = calculate_losses(torch.cat((x, y), 0), f, sizeX, sizeY, calc_derivatives, testID=args.testID)
        lossF.append(term1); lossG.append(term2); lossF2.append(term3)
        print('Success', result.success)
        print('Final vals', 'f(x, y): ', term1, 'LL loss: ', term2, 'UL loss: ', term3)
        raise
        return np.array(lossF), np.array(lossG), np.array(lossF2), (train_accuracy, val_accuracy, test_accuracy), (train_loss, val_loss, test_loss)



