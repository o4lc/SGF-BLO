import numpy as np
import torch
import torchdiffeq
import matplotlib.pyplot as plt
import argparse
import time
import torch.nn.functional as F
from tqdm import tqdm


from utilities import scenario_setup, calculate_accuracy, calculate_loss, get_axs
from utilities import load_setup, conjugate_gradient, calculate_loss, calculate_accuracy, calculate_losses

def calc_derivatives_analytic(x, y):
    if toy_example:
        dfdx = torch.cos(c.T @ x + d.T @ y) * c + 2 *(x+y) / (torch.linalg.norm(x+y)**2 + 1)
        dfdy = torch.cos(c.T @ x + d.T @ y) * d + 2 * (x+y) / (torch.linalg.norm(x+y)**2 + 1)

        dgdx = - (H @ y - x)
        dgdy = H.T @ (H @ y - x)

        dgdyy = H.T @ H
        dgdyx = -H
    else:
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
        dgdyy = torch.zeros((sizeY, sizeY))
        for i in range(dgdy.shape[0]):
            dgdyy[i, :] = torch.autograd.grad(dgdy[i], y, retain_graph=True, create_graph=True,
                                               allow_unused=True, materialize_grads=True)[0][:, 0]
        
    return dfdx, dfdy, dgdx, dgdy, dgdyy, dgdyx


def solveLL(x):
    t0 = time.time()
    y = torch.randn((sizeY, 1), requires_grad=True, dtype=torch.float32)
    lr = 1e-1
    while True:
        dfdx, dfdy, dgdx, dgdy, dgdyy, dgdyx = calc_derivatives(x, y)
        with torch.no_grad():
            HessianInv = dgdyy.inverse()
            y -= lr * HessianInv @ dgdy
            if torch.linalg.norm(dgdy, 2) < 1e-3:
                break

    print('LL error: ', torch.linalg.norm(dgdy, 2), 'Time elapsed:', time.time() - t0)
    return y, dgdy

# Define the system of ODEs
def system(t, variables):
    x, y = variables[:sizeX], variables[sizeX:]
    global dxdt #Because its previous value is required in ProjectMethod 1
    progress_bar.update(1)

    dfdx, dfdy, dgdx, dgdy, dgdyy, dgdyx = calc_derivatives(x, y)
    with torch.no_grad():    
        if method == 'InversionFree':
            a = 2 * dgdyx.T @ dgdy
            b = 2 * dgdyy @ dgdy
            c = -alpha * (torch.linalg.norm(dgdy, 2)**2 - epsilon**2)
            ab = torch.cat((a, b), 0)

            tot = torch.cat((dfdx, dfdy), 0)
            d = ab * torch.maximum(torch.Tensor([0]), -ab.T @ tot - c) / (torch.linalg.norm(a, 2)**2 + torch.linalg.norm(b, 2)**2)
            dtotdt = -tot - d
            dxdt = dtotdt[:sizeX]; dydt = dtotdt[sizeX:]
            
            # if torch.linalg.norm(dgdy, 2) > epsilon and not torch.allclose(torch.linalg.norm(dgdy, 2), torch.Tensor([epsilon])):
            #     print('t=',t, '-', torch.linalg.norm(dgdy, 2), epsilon)
        #  
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

def TTSA(x, y, alpha=0.1, beta=0.1, K=100):
    global A_tr, B_tr, A_val, B_val, A_test, B_test, toy_example, calc_derivatives

    lossF, lossG, lossF2 = [], [], []
    train_accuracy, val_accuracy, test_accuracy = [], [], []
    train_loss, val_loss, test_loss = [], [], []

    for k in tqdm(range(K)):
        # Calculate derivatives for current x and y
        dfdx, dfdy, dgdx, dgdy, dgdyy, dgdyx = calc_derivatives(x, y)

        # Update y
        y = y - beta / (1+k)**(3/5) * dgdy

        # Recalculate derivatives after updating y
        dfdx, dfdy, dgdx, dgdy, dgdyy, dgdyx = calc_derivatives(x, y)

        # Update x
        x = x - alpha / (1+k)**(2/5) * (dfdx - dgdyx.T @ dgdyy.inverse() @ dfdy)

        # Compute and store losses (no gradient tracking)
        term1, term2, term3 = calculate_losses(torch.cat((x, y), 0), f, sizeX, sizeY, calc_derivatives)
        lossF.append(term1); lossG.append(term2); lossF2.append(term3)
        if not toy_example:
            W = y
            train_accuracy.append(calculate_accuracy(A_tr, B_tr, W.reshape(dimY, -1)))
            train_loss.append(calculate_loss(A_tr, B_tr, W.reshape(dimY, -1)))
            val_accuracy.append(calculate_accuracy(A_val, B_val, W.reshape(dimY, -1)))
            val_loss.append(calculate_loss(A_val, B_val, W.reshape(dimY, -1)))
            test_accuracy.append(calculate_accuracy(A_test, B_test, W.reshape(dimY, -1)))
            test_loss.append(calculate_loss(A_test, B_test, W.reshape(dimY, -1)))

    # Convert lists of losses to tensors for easy analysis
    return np.array(lossF), np.array(lossG), np.array(lossF2), (train_accuracy, val_accuracy, test_accuracy), (train_loss, val_loss, test_loss)

def add_loss(W, train_accuracy, val_accuracy, test_accuracy, train_loss, val_loss, test_loss):
    train_accuracy.append(calculate_accuracy(A_tr, B_tr, W.reshape(dimY, -1)))
    train_loss.append(calculate_loss(A_tr, B_tr, W.reshape(dimY, -1)))
    val_accuracy.append(calculate_accuracy(A_val, B_val, W.reshape(dimY, -1)))
    val_loss.append(calculate_loss(A_val, B_val, W.reshape(dimY, -1)))
    test_accuracy.append(calculate_accuracy(A_test, B_test, W.reshape(dimY, -1)))
    test_loss.append(calculate_loss(A_test, B_test, W.reshape(dimY, -1)))
    return train_accuracy, val_accuracy, test_accuracy, train_loss, val_loss, test_loss

def AITBio(x, y0, alpha=0.01, beta=0.01, K=10, D=10):
    global A_tr, B_tr, A_val, B_val, A_test, B_test, toy_example
    y = y0
    lossF, lossG, lossF2 = [], [], []
    train_accuracy, val_accuracy, test_accuracy = [], [], []
    train_loss, val_loss, test_loss = [], [], []
    nu = torch.zeros_like(y0)
    for k in tqdm(range(K)):
        for t in range(D):
            dfdx, dfdy, dgdx, dgdy, dgdyy, dgdyx = calc_derivatives(x, y)
            y = y - alpha * dgdy
            
            term1, term2, term3 = calculate_losses(torch.cat((x, y), 0), f, sizeX, sizeY, calc_derivatives)
            lossF.append(term1); lossG.append(term2); lossF2.append(term3)
            if not toy_example:
                train_accuracy, val_accuracy, test_accuracy, train_loss, val_loss, test_loss = add_loss(y, train_accuracy, val_accuracy, test_accuracy, 
                                                                                                        train_loss, val_loss, test_loss)

        dfdx, dfdy, dgdx, dgdy, dgdyy, dgdyx = calc_derivatives(x, y)
        if False:
            nu =  dgdyy.inverse() @ dfdy
        else:
            nu = conjugate_gradient(dgdyy.detach().numpy(), dfdy.detach().numpy(), nu.detach().numpy(), 10)
        x = x - beta * (dfdx - dgdyx.T @ nu)
        
        term1, term2, term3 = calculate_losses(torch.cat((x, y), 0), f, sizeX, sizeY, calc_derivatives)
        lossF.append(term1); lossG.append(term2); lossF2.append(term3)
        if not toy_example:
            train_accuracy, val_accuracy, test_accuracy, train_loss, val_loss, test_loss = add_loss(y, train_accuracy, val_accuracy, test_accuracy, 
                                                                                                    train_loss, val_loss, test_loss)
        
        
    return np.array(lossF), np.array(lossG), np.array(lossF2), (train_accuracy, val_accuracy, test_accuracy), (train_loss, val_loss, test_loss)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='.',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--toy_example', action='store_true')
    parser.add_argument('--senarioID', type=int, default=0)
    args = parser.parse_args()
    # 
    toy_example = args.toy_example
    torch.manual_seed(0); np.random.seed(0)
    # 
    plt.rcParams.update({'font.size': 13})

    scenarios = scenario_setup(args.senarioID)

    if toy_example:
        f, g, c, d, A, H, dimX, dimY = load_setup(toy_example)
        fig1, ax1, fig11, ax11, fig2, ax2 = get_axs(toy_example)
    else:
        f, g, A_tr, B_tr, A_val, B_val, A_test, B_test, dimX, dimY = load_setup(toy_example, p=scenarios[0][3])
        fig1, ax1, fig11, ax11, fig2, ax2, fig3, ax3, fig4, ax4 = get_axs(toy_example)
    
    calc_derivatives = calc_derivatives_analytic



    sizeX = dimX[0] * dimX[1]; sizeY = dimY[0] * dimY[1]
    

    if toy_example:
        x = torch.randn((sizeX, 1), requires_grad=True, dtype=torch.float32)
        t = torch.linspace(0, 10, 1000)
    else:
        x = torch.zeros((sizeX, 1), requires_grad=True, dtype=torch.float32)
        t = torch.linspace(0, 1000, 1000)

    # y0, dgdy = solveLL(x)
    y0 = torch.randn((sizeY, 1), requires_grad=True, dtype=torch.float32)

    for (method, alpha, epsilon, p) in scenarios:
        print('-- Method:', method, 'Alpha:', alpha, 'Epsilon:', epsilon)

        if not toy_example: f, g, A_tr, B_tr, A_val, solutionB_val, A_test, B_test, dimX, dimY = load_setup(toy_example, p=p)
        # -----------------------------------------------------------------
        t1 = time.time()
        if method in ['InversionFree', 'NewSecondOrder', 'SecondOrder', 'STABLE']:
            initial_conditions = torch.cat((x, y0), 0)
            progress_bar = tqdm(total= 4 * len(t))
            solution = torchdiffeq.odeint(system, initial_conditions, t, method='rk4')
            tt = t
            lossF, lossG, lossF2 = [], [], []
            train_accuracy, val_accuracy, test_accuracy = [], [], []
            train_loss, val_loss, test_loss = [], [], []
            for i in range(len(solution)):
                lossF.append(f(solution[i, :sizeX], solution[i, sizeX:]).detach().numpy().reshape(-1))
                dfdx, dfdy, dgdx, dgdy, dgdyy, dgdyx = calc_derivatives(solution[i, :sizeX], solution[i, sizeX:])
                lossG.append(torch.linalg.norm(dgdy).detach().numpy())
                lossF2.append(torch.linalg.norm(dfdx - dgdyx.T @ dgdyy.inverse() @ dfdy).detach().numpy())
                if not toy_example:
                    train_accuracy, val_accuracy, test_accuracy, train_loss, val_loss, test_loss =\
                          add_loss(solution[i, sizeX:], train_accuracy, val_accuracy, test_accuracy, train_loss, val_loss, test_loss)
                                                                      
            acc = (train_accuracy, val_accuracy, test_accuracy); loss = (train_loss, val_loss, test_loss)
        # elif method == 'AIDBio':
        #     lossF, lossG, lossF2, acc, loss = AIDBio(x, y0, K=np.maximum(1, int(len(t) * 4 / 11)), D=10)
        #     tt = torch.linspace(0, t[-1], lossF.shape[0])
        elif method == 'AITBio':
            lossF, lossG, lossF2, acc, loss = AITBio(x, y0, K=np.maximum(1, int(len(t) * 4 / 11)), D=10)
            tt = torch.linspace(0, t[-1], lossF.shape[0])
        elif method == 'TTSA':
            # y0 = torch.randn((sizeY, 1), requires_grad=True, dtype=torch.float32)
            lossF, lossG, lossF2, acc, loss = TTSA(x, y0, K=np.maximum(1, int(len(t) * 2)))
            tt = torch.linspace(0, t[-1], lossF.shape[0])
        else:
            raise ValueError('Invalid method')
        print('Time taken:', time.time() - t1, '\n')

        
        with torch.no_grad():
            flag_method, flag_alpha, flag_epsilon, flag_p = 1, 1, 1, 1
            try:
                if scenarios[0][0] == scenarios[1][0]: flag_method = 0
                if scenarios[0][1] == scenarios[1][1]: flag_alpha = 0
                if scenarios[0][2] == scenarios[1][2]: flag_epsilon = 0
                if scenarios[0][3] == scenarios[1][3]: flag_p = 0
            except:
                flag_alpha, flag_epsilon, flag_p = 0, 0, 0

            if flag_alpha: strLabel = r': $\alpha$= ' + str(alpha)
            elif flag_epsilon: strLabel = r': $\varepsilon$= ' + str(epsilon)
            elif not toy_example: strLabel = r': p= ' + str(p)
            else: strLabel = ''
            
            ax1.plot(tt, lossF, label= (method + strLabel))
            ax11.plot(tt, lossF2, label=(method + strLabel))
            
            # -----------------------------------------------------
            ax2.plot(tt, lossG, label=(method + strLabel))
            ax2.plot(tt, [epsilon] * len(tt), 'r--')

            if not toy_example: 
                # Plotting accuracy
                print('Train Accuracy:', acc[0][-1].item(), 'Validation Accuracy:', acc[1][-1].item(), 'Test Accuracy:', acc[2][-1].item())
                ax3.plot(tt, acc[2], label=(method + strLabel))
                ax3.set_xlabel('time', fontsize=14)
                ax3.set_ylabel('Test Accuracy', fontsize=14)
                ax3.legend()

                # Plotting loss
                # ax2.plot(t, train_loss, label='Train')
                ax4.plot(tt, loss[1], label=(method + strLabel))
                ax4.set_xlabel('time', fontsize=14)
                ax4.set_ylabel('Validation Loss', fontsize=14)
                ax4.legend()

                fig3.savefig('Result/' + ('toy_example/' if toy_example else 'DHC/') + 'Acc' + '.pdf', dpi=300)
                fig4.savefig('Result/' + ('toy_example/' if toy_example else 'DHC/') + 'Loss' + '.pdf', dpi=300)
            ax1.legend()
            ax1.set_xlabel('time', fontsize=14)
            ax1.set_ylabel('f(x,y)', fontsize=14)

            ax11.legend()
            ax11.set_xlabel('time', fontsize=14)
            ax11.set_ylabel(r'$\|\nabla F(x,y)\|$', fontsize=14)
            ax11.set_yscale('log')

            ax2.legend()
            ax2.set_xlabel('time', fontsize=14)
            ax2.set_ylabel(r'$\|\nabla g(x,y)\|$', fontsize=14)


            # plt.tight_layout()
            scenarioItems = ['method', 'alpha', 'epsilon']
            item = 2 * flag_epsilon + 1 * flag_alpha + 0 * flag_method
            fig1.savefig('Result/' + ('toy_example/' if toy_example else 'DHC/') + scenarioItems[item] + ':' + str(scenarios[0][item]) + '-up1' + '.pdf', dpi=300)
            fig11.savefig('Result/' + ('toy_example/' if toy_example else 'DHC/') + scenarioItems[item] + ':' + str(scenarios[0][item]) + '-up2' + '.pdf', dpi=300)
            fig2.savefig('Result/' + ('toy_example/' if toy_example else 'DHC/') + scenarioItems[item] + ':' + str(scenarios[0][item]) + '-low' + '.pdf', dpi=300)


    plt.close(fig1)
    fig11.show()
    # plt.pause(0)
    plt.show()
    plt.close('all')