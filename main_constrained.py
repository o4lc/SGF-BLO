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

def calc_derivatives_analytic(x, y, lambda_):
    if toy_example:
        dfdx = torch.cos(c.T @ x + d.T @ y) * c + 2 *(x+y) / (torch.linalg.norm(x+y)**2 + 1)
        dfdy = torch.cos(c.T @ x + d.T @ y) * d + 2 * (x+y) / (torch.linalg.norm(x+y)**2 + 1)

        dgdx = - (H @ y - x)
        dgdy = H.T @ (H @ y - x)

        dLdz = torch.cat((dgdy + H_cons.T @ lambda_, H_cons @ y - A_cons @ x), 0)

        dgdyy = H.T @ H
        dgdyx = -H
        
        dLdzx = torch.cat((dgdyx, -A_cons), 0)
        tmp1 = torch.cat((dgdyy, H_cons.T), 1)
        tmp2 = torch.cat((H_cons, torch.zeros((H_cons.shape[0], H_cons.shape[0]))), 1)
        dLdzz = torch.cat((tmp1, tmp2), 0)
    return dfdx, dfdy, dgdx, dgdy, dgdyy, dgdyx, dLdz, dLdzx, dLdzz

def solveLL_constrained(x, H, A):
    t0 = time.time()
    y = torch.randn((sizeY, 1), requires_grad=True, dtype=torch.float32)
    lambda_ = torch.zeros((H.shape[0], 1), requires_grad=False, dtype=torch.float32)
    
    lr = 1e-1  # Learning rate
    tolerance = 1e-3  # Stopping criterion
    
    while True:
        # Compute the derivatives
        dfdx, dfdy, dgdx, dgdy, dgdyy, dgdyx, dLdz, dLdzx, dLdzz = calc_derivatives(x, y, lambda_)

        # Compute residuals (gradients)
        res_grad_y = dgdy + H.T @ lambda_  # Gradient of Lagrangian w.r.t y
        res_constraint = H @ y - A @ x  # Constraint residual

        # Stopping criteria based on the gradient and constraint residuals
        if torch.linalg.norm(res_grad_y, 2) < tolerance and torch.linalg.norm(res_constraint, 2) < tolerance:
            break

        # Form the Newton system (KKT conditions)
        KKT_matrix = torch.cat([torch.cat([dgdyy, H.T], dim=1),
                                torch.cat([H, torch.zeros((H.shape[0], H.shape[0]), dtype=torch.float32)], dim=1)], dim=0)
        KKT_matrix += 1e-5 * torch.eye(KKT_matrix.shape[0])# Regularization

        # Right-hand side (residuals)
        RHS = torch.cat([-res_grad_y, -res_constraint], dim=0)

        # Solve for Newton step [Delta y, Delta lambda]
        delta = torch.linalg.solve(KKT_matrix, RHS)
        delta_y = delta[:sizeY]
        delta_lambda = delta[sizeY:]

        # Update primal and dual variables
        y = y + lr * delta_y
        lambda_ = lambda_ + lr * delta_lambda

    print(f'Constrained LL error: {torch.linalg.norm(res_grad_y, 2):.6f}, Residual: {torch.linalg.norm(res_constraint)} Time elapsed: {time.time() - t0:.2f}s')
    return y, lambda_, dgdy


def constrained_system(t, variables):
    x, y, lamb = variables[:sizeX], variables[sizeX:sizeX+sizeY], variables[sizeX+sizeY:]
    progress_bar.update(1)

    dfdx, dfdy, dgdx, dgdy, dgdyy, dgdyx, dLdz, dLdzx, dLdzz = calc_derivatives(x, y, lamb)
    with torch.no_grad():    
        if method == 'InversionFree':
            a = 2 * dLdzx.T @ dLdz
            b = 2 * dLdzz @ dLdz
            c = -alpha * (torch.linalg.norm(dLdz, 2)**2 - epsilon**2)
            ab = torch.cat((a, b), 0)

            tot = torch.cat((dfdx, dfdy, torch.zeros((lamb.shape))), 0)
            d = ab * torch.maximum(torch.Tensor([0]), -ab.T @ tot - c) / (torch.linalg.norm(a, 2)**2 + torch.linalg.norm(b, 2)**2)
            dtotdt = -tot - d
            dxdt = dtotdt[:sizeX]; dydt = dtotdt[sizeX:sizeX+sizeY]; dlambdt = dtotdt[sizeX+sizeY:]
            
            # if torch.linalg.norm(dgdy, 2) > epsilon and not torch.allclose(torch.linalg.norm(dgdy, 2), torch.Tensor([epsilon])):
            #     print('t=',t, '-', torch.linalg.norm(dgdy, 2), epsilon)
        #  
    return torch.cat((dxdt, dydt, dlambdt), 0)


def add_loss(W, train_accuracy, val_accuracy, test_accuracy, train_loss, val_loss, test_loss):
    train_accuracy.append(calculate_accuracy(A_tr, B_tr, W.reshape(dimY, -1)).reshape(-1))
    val_accuracy.append(calculate_accuracy(A_val, B_val, W.reshape(dimY, -1)).reshape(-1))
    test_accuracy.append(calculate_accuracy(A_test, B_test, W.reshape(dimY, -1)).reshape(-1))

    train_loss.append(calculate_loss(A_tr, B_tr, W.reshape(dimY, -1)).reshape(-1))
    val_loss.append(calculate_loss(A_val, B_val, W.reshape(dimY, -1)).reshape(-1))
    test_loss.append(calculate_loss(A_test, B_test, W.reshape(dimY, -1)).reshape(-1))
    return train_accuracy, val_accuracy, test_accuracy, train_loss, val_loss, test_loss





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='.',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--toy_example', action='store_true')
    parser.add_argument('--senarioID', type=int, default=0)

    args = parser.parse_args()
    # 
    toy_example = args.toy_example
    plt.rcParams.update({
    'font.size': 16,          # General font size
    'xtick.labelsize': 16,    # Tick label size for x-axis
    'ytick.labelsize': 16,    # Tick label size for y-axis
    'axes.labelsize': 16      # Font size for axis labels
})


    scenarios = scenario_setup(args.senarioID)

    if toy_example:
        f, g, Lagrangian, c, d, A, H, A_cons, H_cons, dimX, dimY, dimLambda = load_setup(toy_example, constrained=True)
        fig1, ax1, fig11, ax11, fig2, ax2 = get_axs(toy_example)
    else:
        f, g, A_tr, B_tr, A_val, B_val, A_test, B_test, dimX, dimY = load_setup(toy_example, p=scenarios[0][3])
        fig1, ax1, fig11, ax11, fig2, ax2, fig3, ax3, fig4, ax4 = get_axs(toy_example)
    
    sizeX = dimX[0] * dimX[1]; sizeY = dimY[0] * dimY[1]; sizeLambda = dimLambda[0] * dimLambda[1]
    
    for (method, alpha, epsilon, p) in scenarios:
        torch.manual_seed(0); np.random.seed(0) 
        if toy_example:
            x = torch.randn((sizeX, 1), requires_grad=False, dtype=torch.float32)
            t = torch.linspace(0, 100, 100000)
        else:
            x = torch.zeros((sizeX, 1), requires_grad=False, dtype=torch.float32)
            t = torch.linspace(0, 100, 100)

        if not toy_example: f, g, A_tr, B_tr, A_val, B_val, A_test, B_test, dimX, dimY = load_setup(toy_example, p=p)
        calc_derivatives = calc_derivatives_analytic
        if toy_example and 'InversionFree' in [method for method, _, _, _ in scenarios]:
            y0, lambda0, dgdy = solveLL_constrained(x, H_cons, A_cons)
        else:
            y0 = torch.randn((sizeY, 1), requires_grad=True, dtype=torch.float32)
            lambda0 = torch.randn((sizeLambda, 1), requires_grad=False, dtype=torch.float32)
        
        
        lossF, lossG, lossF2 = [], [], []
        acc, loss = None, None
        # -----------------------------------------------------------------
        print('-- Method:', method, 'Alpha:', alpha, 'Epsilon:', epsilon)
        t1 = time.time()
        if method in ['InversionFree', 'STABLE'] or 'SecondOrder' in method:
            initial_conditions = torch.cat((x, y0, lambda0), 0)
            progress_bar = tqdm(total= 4 * len(t))
            solution = torchdiffeq.odeint(constrained_system, initial_conditions, t, method='rk4')
            progress_bar.close()
            tt = t
            train_accuracy, val_accuracy, test_accuracy = [], [], []
            train_loss, val_loss, test_loss = [], [], []
            for i in range(len(solution)):
                dfdx, dfdy, dgdx, dgdy, dgdyy, dgdyx, dLdz, dLdzx, dLdzz = \
                        calc_derivatives(solution[i, :sizeX], solution[i, sizeX:sizeX+sizeY], solution[i, sizeX+sizeY:])
                with torch.no_grad():
                    lossF.append(f(solution[i, :sizeX], solution[i, sizeX:sizeX+sizeY]).detach().numpy().reshape(-1))
                    lossG.append(torch.linalg.norm(dLdz).detach().numpy())
                    lossF2.append(torch.linalg.norm(dfdx - dgdyx.T @ dgdyy.inverse() @ dfdy).detach().numpy())
                    if not toy_example:
                        train_accuracy, val_accuracy, test_accuracy, train_loss, val_loss, test_loss =\
                            add_loss(solution[i, sizeX:], train_accuracy, val_accuracy, test_accuracy, train_loss, val_loss, test_loss)
                                                                      
            acc = (train_accuracy, val_accuracy, test_accuracy); loss = (train_loss, val_loss, test_loss)
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
            line1, = ax11.plot(tt, lossF2 ,label=(method + strLabel))
            
            # -----------------------------------------------------
            line2, = ax2.plot(tt, lossG, label=(method + strLabel))
            ax2.plot(tt, [epsilon] * len(tt), 'r--')

            if not toy_example: 
                # Plotting accuracy
                print('Train Accuracy:', acc[0][-1].item(), 'Validation Accuracy:', acc[1][-1].item(), 'Test Accuracy:', acc[2][-1].item())
                ax3.plot(tt, acc[2], label=(method + strLabel))
                ax3.set_xlabel('time')
                ax3.set_ylabel('Test Accuracy')
                ax3.legend()

                # Plotting loss
                ax4.plot(tt, loss[1], label=(method + strLabel))
                ax4.set_xlabel('time')
                ax4.set_ylabel('Validation Loss')
                ax4.legend()

                fig3.savefig('Result/' + ('toy_example/' if toy_example else 'DHC/') + 'Acc' + '.pdf', dpi=300,
                             bbox_inches='tight', pad_inches=0.1)
                fig4.savefig('Result/' + ('toy_example/' if toy_example else 'DHC/') + 'Loss' + '.pdf', dpi=300,
                             bbox_inches='tight', pad_inches=0.1)
            ax1.legend()
            ax1.set_xlabel('time')
            ax1.set_ylabel('f(x,y)')

            ax11.legend()
            ax11.set_xlabel('time')
            ax11.set_ylabel(r'$\|F(x,y)\|$')
            ax11.set_yscale('log')
            # ax11.tick_params(axis='y', colors='blue') 
            

            # ax2.legend([line1, line2], ['Norm of Surrogate for Implicit Gradient', 'Norm of Lower-level Gradient'], loc='upper right')
            ax2.legend()
            ax2.set_xlabel('time')
            ax2.set_ylabel(r'$\|\nabla \mathcal{L}(x,y, \lambda)\|$')
            ax2.set_yscale('log')
            # ax2.tick_params(axis='y', colors='green') 

            # plt.tight_layout()
            scenarioItems = ['method', 'alpha', 'epsilon']
            item = 2 * flag_epsilon + 1 * flag_alpha + 0 * flag_method
            fig1.savefig('Result/' + ('toy_example/' if toy_example else 'DHC/') + scenarioItems[item] + ':' + str(scenarios[0][item]) + '-up1' + '.pdf',
                          dpi=300, bbox_inches='tight', pad_inches=0.1)
            fig11.savefig('Result/' + ('toy_example/' if toy_example else 'DHC/') + scenarioItems[item] + ':' + str(scenarios[0][item]) + '-up2' + '.pdf',
                           dpi=300, bbox_inches='tight', pad_inches=0.1)
            fig2.savefig('Result/' + ('toy_example/' if toy_example else 'DHC/') + scenarioItems[item] + ':' + str(scenarios[0][item]) + '-low' + '.pdf',
                          dpi=300, bbox_inches='tight', pad_inches=0.1)


    plt.close(fig1)
    fig11.show()
    # plt.pause(0)
    plt.show()
    plt.close('all')