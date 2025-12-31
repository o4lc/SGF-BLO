import numpy as np
import torch
import torchdiffeq
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import argparse
import time

from tqdm import tqdm
import os
# from torchviz import make_dot
# from scipy.optimize import minimize
# from cyipopt import minimize_ipopt


from bilevel_solver import BilevelSolver
from utilities import add_loss, calculate_losses, cvxpy_QCQP, cvxpy_MOGD
from setup import load_setup, scenario_setup, get_axs

# Define the system of ODEs
def system(t, variables):
    x, y = variables[:sizeX], variables[sizeX:]
    global dxdt #Because its previous value is required in ProjectMethod 1
    progress_bar.update(1)

    if (method == 'IFCT') and not toy_example:
        dfdx, dfdy, dgdx, dgdy, hvp_yy, hvp_yx = solver.calc_derivatives(x, y, matrixVectorProduct=True)
    else:
        dfdx, dfdy, dgdx, dgdy, dgdyy, dgdyx = solver.calc_derivatives(x, y, matrixVectorProduct=False)
        hvp_yy = dgdyy.T @ dgdy
        hvp_yx = dgdyx.T @ dgdy

    with torch.no_grad():    
        if method == 'IFCT':
            a = 2 * hvp_yx
            b = 2 * hvp_yy
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

 


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='.',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--testID', type=int, default=0)
    parser.add_argument('--scenarioID', type=int, default=0)
    parser.add_argument('--use_time', action='store_true', help='Use wall-clock instead of iterations')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    solver = BilevelSolver(args.testID, args.scenarioID, use_time=args.use_time, device=device)
    # 
    EXPERIMENTS = ['toy_example', 'toy_example_nc', 'toy_example_cons', 'toy_CS', 'DHC', 'DHC_LS', 'NN']
    toy_example = (args.testID == 0)
    toy_example_nc = (args.testID == 1)
    toy_example_cons = (args.testID == 2)
    toy_CS = (args.testID == 3)
    DHC = (args.testID == 4)
    DHC_LS = (args.testID == 5)
    NN = (args.testID == 6)


    plt.rcParams.update({
    'font.size': 16,          # General font size
    'xtick.labelsize': 16,    # Tick label size for x-axis
    'ytick.labelsize': 16,    # Tick label size for y-axis
    'axes.labelsize': 16,      # Font size for axis labels,
    'pdf.fonttype': 42,
    'ps.fonttype': 42
})

    if toy_example or toy_example_nc or toy_example_cons:
        fig1, ax1, fig11, ax11, fig2, ax2 = get_axs(toy_example or toy_example_nc or toy_example_cons)
    elif toy_CS:
        fig1, ax1, fig11, ax11, fig2, ax2 = get_axs(toy_CS=toy_CS)
    else:
        fig1, ax1, fig11, ax11, fig2, ax2, fig3, ax3, fig4, ax4 = get_axs(toy_example)

    scenarios = scenario_setup(args.scenarioID)
    p_old = scenarios[0][3]
    solver.load_setup(p=p_old)
    x0, y0, t = solver.setup_solver()

    sizeX = solver.sizeX; sizeY = solver.sizeY
    dimX = solver.dimX; dimY = solver.dimY

    for (method, alpha, epsilon, p, mode, beta) in scenarios: 
        solver.epsilon = epsilon
        solver.num_grad_calc = 0 
        if p != p_old and (DHC or DHC_LS or NN): 
            solver.load_setup(p=p)
            x0, y0, t = solver.setup_solver()
            p_old = p

        lossF, lossG, lossF2 = [], [], []
        acc, loss = None, None

        if args.scenarioID == 4 or args.scenarioID == 5:
            if p == 0: t = torch.linspace(0, 200, 20000)
            elif p == -1: t = torch.linspace(0, 20, 15000)
            else: t = torch.linspace(0, 20, 10000)  
        
        # -----------------------------------------------------------------
        print('-- Method:', method, 'Alpha:', alpha, 'Epsilon:', epsilon, 'P:', p, 'mode:', mode, 'beta:', beta)
        if toy_example or toy_example_nc: alpha_step = 0.01
        elif toy_CS: alpha_step = 0.5
        else: alpha_step = 1

        t1 = time.time()
        # continuous time methods
        if method in ['IFCT', 'NewSecondOrder', 'SecondOrder', 'STABLE']:
            initial_conditions = torch.cat((x0, y0), 0)
            progress_bar = tqdm(total= 4 * len(t))
            solution = torchdiffeq.odeint(system, initial_conditions, t, method='rk4')
            progress_bar.close()
            tt = t
            train_accuracy, val_accuracy, test_accuracy = [], [], []
            train_loss, val_loss, test_loss = [], [], []
            for i in range(len(solution)):
                dfdx, dfdy, dgdx, dgdy, dgdyy, dgdyx = solver.calc_derivatives(solution[i, :sizeX], solution[i, sizeX:])
                with torch.no_grad():
                    lossF.append(solver.f(solution[i, :sizeX], solution[i, sizeX:]).detach().cpu().numpy().reshape(-1))
                    lossG.append(torch.linalg.norm(dgdy).detach().cpu().numpy())
                    lossF2.append(torch.linalg.norm(dfdx - dgdyx.T @ dgdyy.inverse() @ dfdy).detach().cpu().numpy())
                if DHC or DHC_LS or NN:
                    pars = (solver.A_tr, solver.B_tr, solver.A_val, solver.B_val, solver.A_test, solver.B_test)
                    train_accuracy, val_accuracy, test_accuracy, train_loss, val_loss, test_loss =\
                          add_loss(solution[i, sizeX:], train_accuracy, val_accuracy, test_accuracy, train_loss,\
                                    val_loss, test_loss, pars, solver.dimY, solver.arch)
                                                                      
            acc = (train_accuracy, val_accuracy, test_accuracy); loss = (train_loss, val_loss, test_loss)
        elif method == 'IFDT':
            if toy_example: alpha_step = 0.05
            elif toy_example_nc: alpha_step = 0.001
            elif DHC or DHC_LS: alpha_step = 1
            else: alpha_step = 1
            lossF, lossG, lossF2, acc, loss = solver.IFDT(x0, y0, alpha, alpha_step=alpha_step, K=np.maximum(1, int(len(t) * 4)), mode=mode,
                                                    lossS=(lossF, lossG, lossF2), beta=beta)
            tt = torch.linspace(0, t[-1], lossF.shape[0])
        elif method == 'AIDBio':
            if toy_example_nc: alpha_step = 0.01
            elif toy_example_cons: alpha_step = 0.05
            lossF, lossG, lossF2, acc, loss = solver.AIDBio(x0, y0, alpha_step=alpha_step, beta_step=0.01, K=np.maximum(1, int(len(t) * 4 / 11)), D=10)
            tt = torch.linspace(0, t[-1], lossF.shape[0])
        elif method == 'BOME':
            if toy_example or toy_example_nc: alpha_step = 0.05
            elif toy_example_cons: alpha_step = 0.005
            elif toy_CS: alpha_step *= 0.1
            elif DHC or DHC_LS: alpha_step = 1
            else: alpha_step = 0.1
            lossF, lossG, lossF2, acc, loss = solver.BOME(x0, y0, alpha_step=alpha_step, K=np.maximum(1, int(len(t) * 4 / 11)), T=10)
            tt = torch.linspace(0, t[-1], lossF.shape[0])
        elif method == 'VPBGD':
            if toy_example: alpha_step = 0.01
            elif toy_example_nc: alpha_step = 0.1
            elif toy_example_cons: alpha_step = 0.1
            elif DHC or DHC_LS: alpha_step = 0.5
            else: alpha_step = 0.1

            lossF, lossG, lossF2, acc, loss = solver.VPBGD(x0, y0, alpha_step=alpha_step, K=np.maximum(1, int(len(t) * 4) // 11), T=10)
            tt = torch.linspace(0, t[-1], lossF.shape[0])
        elif method == 'NLSolver':
            lossF, lossG, lossF2 = solver.NLSolver(x0, y0)
            tt = torch.linspace(0, t[-1], lossF.shape[0])
        # elif method == 'TTSA':
        #     lossF, lossG, lossF2, acc, loss = solver.TTSA(x, y0, K=np.maximum(1, int(len(t) * 2)))
        #     tt = torch.linspace(0, t[-1], lossF.shape[0])
        else:
            raise ValueError('Invalid method')
        print('Time taken:', time.time() - t1)

        
        with torch.no_grad():
            flag_method, flag_alpha, flag_epsilon, flag_p, flag_w = 1, 1, 1, 1, 1
            try:
                if args.scenarioID == 4 or args.scenarioID == 5: pass
                if scenarios[0][0] == scenarios[1][0]: flag_method = 0
                if scenarios[0][1] == scenarios[1][1]: flag_alpha = 0
                if scenarios[0][2] == scenarios[1][2]: flag_epsilon = 0
                if scenarios[0][3] == scenarios[1][3]: flag_p = 0
                if scenarios[0][5] == scenarios[1][5]: flag_w = 0
            except:
                flag_alpha, flag_epsilon, flag_p = 0, 0, 0

            # print(flag_method, flag_alpha, flag_epsilon, flag_p, flag_w)
            # raise
            if flag_alpha: strLabel = method + r': $\alpha$= ' + str(alpha)
            elif flag_epsilon: strLabel = method + r': $\varepsilon$= ' + str(epsilon)
            elif flag_w: strLabel = r'$w$= ' + str(beta)
            elif DHC or DHC_LS or NN: 
                if method == 'IFDT': 
                    if mode == 'QP1': strLabel = r"$\rho = \|\nabla h(x,y)\|^2$"
                    elif mode == 'QP2': strLabel = r"$\rho = \|\nabla h(x,y)\|\sqrt{h(x,y)}$"
                    else: strLabel = mode
                    strLabel +=  r': p= ' + str(p)
                else: strLabel = method + r': p= ' + str(p)
            elif args.scenarioID == 4 or args.scenarioID == 5: strLabel = r'K = ' + str(len(tt) // 10**3) + r" $\times 10^3$"

            else:
                if method != 'IFDT': strLabel = method
                else:
                    if mode == 'QP1': strLabel = r"$\rho = \|\nabla h(x,y)\|^2$"
                    elif mode == 'QP2': strLabel = r"$\rho = \|\nabla h(x,y)\|\sqrt{h(x,y)}$"
                    else: strLabel = mode

            print('Number of Gradient Calculations:', len(tt), '\n')
            if 'IFCT' not in [method for method, _, _, _, _, _ in scenarios] and \
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

            dir_path = 'Result/' + EXPERIMENTS[args.testID]
            os.makedirs(dir_path, exist_ok=True)

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
                    fign.savefig('Result/' + EXPERIMENTS[args.testID] + '/Loss_tr' + '.pdf', dpi=300,
                                bbox_inches='tight', pad_inches=0.1)

                ax4.plot(tt, loss[1], label=(strLabel))
                ax4.set_xlabel(t_label)
                ax4.set_ylabel('Validation Loss')
                ax4.legend()

                
                save_title = '/Acc.pdf' if not args.use_time else '/Acc_time.pdf'
                fig3.savefig(dir_path + save_title, dpi=300, bbox_inches='tight', pad_inches=0.1)
                
                save_title = '/Loss.pdf' if not args.use_time else '/Loss_time.pdf'
                fig4.savefig(dir_path + save_title, dpi=300, bbox_inches='tight', pad_inches=0.1)
                
            ax1.legend()
            ax1.set_xlabel(t_label)
            ax1.set_ylabel('f(x,y)')

            ax11.legend()
            ax11.set_xlabel(t_label)
            if toy_example_nc or NN:
                ax11.set_ylabel(r'$\|\Delta z\|$')
            else:
                ax11.set_ylabel(r'$\|F(x,y)\|$')
            ax11.set_yscale('log')

            ax2.legend()
            ax2.set_xlabel(t_label)
            ax2.set_ylabel(r'$\|\nabla g(x,y)\|$')


            # plt.tight_layout()
            scenarioItems = ['method', 'alpha', 'epsilon']
            item = 2 * flag_epsilon + 1 * flag_alpha + 0 * flag_method
            save_title = '/' + scenarioItems[item] + ':' + str(scenarios[0][item]) + '-up1.pdf' if not args.use_time \
                else '/' + scenarioItems[item] + ':' + str(scenarios[0][item]) + '-up1_time.pdf'
            fig1.savefig(dir_path + save_title, dpi=300, bbox_inches='tight', pad_inches=0.1)

            save_title = '/' + scenarioItems[item] + ':' + str(scenarios[0][item]) + '-low.pdf' if not args.use_time \
                else '/' + scenarioItems[item] + ':' + str(scenarios[0][item]) + '-low_time.pdf'
            fig11.savefig(dir_path + save_title, dpi=300, bbox_inches='tight', pad_inches=0.1)

            save_title = '/' + scenarioItems[item] + ':' + str(scenarios[0][item]) + '-up2.pdf' if not args.use_time \
                else '/' + scenarioItems[item] + ':' + str(scenarios[0][item]) + '-up2_time.pdf'
            fig2.savefig(dir_path + save_title, dpi=300, bbox_inches='tight', pad_inches=0.1)


    plt.close(fig1)
    fig11.show()
    # plt.pause(0)
    plt.show(block=False)
    plt.close('all')