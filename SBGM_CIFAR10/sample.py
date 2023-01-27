## Use SDE to generate data, SDE->Reverse-time SDE with time-dependant score_model
## Sample from the prior distribution p~N(x;0,1/2(sigma^2-1)I), then use the Euler-Maruyama求解反时间的SDE
## dt原先是负，所以迭代公式变加法且减号消失
## dt->delta t, dw->z~N(0,g^2(t)*delta t*I)代入迭代公式, zt~N(0,I)
import numpy as np
import torch
import tqdm
from scipy import integrate

'''
欧拉采样速度快但是质量不高
参数：模型，每一时刻加了噪声的标准差，扩散系数方程，生成64个图片
'''
## The number of sampling steps 欧拉采样
num_steps = 500  # 采样总步长，速度快步长小


def euler_sample(score_model, marginal_prob_std, diffusion_coeff, batch_size=64, num_steps=num_steps, device='cuda',
                 eps=1e-3):
    # Step1 定义初始时间1和先验分布的随机样本（t从1到0反时间采样）
    t = torch.ones(batch_size, device=device)  # t=1
    init_x = torch.randn(batch_size, 3, 28, 28, device=device) * marginal_prob_std(t)[:, None, None,
                                                                 None]  # 先验初始分布样本，代入t=1得到标准差，利用重参数算出init t
    # 也可以用分布p采样出样本x

    # Step 2 定义采样的逆时间网格以及每一步的时间步长，根据numsteps确定
    time_steps = torch.linspace(1., eps, num_steps, device=device)  # 序列
    step_size = time_steps[0] - time_steps[1]  # delta t: 0-1满足步长是正数

    # Step 3 根据欧拉算法求解逆时间SDE
    x = init_x
    with torch.no_grad():
        for time_step in tqdm.tqdm(time_steps):  # 时间遍历
            batch_time_step = torch.ones(batch_size, device=device) * time_step
            g = diffusion_coeff(batch_time_step)  # 输入每一个时间的步长值得到当前时刻的扩散系数
            mean_x = x + (g ** 2)[:, None, None, None] * score_model(x, batch_time_step) * step_size
            x = mean_x + torch.sqrt(step_size) * g[:, None, None, None] * torch.randn_like(x)  # 加上随机项

    # Step 4 取最后一步的期望值作为生成的样本
    return mean_x


'''
PC(Predictor-Corrector)采样: 融合欧拉数值解法初始值Predictor + 郎之万动力学模拟采样Corrector（适用于已知分数的分布）
Score-matching 方法训练了分数估计网络--任意时刻t时p(x)处的似然分数
不断迭代，N无穷大，epslion趋近0
共同提高采样样本质量
Predictor: xt->x(t-delta t)
Corrector: 基于predictor做很多步的郎之万采样修正xt->p_(t-delta t)(x) #降低数值解法误差
'''
signal_to_noise_ratio = 0.16
num_steps = 500


# N: Number of discretization steps for the reverse-time SDE
# M: Number of corrector steps, M可以放在参数里或者hardcode设定为10
# snr 多出来的一项
def pc_sampler(score_model, marginal_prob_std, diffusion_coeff, batch_size=64, num_steps=num_steps,
               snr=signal_to_noise_ratio, device='cuda', eps=1e-3):
    """
    Generate samples from score-based models with Predictor-Corrector method
    Args:
        score_model: A PyTorch model that represents the time-dependent score-based model.
        marginal_prob_std: A function that gives the standard deviation of the perturbation kernel.
        diffusion_coeff: A function that gives the standard deviation of the perturbation kernel.
        batch_size: The number of samplers to generate by calling this function once of the SDE.
        num_steps: The number of sampling steps.
            Equivalent to e number of discretized time steps.
        device: 'cuda' or 'CPU'
        eps: The smallest time step for numerical stability.

        Return:
            Samples.
    """

    # Step1 定义初始时间1和先验分布的随机样本（t从1到0反时间采样）
    t = torch.ones(batch_size, device=device)  # t=1
    init_x = torch.randn(batch_size, 3, 28, 28, device=device) * marginal_prob_std(t)[:, None, None,
                                                                 None]  # 先验初始分布样本，代入t=1得到标准差，利用重参数算出init t

    # Step 2 定义采样的逆时间网格以及每一步的时间步长，根据numsteps确定
    time_steps = torch.linspace(1., eps, num_steps, device=device)  # 序列
    step_size = time_steps[0] - time_steps[1]  # delta t: 0-1满足步长是正数

    # Step 3 根据欧拉算法求解逆时间SDE
    x = init_x
    with torch.no_grad():
        for time_step in tqdm.tqdm(time_steps):  # 时间遍历time:1~0
            batch_time_step = torch.ones(batch_size, device=device) * time_step

            # Corrector Step(Langvin MCMC)
            grad = score_model(x, batch_time_step)
            grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
            noise_norm = np.sqrt(np.prod(x.shape[1:]))
            langevin_step_size = 2 * (snr * noise_norm / grad_norm) ** 2  # 校正步长，保证每一步信噪比是固定量级，t减小，郎之万步长减小
            print(f"{langevin_step_size}=")

            # hardcode for M = 10,硬编码是10
            for _ in range(10):
                x = x + langevin_step_size * grad + torch.sqrt(2 * langevin_step_size) * torch.randn_like(
                    x)  # 郎之万采样迭代公式
                grad = score_model(x, batch_time_step)
                grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
                noise_norm = np.sqrt(np.prod(x.shape[1:]))
                langevin_step_size = 2 * (snr * noise_norm / grad_norm) ** 2  # 校正步长，保证每一步信噪比是固定量级，t减小，郎之万步长减小
                print(f"{langevin_step_size}=")

            # Predictor step (Euler-Maruyama)
            g = diffusion_coeff(batch_time_step)
            x_mean = x + (g ** 2)[:, None, None, None] * score_model(x, batch_time_step) * step_size
            x = x_mean + torch.sqrt(g ** 2 * step_size)[:, None, None, None] * torch.randn_like(x)

    # Step 4 取最后一步的欧拉数值求解的期望作为最终生成的样本，P、C顺序可反
    return x_mean


"""
伴随常微分数值生成采样数据，因为上面两种方法不能精确计算SBM取得的对数似然是多少，引入ODE（概率流常微分方程PFODE）求解对数似然，随机项g(t)dw读掉
任意SDE存在一个伴随ODE，伴随ODE和SDE同同样的边缘概率密度轨迹（反时间方向求解ODE，获得样本最终也和SDE求得的服从同一个分布）
从Pt中抽样，从反时间方向对ODE进行积分，最终获得样本P0，dx = -1/2*simga^(2t)*s_(theta)(x,t)dt
This can be done using many block-box ODE solvers provided by packages such as scipy !!!
"""
## The error tolerance for the black-box ODE solver
error_tolerance = 1e-5  # @param {'type':'number'}


def ode_sampler(score_model, marginal_prob_std, diffusion_coeff, batch_size=64, atol=error_tolerance,
                rtol=error_tolerance, device='cuda', z=None, eps=1e-3):
    # Step 1 定义初始时间1和初始值x
    t = torch.ones(batch_size, device=device)  # t=1
    if z is None:
        init_x = torch.randn(batch_size, 3, 28, 28, device=device) * marginal_prob_std(t)[:, None, None, None]
    else:
        init_x = z

    shape = init_x.shape

    # Step 2 定义分数预测函数和常微分函数
    def score_eval_wrapper(sample, time_steps):
        """A wrapper of the score-based model for use by the ODE solver"""
        sample = torch.tensor(sample, device=device, dtype=torch.float32).reshape(shape)
        time_steps = torch.tensor(time_steps, device=device, dtype=torch.float32).reshape((sample.shape[0]), )
        with torch.no_grad():
            score = score_model(sample, time_steps)
        # cpu()/cuda()?
        return score.cpu().numpy().reshape((-1,)).astype(np.float64)

    def ode_func(t, x):
        """The ODE function for use by the ODE solver."""
        time_steps = np.ones((shape[0],)) * t
        g = diffusion_coeff((torch.tensor(t)).cpu().numpy())
        return -0.5 * (g ** 2) * score_eval_wrapper(x, time_steps)

    # Step 3 调用常微分求解算子来解出t=eps时刻的值，即预测的样本
    # 关键！ RK45 solve_ivp求解
    res = integrate.solve_ivp(ode_func, (1, eps), init_x.reshape(-1).cpu().numpy(), rtol=rtol, atol=atol, method='RK45')
    print(f"Number of function evaluation: {res.nfev}")

    x = torch.tensor(res.y[:, -1], device=device).reshape(shape)

    return x
