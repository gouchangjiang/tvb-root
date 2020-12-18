import tqdm
import numpy as np
import numba as nb
from numpy.random import SFC64
from numba.cuda.random import create_xoroshiro128p_states

from tvb.simulator._numba.montbrio import make_gpu_loop, make_loop

from tvb.simulator._ispc.montbrio import run_ispc_montbrio


def make_linear_cfun(scale=0.01):
    k = nb.float32(scale)
    @nb.njit(inline='always')
    def pre(xj, xi):
        return xj
    @nb.njit(inline='always')
    def post(gx):
        return k * gx
    return pre, post


def default_icfun(t, rV):
    rV[0] = 0.0
    rV[1] = -2.0


def run_loop(weights, delays,
             total_time=60e3, bold_tr=1800, coupling_scaling=0.01,
             r_sigma=1e-3, V_sigma=1e-3,
             I=1.0, Delta=1.0, eta=-5.0, tau=100.0, J=15.0, cr=0.01, cv=0.0,
             dt=1.0,
             nh=256,  # history buf len, must be power of 2 & greater than delays.max()/dt
             nto=16,   # num parts of nh for tavg, e.g. nh=256, nto=4: tavg over 64 steps
             progress=False,
             icfun=default_icfun,
             filename="CPU_result_"):
    assert weights.shape == delays.shape and weights.shape[0] == weights.shape[1]
    nn = weights.shape[0]
    w = weights.astype(np.float32)
    d = (delays / dt).astype(np.int32)
    assert d.max() < nh
    # inner loop setup dimensions, constants, buffers
    r, V = rV = np.zeros((2, nh, nn), 'f')
    nrV = np.zeros((2, 1), 'f')  # no stack arrays in Numba
    icfun(-np.r_[:nh]*dt, rV)
    I, Delta, eta, tau, J, cr, cv, r_sigma, V_sigma = [nb.float32(_) for _ in (I, Delta, eta, tau, J, cr, cv, r_sigma, V_sigma)]
    tavg = np.zeros((nto, 2, nn), 'f')                               # buffer for temporal average
    bold_state = np.zeros((nn, 4), 'f')                         # buffer for bold state
    bold_state[:,1:] = 1.0
    bold_out = np.zeros((nn,), 'f')                             # buffer for bold output
    rng = np.random.default_rng(SFC64(42))                      # create RNG w/ known seed
    # first call to jit the function
    cfpre, cfpost = make_linear_cfun(coupling_scaling)
    loop = make_loop(nh, nto, nn, dt, cfpre, cfpost)
    #loop(r, V, wrV, w, d, tavg, bold_state, bold_out, I, Delta, eta, tau, J, cr, cv, r_sigma, V_sigma)
    # outer loop setup
    win_len = nh * dt
    total_wins = int(total_time / win_len)
    print(total_time, win_len, total_wins)
    bold_skip = int(bold_tr / win_len)
    tavg_trace = np.empty((total_wins, ) + tavg.shape, 'f')
    bold_trace = np.empty((total_wins//bold_skip + 1, ) + bold_out.shape, 'f')
    # start time stepping
    for t in (tqdm.trange if progress else range)(total_wins):
        wrV = rng.standard_normal(size=(2, nh, nn), dtype='f') #, out=wrV)  # ~15% time here
        #wrV = rng.uniform(low=0, high=1 , size=(2, nh, nn))
        loop(nrV, r, V, wrV, w, d, tavg, bold_state, bold_out, I, Delta, eta, tau, J, cr, cv, r_sigma, V_sigma)
        tavg_trace[t] = tavg
        file_name_r = filename + "_r_t" + str(t) + ".txt"
        file_name_V = filename + "_V_t" + str(t) + ".txt"
        np.savetxt(file_name_r, tavg[:,0,:])
        np.savetxt(file_name_V, tavg[:,1,:])

        if t % bold_skip == 0:
            bold_trace[t//bold_skip] = bold_out
    return tavg_trace.reshape((-1,) + tavg.shape[1:]), bold_trace

def run_gpu_loop(weights, delays, r_sigmas, V_sigmas,
             total_time=60e3, bold_tr=1800, coupling_scaling=0.01,
             #r_sigma=1e-3, V_sigma=1e-3,
             I=1.0, Delta=1.0, eta=-5.0, tau=100.0, J=15.0, cr=0.01, cv=0.0,
             dt=1.0,
             nh=256,  # history buf len, must be power of 2 & greater than delays.max()/dt
             nto=16,   # num parts of nh for tavg, e.g. nh=256, nto=4: tavg over 64 steps
             progress=False,
             icfun=default_icfun,
             rng_seed=42):
    assert weights.shape == delays.shape and weights.shape[0] == weights.shape[1]
    nn = weights.shape[0]
    w = weights.astype(np.float32)
    d = (delays / dt).astype(np.int32)
    assert d.max() < nh
    assert nto <= nh, 'oversampling <= buffer size'
    make_loop = make_gpu_loop
    # TODO
    block_dim_x = 96         # nodes
    grid_dim_x = 64, 16, 16  # subjects, noise, coupling
    nt = np.prod(grid_dim_x)
    # allocate workspace stuff
    print('allocating memory..')
    if True: # TODO no dedent in lab editor..
        r, V = rV = np.zeros((2, nh, nn, nt), 'f')
        nrV = np.zeros((2, nt), 'f')  # no stack arrays in Numba
        print('creating rngs..', end='')
        rngs = create_xoroshiro128p_states(int(nt * nn * 2), rng_seed)
        print('done')
        tavg = np.zeros((nto, 2, nn, nt), 'f')                               # buffer for temporal average
        bold_state = np.zeros((nn, 4, nt), 'f')                         # buffer for bold state
        bold_state[:,1:] = 1.0
        bold_out = np.zeros((nn, nt), 'f')                             # buffer for bold output
        icfun(-np.r_[:nh]*dt, rV)
    I, Delta, eta, tau, J, cr, cv, r_sigmas, V_sigmas = [
        nb.float32(_) for _ in (I, Delta, eta, tau, J, cr, cv, r_sigmas, V_sigmas)]
    print('workspace allocations done')
    # first call to jit the function
    cfpre, cfpost = make_linear_cfun(coupling_scaling)
    loop = make_loop(nh, nto, nn, dt, cfpre, cfpost, block_dim_x)
    # outer loop setup
    win_len = nto * dt
    total_wins = int(total_time / win_len)
    bold_skip = int(bold_tr / win_len)
    # pinned memory for speeding up kernel invocations
    from numba.cuda import to_device, pinned_array
    g_nrV, g_r, g_V, g_rngs, g_w, g_d, g_tavg, g_bold_state, g_bold_out, g_r_sigmas, g_V_sigmas = [
        to_device(_) for _ in (nrV, r, V, rngs, w, d, tavg, bold_state, bold_out, r_sigmas, V_sigmas)]
    p_tavg = pinned_array(tavg.shape, dtype=np.float32)
    p_bold_out = pinned_array(bold_out.shape, dtype=np.float32)
    # TODO mem map this, since it will get too big
    # tavg_trace = np.zeros((total_wins, ) + tavg.shape, 'f')
    bold_trace = np.zeros((total_wins//bold_skip + 1, ) + bold_out.shape, 'f')
    # start time stepping
    print('starting time stepping')
    for t in (tqdm.trange if progress else range)(total_wins):
        print(t)
        loop[grid_dim_x, block_dim_x](g_nrV, g_r, g_V, g_rngs, g_w, g_d, g_tavg, g_bold_state, g_bold_out,
                     I, Delta, eta, tau, J, cr, cv, g_r_sigmas, g_V_sigmas)
        g_tavg.copy_to_host(p_tavg)
        for time_step in range(nto):
            file_name_r = "D:/result/GPU_result_r_t" + str(t) + "_ts" + str(time_step) +".txt"
            file_name_V = "D:/result/GPU_result_V_t" + str(t) + "_ts" + str(time_step) +".txt"
            np.savetxt(file_name_r, p_tavg[time_step,0])
            np.savetxt(file_name_V, p_tavg[time_step,1])
        print(p_tavg)

        # tavg_trace[t] = p_tavg
        if t % bold_skip == 0:
            g_bold_out.copy_to_host(p_bold_out)
            bold_trace[t//bold_skip] = p_bold_out
    # return tavg_trace, bold_trace



def grid_search(loop, **params):
    import joblib, itertools
    n_jobs = params.pop('n_jobs', 1)
    verbose = params.pop('verbose', 1)
    keys = list(params.keys())
    vals = [params[key] for key in keys]
    # expand product of search dimensions into dict of run_loop kwargs
    args = [dict(list(zip(keys, _))) for _ in itertools.product(*vals)]
    # run in parallel and return
    return args, joblib.Parallel(n_jobs=n_jobs, verbose=verbose, backend='multiprocessing')(
        joblib.delayed(loop)(**arg) for arg in args)


if __name__ == '__main__':
    nn = 96 # number of nodes
    w = np.random.randn(nn, nn)**2 #standard normalization
    d = np.random.rand(nn, nn)**2 * 10 #uniform distribution
    r_sigmas = np.arange(3e-3, 0.011, 5e-4, dtype=np.float32)
    V_sigmas = np.arange(1e-3, 0.009, 5e-4, dtype=np.float32)

    #r_sigmas = np.zeros(16, 'f')
    #V_sigmas = np.zeros(16, 'f')

    ns = 60
    #params = dict(dt=0.05, total_time=60e3, I=1.0, r_sigma=3e-3, V_sigma=1e-3, tau=10.0, progress=True)
    params = dict(dt=0.05, total_time=60e3, I=1.0, tau=10.0, progress=True)
    run_gpu_loop(w, d, r_sigmas, V_sigmas, **params)

    # for index_r in range(len(r_sigmas)):
    #     for index_v in range(len(V_sigmas)):
    #         filenameparameter = "D:/result/CPU_result_"+"r"+str(index_r)+"v"+str(index_v)
    #         params = dict(dt=0.05, total_time=60e3, I=1.0, r_sigma=r_sigmas[0], V_sigma=V_sigmas[0], tau=10.0,
    #                       progress=True, filename=filenameparameter)
    #         run_loop(w, d, **params)
