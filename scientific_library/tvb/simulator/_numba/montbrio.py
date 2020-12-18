import math

import numba as nb
import numpy as np
from numba import cuda
from numba.cuda.random import xoroshiro128p_uniform_float32

from .monitors import fmri, fmri_gpu

def dr(r, V, o_tau, pi, tau, Delta):
    "Time-derivative of r(t) in the Montbrio model."
    return o_tau * (Delta / (pi * tau) + 2 * V * r)


def dV(r, V, o_tau, pi, tau, eta, J, I, cr, rc, cv, Vc):
    "Time-derivative of V(t) in the Montbrio model."
    return o_tau * (V ** 2 - (pi ** 2) * (tau ** 2) * (r ** 2) + eta + J * tau * r + I + cr * rc + cv * Vc)


def make_rk4_rV(dt, sqrt_dt, o_6, use_cuda=False):
    if use_cuda:
        jit = cuda.jit(inline='always',device=True)
        itx = cuda.threadIdx.x
    else:
        jit = nb.njit(fastmath=True,boundscheck=False,inline='always')
        itx = 0
    dr_ = jit(dr)
    dV_ = jit(dV)
    @jit
    def rk4_rV(it, nrV, rti, Vti,
               o_tau, pi, tau, Delta, eta, J, I, cr, rc, cv, Vc,
               r_sigma, V_sigma, z0, z1):
        dr_0 = dr_(rti, Vti, o_tau, pi, tau, Delta)
        dV_0 = dV_(rti, Vti, o_tau, pi, tau, eta, J, I, cr, rc, cv, Vc)
        kh = nb.float32(0.5)
        dr_1 = dr_(rti + dt * kh * dr_0, Vti + dt * kh * dV_0, o_tau, pi, tau, Delta)
        dV_1 = dV_(rti + dt * kh * dr_0, Vti + dt * kh * dV_0, o_tau, pi, tau, eta, J, I, cr, rc, cv, Vc)
        dr_2 = dr_(rti + dt * kh * dr_1, Vti + dt * kh * dV_1, o_tau, pi, tau, Delta)
        dV_2 = dV_(rti + dt * kh * dr_1, Vti + dt * kh * dV_1, o_tau, pi, tau, eta, J, I, cr, rc, cv, Vc)
        kh = nb.float32(1.0)
        dr_3 = dr_(rti + dt * kh * dr_2, Vti + dt * kh * dV_2, o_tau, pi, tau, Delta)
        dV_3 = dV_(rti + dt * kh * dr_2, Vti + dt * kh * dV_2, o_tau, pi, tau, eta, J, I, cr, rc, cv, Vc)
        nrV[0, it] = rti + o_6 * dt * (dr_0 + 2 * (dr_1 + dr_2) + dr_3) + sqrt_dt * r_sigma * z0
        nrV[0, it] *= nrV[0, it] > 0
        nrV[1, it] = Vti + o_6 * dt * (dV_0 + 2 * (dV_1 + dV_2) + dV_3) + sqrt_dt * V_sigma * z1
    if use_cuda:
        def rk4_rV_wrapper(nrV, rti, Vti,
               o_tau, pi, tau, Delta, eta, J, I, cr, rc, cv, Vc,
               r_sigma, V_sigma, z0, z1):
            rk4_rV(cuda.threadIdx.x, nrV, rti, Vti,
               o_tau, pi, tau, Delta, eta, J, I, cr, rc, cv, Vc,
               r_sigma, V_sigma, z0, z1)
    else:
        def rk4_rV_wrapper(nrV, rti, Vti,
               o_tau, pi, tau, Delta, eta, J, I, cr, rc, cv, Vc,
               r_sigma, V_sigma, z0, z1):
            rk4_rV(nb.uint32(0), nrV, rti, Vti,
               o_tau, pi, tau, Delta, eta, J, I, cr, rc, cv, Vc,
               r_sigma, V_sigma, z0, z1)
    return jit(rk4_rV_wrapper)


def setup_const(nh, nto, nn, dt):
    nh, nn = [nb.uint32(_) for _ in (nh, nn)]
    dt, pi = [nb.float32(_) for _ in (dt, np.pi)]
    sqrt_dt = nb.float32(np.sqrt(dt))
    o_nh = nb.float32(1 / nh * nto)
    o_6 = nb.float32(1 / 6)
    return nh, nn, dt, pi, sqrt_dt, o_nh, o_6


def make_gpu_loop(nh, nto, nn, dt, cfpre, cfpost, blockDim_x):
    nh, nn, dt, pi, sqrt_dt, o_nh, o_6 = setup_const(nh, nto, nn, dt)
    rk4_rV = make_rk4_rV(dt, sqrt_dt, o_6, use_cuda=True)
    @cuda.jit(fastmath=True)
    def loop(_, r, V, rngs, w, d, tavg, bold_state, bold_out, I, Delta, eta, tau, J, cr, cv, r_sigmas, V_sigmas):
        if cuda.blockIdx.y >=len(r_sigmas): print('cuda.blockIdx.y is out of bounds')
        if cuda.blockIdx.z >= len(V_sigmas): print('cuda.blockIdx.z is out of bounds')
        r_sigma = r_sigmas[cuda.blockIdx.y] ## select its corresponding parameter r_sigma
        V_sigma = V_sigmas[cuda.blockIdx.z] ## select its corresponding parameter V_sigma
        itx = cuda.threadIdx.x  ## thread id in a 1-d block
        block_index = cuda.blockIdx.x*(cuda.gridDim.y*cuda.gridDim.z)+cuda.blockIdx.y*cuda.gridDim.z+cuda.blockIdx.z
        #if(block_index==123): print('block_index: ', block_index)# select a number as you want to check if block_index is represented correctly
        #if block_index>16383: print('the block index is out of bound')
        it = block_index*cuda.blockDim.x+itx
        #if (it == 9999): print('global_thread_index: ', it) ## select a number as you want to check if it is represented correctly
        ## index of the thread, don't forget check boundaries
        num_threads= cuda.gridDim.x*cuda.gridDim.y*cuda.gridDim.z*cuda.blockDim.x #total number of threads
        #if it>1572863: print('the thread index is out of bound')

        #if it==0: print('hello from ', cuda.blockIdx.x, cuda.threadIdx.x)
        #if it==0: print("NT =", NT)
        o_tau = nb.float32(1 / tau)
        #if it==0: print("o_tau = ", o_tau)
        assert r.shape[0] == V.shape[0] == nh  # shape asserts help numba optimizer
        assert r.shape[1] == V.shape[1] == nn
        #if it==0: print("creating nrV shared..")
        nrV = cuda.shared.array((2, blockDim_x), nb.float32)
        #if it==0: print("zeroing tavg..")
        pi_2 = nb.float32(np.pi * 2)
        for j in range(nto):
            tavg[j, 0, itx, block_index] = nb.float32(0.0)
            tavg[j, 1, itx, block_index] = nb.float32(0.0)

        #if it==0: print('tavg zero\'d', -1, nh - 1)
        for t0 in range(-1, nh - 1):
            #if itx==0: print('t0=', t0)
            t = nh-1 if t0<0 else t0
            #if itx==0: print('t=', t)
            t1 = t0 + 1
            #if itx==0: print('t1=', t1)
            #if itx==0: print('nh//nto', nh // nto)
            t0_nto = t0 // (nh // nto)
            #print('t0_nto: ', t0_nto)
            #if it==0: print(t, t1, t0_nto)
            rc = nb.float32(0)
            Vc = nb.float32(0)
            for j in range(nn):
                dij = (t - d[itx, j] + nh) & (nh-1)
                rc += w[itx, j] * cfpre(r[dij, j, block_index], r[t, itx, block_index])
                Vc += w[itx, j] * cfpre(V[dij, j, block_index], V[t, itx, block_index])
            rc = cfpost(rc)
            Vc = cfpost(Vc)

            # RNG + Box Muller, random number generation using Box Muller transform
            if it >= num_threads: print('index of threads is out of bound: ', it)
            u1 = xoroshiro128p_uniform_float32(rngs, it)  # take one from the first half of rng states
            u2 = xoroshiro128p_uniform_float32(rngs, (num_threads + it))  # take one from the second half of rng states

            z0 = math.sqrt(-nb.float32(2.0) * math.log(u1)) * math.cos(pi_2 * u2)
            z1 = math.sqrt(-nb.float32(2.0) * math.log(u1)) * math.sin(pi_2 * u2)

            # RK4
            rk4_rV(nrV, r[t, itx, block_index], V[t, itx, block_index],
                   o_tau, pi, tau, Delta, eta, J, I, cr, rc, cv, Vc,
                   r_sigma, V_sigma, z0, z1)

            r[t1, itx, block_index] = nrV[0, itx]
            V[t1, itx, block_index] = nrV[1, itx]
            #if itx==0: print('nrV[0, itx], nrV[1, itx], o_nh: ',nrV[0, itx], nrV[1, itx], o_nh)
            tavg[t0_nto, 0, itx, block_index] += nrV[0, itx] * o_nh
            tavg[t0_nto, 1, itx, block_index] += nrV[1, itx] * o_nh
            #if itx==0: print(t1, o_nh, tavg[t0_nto, 0, itx, block_index], tavg[t0_nto, 1, itx, block_index])
            bold_out[itx, block_index] = fmri_gpu(block_index, bold_state[itx], nrV[0, itx], dt)
            cuda.syncthreads()
            #if itx==0: print("loop body done")
    return loop

def make_loop(nh, nto, nn, dt, cfpre, cfpost):
    nh, nn, dt, pi, sqrt_dt, o_nh, o_6 = setup_const(nh, nto, nn, dt)
    rk4_rV = make_rk4_rV(dt, sqrt_dt, o_6)
    @nb.njit(boundscheck=False, fastmath=True)
    def loop(nrV, r, V, wrV, w, d, tavg, bold_state, bold_out, I, Delta, eta, tau, J, cr, cv, r_sigma, V_sigma):
        o_tau = nb.float32(1 / tau)
        assert r.shape[0] == V.shape[0] == nh  # shape asserts help numba optimizer
        assert r.shape[1] == V.shape[1] == nn
        for j in range(nto):
            for i in range(nn):
                tavg[j, 0, i] = nb.float32(0.0)
                tavg[j, 1, i] = nb.float32(0.0)
        for t0 in range(-1, nh - 1):
            t = nh-1 if t0<0 else t0
            t1 = t0 + 1
            t0_nto = t0 // (nh // nto)
            for i in range(nn):
                rc = nb.float32(0) # using array here costs 50%+
                Vc = nb.float32(0)
                for j in range(nn):
                    dij = (t - d[i, j] + nh) & (nh-1)
                    rc += w[i, j] * cfpre(r[dij, j], r[t, i])
                    Vc += w[i, j] * cfpre(V[dij, j], V[t, i])
                rc = cfpost(rc)
                Vc = cfpost(Vc)
                rk4_rV(nrV, r[t, i], V[t, i],
                       o_tau, pi, tau, Delta, eta, J, I, cr, rc, cv, Vc,
                       r_sigma, V_sigma, wrV[0, t1, i], wrV[1, t1, i])
                r[t1, i] = nrV[0, 0]
                V[t1, i] = nrV[1, 0]
                tavg[t0_nto, 0, i] += r[t1, i] * o_nh
                tavg[t0_nto, 1, i] += V[t1, i] * o_nh
                # bold_out[i] = fmri(bold_state[i], tavg[0, 0, i], dt)
    return loop