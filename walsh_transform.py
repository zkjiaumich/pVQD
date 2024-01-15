import numpy as np
from tqdm import tqdm

#walsh function w_j(x_k) in sequency order
def w(j, k, n):
    j_binary = format(j, f'0{n}b')
    k_dyadic = format(k, f'0{n}b')[::-1]
    exponent = 0
    for i in range(n):
        exponent += int(j_binary[i])*int(k_dyadic[i])
    return (-1)**exponent

#full array of discrete walsh-fourier transform coefficients a_j
def wft(f, n, x_grid, verbose=True):
    N = 2**n
    #coefficient a_j in the discrete walsh-fourier transform of f
    def a_j(f, j, n):
        a_val = 0
        for k in range(N):
            a_val += f(x_grid[k])*w(j, k, n)
        return a_val/N

    transform = []
    if verbose: progress = tqdm(total=N, desc='working on wft')
    for j in range(N):
        transform.append(a_j(f, j, n))
        if verbose: progress.update(1)
    if verbose: progress.close()
    return np.array(transform)

#array of inverse discrete walsh-fourier transform coefficients f_k (keeping terms_kept most significant terms)
def iwft(a, n, terms_kept=None, verbose=True):
    N = 2**n
    if terms_kept is not None:
        sorted_indices = np.argsort(np.abs(a))[::-1]
        kept_indices = sorted_indices[:terms_kept]
    else:
        kept_indices = list(range(N))
    #coefficient f_k in the inverse discrete walsh-fourier transform
    def f_k(a_vals, k, n, kept_indices):
        f_val = 0
        for j in kept_indices:
            f_val += a_vals[j]*w(j, k, n)
        return f_val

    inverse_transform = []
    if verbose: progress = tqdm(total=N, desc='working on iwft')
    for k in range(N):
        inverse_transform.append(f_k(a, k, n, kept_indices))
        if verbose: progress.update(1)
    if verbose: progress.close()
    return np.array(inverse_transform)
