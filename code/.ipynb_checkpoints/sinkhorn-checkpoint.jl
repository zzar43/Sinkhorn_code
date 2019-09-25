function cost_func_1d(N)
    x = range(0, stop=1, length=N)
    # Build loss matrix
    M = zeros(N,N)
    for i = 1:N
        for j = 1:N
            M[i,j] = (x[i] - x[j])^2
        end
    end
    return M
end

function cost_matrix_2d(Nx, Ny)
    x = range(0, stop=1, length=Nx)
    y = range(0, stop=1, length=Ny)'
    X = repeat(x,1,Ny)
    Y = repeat(y,Nx,1)
    XX = X[:]
    YY = Y[:]
    M = zeros(Nx*Ny, Nx*Ny)
    for i = 1:Nx*Ny
        for j = 1:Nx*Ny
            M[i,j] = (XX[i] - XX[j])^2 + (YY[i] - YY[j])^2
        end
    end
    
    return M
end


function sinkhorn_basic(r, c, M; lambda=100, numItermax = 1000, stopThr = 1e-6, verbose=false)
    T = 0 .* M
    I = findall(x->x>0, r)
    N = length(r)
    Ns = length(I)
    r = r[I]
    M1 = M[I,:]
    K = exp.(-1 .* lambda .* M1)
    
    u = ones(Ns) ./ Ns
    v = ones(N) ./ N
    u0 = ones(Ns) ./ Ns

    K_tilde = diagm(1 ./ r) * K;

    numIter = 1
    e = stopThr
    while (numIter <= numItermax) && (e >= stopThr)
        v = c ./ (K' * u)
        u = 1 ./ (K_tilde * v)

        e = norm(u0-u)
        numIter += 1
        u0[:] = u[:]
    end
    v = c ./ (K' * u)

    T[I,:] = diagm(u) * K * diagm(v)
    d = sum(T.*M)
    alpha = zeros(N)
    alpha[I] = -1/lambda .* log.(u)  + (log.(u)'*ones(length(u)))./(lambda*length(u)) .* ones(length(u))
    
    if verbose == true
        println("Iteration number: ", numIter-1)
        println("Error: ", e)
    end
    return T, alpha, d
end

function sinkhorn_signal_1d(r, c, M; lambda=100, numItermax = 1000, stopThr = 1e-6, verbose=false)
    
    N = length(r)
    # normalization
    cp = zeros(N)
    cn = zeros(N)
    cp[c .>= 0] = c[c .>= 0]
    cn[c .< 0] = c[c .< 0]
    cn = abs.(cn)
    
    rp = zeros(N)
    rn = zeros(N)
    rp[r .>= 0] = r[r .>= 0]
    rn[r .< 0] = r[r .< 0]
    rn = abs.(rn)

#     normalization
    cp = cp ./ norm(cp,1)
    cn = cn ./ norm(cn,1)
    rp = rp ./ norm(rp,1)
    rn = rn ./ norm(rn,1)

#     compute sinkhorn
    Tp, ap, dp = sinkhorn_basic(rp, cp, M; lambda=lambda, numItermax = numItermax, stopThr = stopThr, verbose=verbose);
    Tn, an, dn = sinkhorn_basic(rn, cn, M; lambda=lambda, numItermax = numItermax, stopThr = stopThr, verbose=verbose);

    T = Tp + Tn;
    a = ap - an;
    d = dp + dn;
    return T, a, d
end

function sinkhorn_signal_2d(f, g, M; lambda=100, numItermax = 1000, stopThr = 1e-6, verbose=false)
    Nx, Ny = size(f)
    N = Nx * Ny
    f = f[:]
    g = g[:]
    # normalization
    fp = zeros(N)
    fn = zeros(N)
    fp[f .>= 0] = f[f .>= 0]
    fn[f .< 0] = f[f .< 0]
    fn = abs.(fn)

    gp = zeros(N)
    gn = zeros(N)
    gp[g .>= 0] = g[g .>= 0]
    gn[g .< 0] = g[g .< 0]
    gn = abs.(gn)

    fp = fp ./ norm(fp,1)
    fn = fn ./ norm(fn,1)
    gp = gp ./ norm(gp,1)
    gn = gn ./ norm(gn,1);
    
    Tp, ap, dp = sinkhorn_basic(fp, gp, M; lambda=lambda, numItermax=numItermax, stopThr=stopThr, verbose=verbose)
    Tn, an, dn = sinkhorn_basic(fn, gn, M; lambda=lambda, numItermax=numItermax, stopThr=stopThr, verbose=verbose)
    
    T = Tp + Tn;
    a = ap - an;
    d = dp + dn;
    return T, a, d
end