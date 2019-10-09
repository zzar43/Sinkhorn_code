# Cost matrix

function cost_func_1d(x; p=2)
#     x = range(0, stop=1, length=N)
    N = length(x)
    # Build loss matrix
    M = zeros(N,N)
    for i = 1:N
        for j = 1:N
            M[i,j] = (x[i] - x[j])^p
        end
    end
    return M
end

function cost_matrix_2d(x, y; p=2)
#     x, y: row vectors, for the domain
    Nx = length(x)
    Ny = length(y)
    X = repeat(x, 1, Ny)
    Y = repeat(y', Nx, 1)
    X = reshape(X, Nx*Ny, 1)
    Y = reshape(Y, Nx*Ny, 1)
    M = zeros(Nx*Ny, Nx*Ny)
    for i = 1:Nx*Ny
        for j = 1:Nx*Ny
            M[i,j] = sqrt((X[i]-X[j])^2 + (Y[i]-Y[j])^2)^p
        end
    end
    
    return M
end

# test functions
function gauss_func(t, b, c)
    y = exp.(-(t.-b).^2 ./ (2*c^2));
    return y
end

function sin_func(t, omega, phi)
    return sin.(2*pi*omega*(t .- phi));
end

function ricker_func(t, t0, sigma)
    t = t.-t0;
    f = (1 .- t.^2 ./ sigma.^2) .* exp.(- t.^2 ./ (2 .* sigma.^2));
    return f
end
function gaussian_2d(X,Y,center,sigma)
    g = exp.(-(X.-center[1]).^2 ./ (sigma[1]^2) -(Y.-center[2]).^2 ./ (sigma[2]^2))
    g = g ./ maximum(g)
    return g
end

function ricker_2d(X,Y,center,sigma)
    g = (1 .- (X.-center[1]).^2 ./ (sigma[1]^2) .- (Y.-center[2]).^2 ./ (sigma[2]^2)) .* exp.(-(X.-center[1]).^2 ./ (sigma[1]^2) -(Y.-center[2]).^2 ./ (sigma[2]^2))
    g = g ./ maximum(g)
    return g
end


# basic shinkhorn functions
function sinkhorn_basic(r, c, M; lambda=100, numItermax=1000, stopThr=1e-6, verbose=false)
    if typeof(r) != Array{Float64,1}
        error("Check type of r and c. Should be Array{Float64,1}.")
    end
    
    T = 0 .* M
    II = findall(x->x>0, r)
    N = length(r)
    Ns = length(II)
    r = r[II]
    M1 = M[II,:]
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

    T[II,:] = diagm(u) * K * diagm(v)
    d = sum(T.*M)
    alpha = zeros(N)
    alpha[II] = -1/lambda .* log.(u)  + (log.(u)'*ones(length(u)))./(lambda*length(u)) .* ones(length(u))
    
    if verbose == true
        println("Iteration number: ", numIter-1)
        println("Error: ", e)
    end
    return T, -alpha, d
end

function sinkhorn_basic_2d(r, c, M; lambda=100, numItermax=1000, stopThr=1e-6, verbose=false)
    Nx, Ny = size(r)
    r = reshape(r, Nx*Ny);
    c = reshape(c, Nx*Ny);
    r = r ./ norm(r,1)
    c = c ./ norm(c,1)
    
    T,a,d = sinkhorn_basic(r, c, M; lambda=lambda, numItermax=numItermax, stopThr=stopThr, verbose=verbose);
    a = reshape(a, Nx, Ny)
    
    return T, a, d
end

# function sinkhorn_signal_1d(r, c, M; lambda=100, numItermax = 1000, stopThr = 1e-6, verbose=false)
    
#     N = length(r)
#     # normalization
#     cp = zeros(N)
#     cn = zeros(N)
#     cp[c .>= 0] = c[c .>= 0]
#     cn[c .< 0] = c[c .< 0]
#     cn = abs.(cn)
    
#     rp = zeros(N)
#     rn = zeros(N)
#     rp[r .>= 0] = r[r .>= 0]
#     rn[r .< 0] = r[r .< 0]
#     rn = abs.(rn)

# #     normalization
#     cp = cp ./ norm(cp,1)
#     cn = cn ./ norm(cn,1)
#     rp = rp ./ norm(rp,1)
#     rn = rn ./ norm(rn,1)

# #     compute sinkhorn
#     Tp, ap, dp = sinkhorn_basic(rp, cp, M; lambda=lambda, numItermax = numItermax, stopThr = stopThr, verbose=verbose);
#     Tn, an, dn = sinkhorn_basic(rn, cn, M; lambda=lambda, numItermax = numItermax, stopThr = stopThr, verbose=verbose);

#     T = Tp + Tn;
#     a = ap - an;
#     d = dp + dn;
#     return T, a, d
# end

# function sinkhorn_signal_2d(f, g, M; lambda=100, numItermax = 1000, stopThr = 1e-6, verbose=false)
#     Nx, Ny = size(f)
#     N = Nx * Ny
#     f = f[:]
#     g = g[:]
#     # normalization
#     fp = zeros(N)
#     fn = zeros(N)
#     fp[f .>= 0] = f[f .>= 0]
#     fn[f .< 0] = f[f .< 0]
#     fn = abs.(fn)

#     gp = zeros(N)
#     gn = zeros(N)
#     gp[g .>= 0] = g[g .>= 0]
#     gn[g .< 0] = g[g .< 0]
#     gn = abs.(gn)

#     fp = fp ./ norm(fp,1)
#     fn = fn ./ norm(fn,1)
#     gp = gp ./ norm(gp,1)
#     gn = gn ./ norm(gn,1);
    
#     Tp, ap, dp = sinkhorn_basic(fp, gp, M; lambda=lambda, numItermax=numItermax, stopThr=stopThr, verbose=verbose)
#     Tn, an, dn = sinkhorn_basic(fn, gn, M; lambda=lambda, numItermax=numItermax, stopThr=stopThr, verbose=verbose)
    
#     T = Tp + Tn;
#     a = ap - an;
#     d = dp + dn;
#     return T, a, d
# end

# function sinkhorn_signal_1d_linear(r, c, M; lambda=100, numItermax = 1000, stopThr = 1e-6, verbose=false)
    
#     N = length(r)
#     # normalization
#     cp = zeros(N)
#     cn = zeros(N)
#     cp[c .>= 0] = c[c .>= 0]
#     cn[c .< 0] = c[c .< 0]
#     cn = abs.(cn)
    
#     rp = zeros(N)
#     rn = zeros(N)
#     rp[r .>= 0] = r[r .>= 0]
#     rn[r .< 0] = r[r .< 0]
#     rn = abs.(rn)

# #     normalization
#     cp = cp ./ norm(cp,1)
#     cn = cn ./ norm(cn,1)
#     rp = rp ./ norm(rp,1)
#     rn = rn ./ norm(rn,1)

# #     compute sinkhorn
#     Tp, ap, dp = sinkhorn_basic(rp, cp, M; lambda=lambda, numItermax = numItermax, stopThr = stopThr, verbose=verbose);
#     Tn, an, dn = sinkhorn_basic(rn, cn, M; lambda=lambda, numItermax = numItermax, stopThr = stopThr, verbose=verbose);

#     T = Tp + Tn;
#     a = ap - an;
#     d = dp + dn;
#     return T, a, d
# end

# Linear normalization
function sinkhorn_signal_1d_linear(r, c, M; balance_coef=1, lambda=100, numItermax = 1000, stopThr = 1e-6, verbose=false)
    mi = min(minimum(r), minimum(c))
    r = r .- mi .+ balance_coef
    c = c .- mi .+ balance_coef
    r = r ./ norm(r,1)
    c = c ./ norm(c,1)

    T, alpha, d = sinkhorn_basic(r, c, M; lambda=lambda, numItermax = numItermax, stopThr = stopThr, verbose=verbose);
    
    return T, alpha, d
end

function sinkhorn_signal_2d_linear(r, c, M; balance_coef=1, lambda=100, numItermax = 1000, stopThr = 1e-6, verbose=false)
    Nx, Ny = size(r)
    r = reshape(r, Nx*Ny)
    c = reshape(c, Nx*Ny)
    
    mi = min(minimum(r), minimum(c))
    r = r .- mi .+ balance_coef
    c = c .- mi .+ balance_coef
    r = r ./ norm(r,1)
    c = c ./ norm(c,1)

    T, alpha, d = sinkhorn_basic(r, c, M; lambda=lambda, numItermax = numItermax, stopThr = stopThr, verbose=verbose);
    alpha = reshape(alpha, Nx, Ny)
    
    return T, alpha, d
end



# Normalization with adding a term
# function sinkhorn_normalized_add(r, c, M; balance_coef=10, lambda=100, numItermax=1000, stopThr=1e-6, verbose=false)
#     N = length(r)
#     r1 = r ./ norm(r,1)
#     c1 = c ./ norm(c,1)
#     T, a, d = sinkhorn_basic(r1, c1, M; lambda=lambda, numItermax=numItermax, stopThr=stopThr, verbose=verbose)
    
#     aaa = - (1 ./ sum(r)^2) * r * ones(N)' + 1 ./ sum(r) * I(N)
#     grad1 = a' * aaa
#     if sum(r) > sum(c)
#         grad2 = balance_coef * ones(N)'
#     elseif sum(r) < sum(c)
#         grad2 = -balance_coef * ones(N)'
#     else
#         grad2 = 0 * ones(N)'
#     end
#     grad = grad1 + grad2
    
#     dist = d + balance_coef*abs(sum(r)-sum(c))
#     return T, grad, dist
# end

# that might be slow when N is large
function proj_p(f)
    N = length(f)
    Pp = zeros(N)
    p_ind = findall(x->x>=0, f)
    Pp[p_ind] .= 1
    Pp = diagm(Pp)
    
    return Pp
end

function proj_n(f)
    N = length(f)
    Pn = zeros(N)
    p_ind = findall(x->x<0, f)
    Pn[p_ind] .= -1
    Pn = diagm(Pn)
    
    return Pn
end

function sinkhorn_signal_1d_nor_add(r, c, M; balance_coef=0, lambda=100, numItermax=1000, stopThr=1e-6, verbose=false)
#     math form: d(r,c) = W_2^2(r,c) + balance_coef*(\|r\| - \|c\|)
#     balance_coef = 0: sinkhorn distance with + and - normalization
    N = length(r)
    Prp = proj_p(r)
    Prn = proj_n(r)
    Pcp = proj_p(c)
    Pcn = proj_n(c)

    rp = Prp * r
    rn = Prn * r
    cp = Pcp * c
    cn = Pcn * c
    
    rp1 = rp ./ norm(rp,1)
    cp1 = cp ./ norm(cp,1)
    Tp, ap, dp = sinkhorn_basic(rp1, cp1, M; lambda=1e3, numItermax=1000, stopThr=1e-6, verbose=false)
    aap = - (1 ./ sum(rp)^2) * r * ones(N)' + 1 ./ sum(rp) * I(N)
    gradp1 = ap' * aap * Prp

    if sum(rp) > sum(cp)
        gradp2 = balance_coef * ones(N)' * Prp
    elseif sum(r) < sum(c)
        gradp2 = -balance_coef * ones(N)' * Prp
    else
        gradp2 = 0 * ones(N)'
    end

    rn1 = rn ./ norm(rn,1)
    cn1 = cn ./ norm(cn,1)
    Tn, an, dn = sinkhorn_basic(-rn1, -cn1, M; lambda=1e3, numItermax=1000, stopThr=1e-6, verbose=false)
    aan = - (1 ./ sum(rn)^2) * r * ones(N)' + 1 ./ sum(rn) * I(N)
    gradn1 = an' * aan * (-Prn)

    if sum(-rn) > sum(-cn)
        gradn2 = balance_coef * ones(N)' * (-Prn)
    elseif sum(-rn) < sum(-cn)
        gradn2 = -balance_coef * ones(N)' * (-Prn)
    else
        gradn2 = 0 * ones(N)'
    end

    grad = gradp1 + gradp2 + gradn1 + gradn2
    dist = dp + dn + balance_coef*abs(sum(rp)-sum(cp)) + balance_coef*abs(sum(-rn)-sum(-cn))
    
    return Tp-Tn, grad, dist
end

function sinkhorn_signal_2d_nor_add(r, c, M; balance_coef=0, lambda=100, numItermax=1000, stopThr=1e-6, verbose=false)
    
    Nx, Ny = size(r)
    
    r = reshape(r, Nx*Ny)
    c = reshape(c, Nx*Ny)
    
    N = length(r)
    Prp = proj_p(r)
    Prn = proj_n(r)
    Pcp = proj_p(c)
    Pcn = proj_n(c)

    rp = Prp * r
    rn = Prn * r
    cp = Pcp * c
    cn = Pcn * c
    
    rp1 = rp ./ norm(rp,1)
    cp1 = cp ./ norm(cp,1)
    Tp, ap, dp = sinkhorn_basic(rp1, cp1, M; lambda=1e3, numItermax=1000, stopThr=1e-6, verbose=false)
    aap = - (1 ./ sum(rp)^2) * r * ones(N)' + 1 ./ sum(rp) * I(N)
    gradp1 = ap' * aap * Prp

    if sum(rp) > sum(cp)
        gradp2 = balance_coef * ones(N)' * Prp
    elseif sum(r) < sum(c)
        gradp2 = -balance_coef * ones(N)' * Prp
    else
        gradp2 = 0 * ones(N)'
    end

    rn1 = rn ./ norm(rn,1)
    cn1 = cn ./ norm(cn,1)
    Tn, an, dn = sinkhorn_basic(-rn1, -cn1, M; lambda=1e3, numItermax=1000, stopThr=1e-6, verbose=false)
    aan = - (1 ./ sum(rn)^2) * r * ones(N)' + 1 ./ sum(rn) * I(N)
    gradn1 = an' * aan * (-Prn)

    if sum(-rn) > sum(-cn)
        gradn2 = balance_coef * ones(N)' * (-Prn)
    elseif sum(-rn) < sum(-cn)
        gradn2 = -balance_coef * ones(N)' * (-Prn)
    else
        gradn2 = 0 * ones(N)'
    end

    grad = gradp1 + gradp2 + gradn1 + gradn2
    dist = dp + dn + balance_coef*abs(sum(rp)-sum(cp)) + balance_coef*abs(sum(-rn)-sum(-cn))
    
    grad = reshape(grad, Nx, Ny)
    return Tp-Tn, grad, dist
end

# New way to normalize
# d(f,g) = \theta W_2^2\left(\frac{\hat f}{\|\hat f\|_1}, \frac{\hat g}{\|\hat g\|_1}\right) + (1-\theta) \left| \|\hat f\|_1 - \|\hat g\|_1 \right|.
# For 1d signal, 2d signal and 2d positive function

function find_support_1d(f; threshold=1e-5)
    N = length(f)
    ind_f = zeros(Int, N)
    for i = 2:(N-1)
        if (abs(f[i-1])>threshold) || (abs(f[i])>threshold) || (abs(f[i+1])>threshold)
            ind_f[i] = 1
        end
    end
    supp_f = findall(x->x!=0, ind_f)
    return supp_f
end

function find_support_2d(f; threshold=1e-5)
    Nx, Ny = size(f)
    ind_f = zeros(Int, Nx, Ny)
    for i = 2:(Nx-1)
        for j = 2:(Ny-1)
            side = (abs(f[i-1,j])>threshold) || (abs(f[i+1,j])>threshold) || (abs(f[i,j+1])>threshold) || (abs(f[i,j-1])>threshold)
            if (abs(f[i,j])>threshold) && side
                ind_f[i,j] = 1
            elseif (abs(f[i-1,j])>threshold) && (abs(f[i+1,j])>threshold) && (abs(f[i,j+1])>threshold) && (abs(f[i,j-1])>threshold)
                ind_f[i,j] = 1
            end
        end
    end
    supp_f = findall(x->x!=0, ind_f)
    return supp_f
end

function normalized_sinkhorn_1d(r, c, M; theta=1e-3, lambda=1e3, numItermax=1000, stopThr=1e-6, verbose=false)
    f = copy(r)
    g = copy(c)
    N = length(f)
    
    T, grad0, dist0 = sinkhorn_basic(f./norm(f,1), g./norm(g,1), M; lambda=lambda, numItermax=numItermax, stopThr=stopThr, verbose=verbose)
    
#     gradient
    temp_coef = -1 ./ norm(f,1)^2 * f * sign.(f)' + 1/norm(f,1) * I(N)
    grad1 = grad0' * temp_coef
#     grad2 = sign(norm(g,1)-norm(f,1)) * sign.(f)'
    grad2 = sign(norm(f,1)-norm(g,1)) * sign.(f)'
    grad = (1-theta) * grad1 + theta * grad2
    
#     distant
    dist = (1-theta) * dist0 + theta * abs(norm(f,1)-norm(g,1))
    return T, grad, dist
end

function normalized_sinkhorn_signal_1d(r, c, M; theta=1e-3, lambda=1e3, numItermax=1000, stopThr=1e-6, verbose=false)
    f = copy(r)
    g = copy(c)
    
    Pfp = proj_p(f)
    Pfn = proj_n(f)
    Pgp = proj_p(g)
    Pgn = proj_n(g)

    fp = Pfp * f
    fn = Pfn * f
    gp = Pgp * g
    gn = Pgn * g
    
    Tp, ggp, dp = normalized_sinkhorn_1d(fp, gp, M; theta=theta, lambda=lambda, numItermax=numItermax, stopThr=stopThr, verbose=verbose)
    Tn, ggn, dn = normalized_sinkhorn_1d(fn, gn, M; theta=theta, lambda=lambda, numItermax=numItermax, stopThr=stopThr, verbose=verbose)
    
    grad = ggp * Pfp + ggn * Pfn
    dist = dp + dn
    T = Tp*Pfp + Tn*Pfn
    
    return T, grad, dist
end

# function normalized_sinkhorn_signal_1d(r, c, M; theta=0.5, normal_coef=0, lambda=1e3, numItermax=1000, stopThr=1e-6, verbose=false, supp_th=1e-10)
#     f = copy(r)
#     g = copy(c)
    
#     N = length(f)
#     # normalization
#     if normal_coef == 0
#         mi = 2 * abs(min(minimum(f), minimum(g)))
#     elseif abs(min(minimum(f), minimum(g))) >= normal_coef
#         error("Please increase the normalization coef.")
#     else
#         mi = normal_coef
#     end
    
#     if supp_th == 0
#         supp_f = 1:length(f)
#         supp_g = 1:length(g)
#     else
#         supp_f = find_support_1d(f; threshold=supp_th)
#         supp_g = find_support_1d(g; threshold=supp_th)
#     end
    
#     f_hat = zeros(N)
#     g_hat = zeros(N)
#     f_hat[supp_f] = f[supp_f] .+ mi
#     g_hat[supp_g] = g[supp_g] .+ mi
    
#     # sinkrhon
#     T, gg, dist1 = sinkhorn_basic(f_hat./norm(f_hat,1), g_hat./norm(g_hat,1), M; lambda=lambda, numItermax=numItermax, stopThr=stopThr, verbose=verbose);
    
#     temp1 = -1 ./ (norm(f_hat,1)^2) * f_hat * sign.(f_hat)' + (1 ./ norm(f_hat,1)) * I(N)
#     grad1 = gg' * temp1

#     if norm(f_hat,1) > norm(g_hat,1)
#         grad2 = sign.(f_hat)'
#     elseif norm(f_hat,1) < norm(g_hat,1)
#         grad2 = -1 * sign.(f_hat)'
#     else
#         grad2 = zeros(N)'
#     end

#     grad = (1-theta) * grad1 + theta * grad2
#     dist = (1-theta) * dist1 + theta * abs(norm(f_hat,1)-norm(g_hat,1))
    
#     return T, grad, dist
# end

# function normalized_sinkhorn_signal_2d(r, c, M; theta=0.5, normal_coef=0, lambda=1e3, numItermax=100, stopThr=1e-6, verbose=false, supp_th=1e-5)
#     f = copy(r)
#     g = copy(c)
    
#     Nx, Ny = size(f)
    
#     # normalization
#     if normal_coef == 0
#         mi = 2 * abs(min(minimum(f), minimum(g)))
#     elseif abs(min(minimum(f), minimum(g))) >= normal_coef
#         error("Please increase the normalization coef.")
#     else
#         mi = normal_coef
#     end
    
#     if supp_th == 0
#         supp_f = 1:length(f)
#         supp_g = 1:length(g)
#     else
#         supp_f = find_support_1d(f; threshold=supp_th)
#         supp_g = find_support_1d(g; threshold=supp_th)
#     end
    
#     f_hat = zeros(Nx,Ny)
#     g_hat = zeros(Nx,Ny)
#     f_hat[supp_f] = f[supp_f] .+ mi
#     g_hat[supp_g] = g[supp_g] .+ mi
    
#     # sinkrhon
#     T, gg, dist1 = sinkhorn_basic_2d(f_hat./norm(f_hat,1), g_hat./norm(g_hat,1), M; lambda=lambda, numItermax=numItermax, stopThr=stopThr, verbose=verbose);
    
#     temp1 = -1 ./ (norm(f_hat[:],1)^2) * f_hat[:] * sign.(f_hat[:])' + (1 ./ norm(f_hat[:],1)) * I(Nx*Ny)
#     grad1 = reshape(gg, Nx*Ny)' * temp1
#     grad1 = reshape(grad1, Nx, Ny)

#     if norm(f_hat,1) > norm(g_hat,1)
#         grad2 = sign.(f_hat)
#     elseif norm(f_hat,1) < norm(g_hat,1)
#         grad2 = -1 * sign.(f_hat)
#     else
#         grad2 = zeros(Nx,Ny)
#     end

#     grad = (1-theta) * grad1 + theta * grad2
#     dist = (1-theta) * dist1 + theta * abs(norm(f_hat,1)-norm(g_hat,1))
    
#     return T, grad, dist
# end

# function normalized_sinkhorn_2d(r, c, M; theta=0.5, lambda=1e3, numItermax=100, stopThr=1e-6, verbose=false)
#     f = copy(r)
#     g = copy(c)
    
#     Nx, Ny = size(f)
    
#     # sinkrhon
#     T, gg, dist1 = sinkhorn_basic_2d(f./(norm(f,1)), g./(norm(g,1)), M; lambda=lambda, numItermax=numItermax, stopThr=stopThr, verbose=verbose);
    
#     temp1 = -1 ./ (norm(f[:],1)^2) * f[:] * sign.(f[:])' + (1 ./ norm(f[:],1)) * I(Nx*Ny)
#     grad1 = reshape(gg, Nx*Ny)' * temp1
#     grad1 = reshape(grad1, Nx, Ny)

#     if norm(f,1) > norm(g,1)
#         grad2 = sign.(f)
#     elseif norm(f,1) < norm(g,1)
#         grad2 = -1 * sign.(f)
#     else
#         grad2 = zeros(Nx,Ny)
#     end

#     grad = (1-theta) * grad1 + theta * grad2
#     dist = (1-theta) * dist1 + theta * abs(norm(f,1)-norm(g,1))
    
#     return T, grad, dist
# end
