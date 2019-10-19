using LinearAlgebra
include("sinkhorn.jl")

# barycenter with entropy regularization
function barycenter_sinkhorn_1d(P, M, reg; lambda=0, iterMax=100, errThr=1e-5, verbose=true)
    # M = M ./ maximum(M)
    N, Nd = size(P)
    if N != size(M,1) || N != size(M,2)
        error("Please check the size of cost matrix and P.")
    end
    
    if lambda == 0
        lambda = ones(Nd) ./ Nd
    elseif length(lambda) == Nd
        if sum(lambda) != 1
            lambda = lambda ./ sum(lambda)
        end
    else
        error("Please check the size of lambda. length(lambda) != size(P,2).")
    end
    
    P = P ./ sum(P,dims=1)
    
    # initialize
    U = ones(N,Nd) ./ N
    V = ones(N,Nd) ./ N
    K = exp.(-M ./ reg)

    p = ones(N)
    for i = 1:Nd
        p = p .* (U[:,i] .* K .* V[:,i]' * ones(N)).^lambda[i]
    end
    
    # main iteration
    p0 = ones(N) ./ N
    for iter = 1:iterMax
        p0 = copy(p)

        U = p ./ (K * V)
        V = P ./ (K' * U)

        p = ones(N)
        for i = 1:Nd
            p = p .* (U[:,i] .* K .* V[:,i]' * ones(N)).^lambda[i]
        end
        if any(isnan.(p)) || any(isinf.(p))
            if verbose == true
                println("Numerical error. Try to increase reg.")
            end
            p = copy(p0)
            break
        end
    end
    
    err = abs(1 - norm(p,1))
    if verbose==true && err>errThr
        println("Not converge. Try to increase iterMax.")
    end
    
    return p
end

# unbalanced barycenter with entropy regularization
function barycenter_unbalanced_1d(P, M, reg, reg_m; lambda=0, iterMax=100, verbose=true)
    N, Nd = size(P)
    if N != size(M,1) || N != size(M,2)
        error("Please check the size of cost matrix and P.")
    end

    if lambda == 0
        lambda = ones(Nd) ./ Nd
    elseif length(lambda) == Nd
        if sum(lambda) != 1
            lambda = lambda ./ sum(lambda)
        end
    else
        error("Please check the size of lambda. length(lambda) != size(P,2).")
    end
    
    # initialize
    U = ones(N,Nd) ./ N
    V = ones(N,Nd) ./ N
    K = exp.(-M ./ reg)
    fi = reg_m / (reg + reg_m)

    p = zeros(N)
    for i = 1:Nd
        p = p + (U[:,i] .* K .* V[:,i]' * ones(N)).^(1-fi) .* lambda[i]
    end
    p = p .^ (1/(1-fi));
    
    # main iteration
    p0 = ones(N) ./ N
    for iter = 1:iterMax
        p0 = copy(p)

        U = p ./ (K * V)
        U = U .^ fi
        V = P ./ (K' * U)
        V = V .^ fi

        p = zeros(N)
        for i = 1:Nd
            p = p + (U[:,i] .* K .* V[:,i]' * ones(N)).^(1-fi) .* lambda[i]
        end
        p = p .^ (1/(1-fi));
        if any(isnan.(p)) || any(isinf.(p))
            if verbose == true
                println("Numerical error. Try to increase reg.")
            end
            p = copy(p0)
            break
        end
    end

#     T = zeros(N,N,Nd)
#     for i = 1:Nd
#        T[:,:,i] = U[:,i] .* K .* V[:,i]'
#     end
    
#     return p, T
    return p
end

# Barycenter, signal, entropy
function barycenter_unbalanced_1d_signal(P, M, reg, reg_m; lambda=0, iterMax=100, verbose=false)
    N, Nd = size(P)
    Pp = zeros(N,Nd)
    Pn = zeros(N,Nd)
    Pp[findall(x->x>0, P)] = P[findall(x->x>0, P)]
    Pn[findall(x->x<0, P)] = -P[findall(x->x<0, P)]

    pp = barycenter_unbalanced_1d(Pp, M, reg, reg_m; lambda=lambda, iterMax=iterMax, verbose=verbose)
    pn = barycenter_unbalanced_1d(Pn, M, reg, reg_m; lambda=lambda, iterMax=iterMax, verbose=verbose)

    return pp - pn
end