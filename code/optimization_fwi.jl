using Printf

function line_search_backtracking(op_fn, xk, fk, gradk, alpha, min_value, max_value; rho=0.9, c=0.9, maxSearchTime=30)
    pk = -gradk
    @printf "Start line search. fk: %1.5e\n" fk
    xkk = update_fn(xk, alpha, gradk, min_value, max_value)
    fk1, gradk1 = op_fn(xkk)
    @printf "    alpha: %1.5e" alpha
    @printf "    fk1: %1.5e" fk1
    @printf "    fk-c*alpha*gradk^2: %1.5e\n" (fk + c*alpha*sum(gradk.*pk))
    
    searchTime = 0
    for iter = 1:maxSearchTime
        if fk1 <= (fk + c*alpha*sum(gradk.*pk))
            break
        end
        alpha = rho * alpha
        xkk = update_fn(xk, alpha, gradk, min_value, max_value)
        fk1, gradk1 = op_fn(xkk)   
        @printf "    alpha: %1.5e" alpha
        @printf "    fk1: %1.5e" fk1
        @printf "    fk-c*alpha*gradk^2: %1.5e\n" (fk + c*alpha*sum(gradk.*pk))
        searchTime += 1
    end
    if fk1 > fk + c*alpha*sum(gradk.*pk)
        println("Line search failed. Search time: ", searchTime, ". Try to decrease search coef c.")
        alpha = 0
    else
        println("Line search succeed. Search time: ", searchTime, ".")
    end

    return alpha
end

function gradient_descent(fn, x0, alpha, iterNum, min_value, max_value; rho=0.9, c=0.9, maxSearchTime=30, threshold=1e-5)

    xk = convert(Array{Float64,1}, x0[:])
    fn_value = zeros(iterNum+1)
    
    fk, gradk = fn(xk)
    fn_value[1] = fk
    
    for iter = 1:iterNum
        println("Main iteration: ", iter)
        
#         Line search
        alpha0 = line_search_backtracking(fn, xk, fk, gradk, alpha, min_value, max_value; rho=rho, c=c, maxSearchTime=maxSearchTime)
#         println(alpha0)
        if alpha0 == 0
            println("----------------------------------------------------------------")
            println("Line search Failed. Try decrease line search coef alpha. Interupt.")
            println("----------------------------------------------------------------")
            break
        else
            xk = update_fn(xk, alpha0, gradk, min_value, max_value)
        end
        
#         Compute gradient for next iteration
        fk, gradk = fn(xk)
        fn_value[iter+1] = fk
        println("----------------------------------------------------------------")
        if fk <= threshold
            @printf "fk: %1.5e " fk
            println("Iteration is done.")
            println("----------------------------------------------------------------\n")
            break
        end
        if iter == iterNum 
            @printf "fk: %1.5e " fk
            println("Iteration is done. \n")
            println("----------------------------------------------------------------\n")
        end
    end

    return xk, fn_value
end

function gradient_descent_test(fn, x0, alpha, iterNum, min_value, max_value; rho=0.9, c=0.9, threshold=1e-5)
    
    xk = convert(Array{Float64,1}, x0[:])
    fn_value = zeros(iterNum+1)
    
    fk, gradk = fn(xk)
    gradk = gradk ./ maximum(abs.(gradk))
#     println(maximum(gradk))
    
    for iter = 1:iterNum
        println("Main iteration: ", iter)
        xk = update_fn(xk, alpha, gradk, min_value, max_value)
        
#         Compute gradient for next iteration
        fk, gradk = fn(xk)
        gradk = gradk ./ maximum(abs.(gradk))
        fn_value[iter+1] = fk
        @printf "fk: %1.5e " fk
        println("----------------------------------------------------------------")
        if fk <= threshold
            @printf "fk: %1.5e " fk
            println("Iteration is done.")
            println("----------------------------------------------------------------\n")
            break
        end
        if iter == iterNum 
            @printf "fk: %1.5e " fk
            println("Iteration is done. \n")
            println("----------------------------------------------------------------\n")
        end
    end

    return xk, fn_value
end

function nonlinear_cg(fn, x0, alpha, iterNum, min_value, max_value; rho=0.9, c=0.9, maxSearchTime=30, threshold=1e-5)
    xk = convert(Array{Float64,1}, x0[:])
    fn_value = zeros(iterNum+1)

    fk, gradk = fn(xk)
    fn_value[1] = fk
    d0 = -gradk
    r0 = -gradk
    
    iter = 1
    println("Main iteration: ", iter)
    alpha0 = line_search_backtracking(fn, xk, fk, -d0, alpha, min_value, max_value; rho=rho, c=c, maxSearchTime=maxSearchTime)

    if alpha0 == 0
        println("----------------------------------------------------------------")
        println("Line search Failed. Try decrease line search coef alpha. Interupt.")
        println("----------------------------------------------------------------")
    else

    #     update
        xk = update_fn(xk, alpha0, gradk, min_value, max_value)
    #     compute gradient for next iteration
        fk, gradk = fn(xk)
        fn_value[2] = fk
        r1 = -gradk
        beta = (r1'*(r1-r0))/(r0'*r0)
        beta = max(beta, 0)
        d1 = r1 + beta*d0
        println("----------------------------------------------------------------")

        for iter = 2:iterNum
            println("Main iteration: ", iter)
    #         line search
            alpha0 = line_search_backtracking(fn, xk, fk, -d1, alpha, min_value, max_value; rho=rho, c=c, maxSearchTime=maxSearchTime)
    #         update
            if alpha0 == 0
                println("----------------------------------------------------------------")
                println("Line search Failed. Try decrease line search coef alpha. Interupt.")
                println("----------------------------------------------------------------")
                break
            else
                xk = update_fn(xk, alpha0, gradk, min_value, max_value)
            end
            r0[:] = r1[:]
            d0[:] = d1[:]
    #     compute gradient for next iteration
            fk, gradk = fn(xk)
            fn_value[iter+1] = fk
            r1 = -gradk
            beta = (r1'*(r1-r0))/(r0'*r0)
            beta = max(beta, 0)
            d1 = r1 + beta*d0

            println("----------------------------------------------------------------")
            if fk <= threshold
                @printf "fk: %1.5e " fk
                println("Iteration is done.")
                println("----------------------------------------------------------------\n")
                break
            end
            if iter == iterNum 
                @printf "fk: %1.5e " fk
                println("Iteration is done. \n")
                println("----------------------------------------------------------------\n")
            end
        end
    end

    return xk, fn_value
end

function update_fn(xk, alphak, gradk, min_value, max_value)
    xk1 = xk - alphak * gradk
    xk1[findall(ind->ind<min_value,xk1)] .= min_value
    xk1[findall(ind->ind>max_value,xk1)] .= max_value
    return xk1
end

function obj_fn(data0, u, c, rho, Nx, Ny, Nt, h, dt, source, source_position, receiver_position; pml_len=10, pml_coef=100)
    x = reshape(c, Nx, Ny)

    data, u = multi_solver(x, rho, Nx, Ny, Nt, h, dt, source, source_position, receiver_position; pml_len=pml_len, pml_coef=pml_coef)

    fk = 0.5 * norm(data - data0) ^ 2 * dt

    gradk = grad_l2(data, u, data0, x, rho, Nx, Ny, Nt, h, dt, source_position, receiver_position; pml_len=pml_len, pml_coef=pml_coef);
    
    gradk = reshape(gradk, Nx*Ny, 1)
    
    return fk, gradk
end

function obj_fn_parallel(data0, c, rho, Nx, Ny, Nt, h, dt, source, source_position, receiver_position; pml_len=10, pml_coef=100)
    x = reshape(c, Nx, Ny)

    data, u = multi_solver_parallel(x, rho, Nx, Ny, Nt, h, dt, source, source_position, receiver_position; pml_len=pml_len, pml_coef=pml_coef)

    fk = 0.5 * norm(data - data0) ^ 2 * dt

    gradk = grad_l2_parallel(data, u, data0, x, rho, Nx, Ny, Nt, h, dt, source_position, receiver_position; pml_len=pml_len, pml_coef=pml_coef);
    
    gradk = reshape(gradk, Nx*Ny, 1)
    return fk, gradk
end

function obj_fn_sinkhorn_parallel(data0, c, rho, Nx, Ny, Nt, h, dt, source, source_position, receiver_position; pml_len=10, pml_coef=100, reg=5e-3, reg_m=1e2, reg_p=0, iterMax=100, verbose=false)
    x = reshape(c, Nx, Ny)

    data, u = multi_solver_parallel(x, rho, Nx, Ny, Nt, h, dt, source, source_position, receiver_position; pml_len=pml_len, pml_coef=pml_coef)

#     gradk, fk = grad_sinkhorn_parallel(data, u, data0, x, rho, Nx, Ny, Nt, h, dt, source_position, receiver_position; pml_len=pml_len, pml_coef=pml_coef, lambda=lambda, numItermax=numItermax, stopThr=stopThr)
    gradk, fk = grad_sinkhorn_parallel(data, u, data0, c, rho, Nx, Ny, Nt, h, dt, source_position, receiver_position; reg_p=reg_p, pml_len=pml_len, pml_coef=pml_coef, reg=reg, reg_m=reg_m, iterMax=iterMax, verbose=verbose)

    gradk = reshape(gradk, Nx*Ny, 1)
    return fk, gradk
end




# using Printf


# function line_search_backtracking(op_fn, xk, fk, gradk, alpha, min_value, max_value; rho=0.9, c=1e-4, maxSearchTime=30)
# #     print("fk: ", fk)
#     @printf "Start line search. alpha: %0.5e" alpha
#     @printf ". fk: %0.5e" fk
#     pk = -gradk
    
#     xkk = xk+alpha*pk
#     xkk[findall(ind->ind<min_value,xkk)] .= min_value
#     xkk[findall(ind->ind>max_value,xkk)] .= max_value
#     fk1, gradk1 = op_fn(xkk)
#     @printf ". fk1: %0.5e.\n" fk1

#     searchTime = 0
#     for iter = 1:maxSearchTime
#         if fk1 < fk + c*alpha*sum(gradk.*pk)
#             break
#         end
#         alpha = rho * alpha
        
#         xkk = xk+alpha*pk
#         xkk[findall(ind->ind<min_value,xkk)] .= min_value
#         xkk[findall(ind->ind>max_value,xkk)] .= max_value
#         fk1, gradk1 = op_fn(xkk)
        
#         searchTime += 1
# #         println("Search time: ", searchTime, ". alpha: ", alpha, ". fk1: ", fk1)
#         @printf "Search time: %d" searchTime 
#         @printf ". alpha: %0.5e" alpha 
#         @printf ". fk1: %0.5e\n" fk1
#     end

#     if fk1 >= fk + c*alpha*sum(gradk.*pk)
#         println("Line search failed. Search time: ", searchTime, ". Try to decrease search coef alpha, rho, c.")
#         xk1 = xk
#         fk1 = fk
#         gradk1 = gradk
#     elseif fk1 < fk + c*alpha*sum(gradk.*pk)
#         println("Line search succeed. Search time: ", searchTime, ".")
#         xk1 = xkk
#     end
    
#     return xk1, fk1, gradk1
# end

# function gradient_descent(fn, x0, alpha, iterNum, min_value, max_value; rho=0.9, c=0.9, maxSearchTime=30)
# #     fn: returns both value of f and gradient of f
#     xk = x0[:]
#     fn_value = zeros(iterNum+1)
    
#     fk, gradk = fn(xk)
#     fn_value[1] = fk
    
#     for iter = 1:iterNum
#         println("Main iteration: ", iter)
        
#         xk1, fk, gradk = line_search_backtracking(fn, xk, fk, gradk, alpha, min_value, max_value; rho=rho, c=c, maxSearchTime=maxSearchTime)
        
#         if xk1 == xk
#             xk[findall(ind->ind<min_value,xk)] .= min_value
#             xk[findall(ind->ind>max_value,xk)] .= max_value
#             println("Line search Failed. Try decrease line search coef alpha. Interupt.")
#             println("--------------------------------")
#             break
#         else
#             xk[:] = xk1[:]
#             xk[findall(ind->ind<min_value,xk)] .= min_value
#             xk[findall(ind->ind>max_value,xk)] .= max_value
#         end
        
#         fn_value[iter+1] = fk
#         println("--------------------------------")

#     end
#     return xk, fn_value
# end

# function obj_fn(data, data0, u, c, rho, Nx, Ny, Nt, h, dt, source, source_position, receiver_position; pml_len=10, pml_coef=100)
#     x = reshape(c, Nx, Ny)

#     data, u = multi_solver(x, rho, Nx, Ny, Nt, h, dt, source, source_position, receiver_position; pml_len=pml_len, pml_coef=pml_coef)

#     fk = 0.5 * norm(data - data0) ^ 2 * dt

#     gradk = grad_l2(data, u, data0, x, rho, Nx, Ny, Nt, h, dt, source_position, receiver_position; pml_len=pml_len, pml_coef=pml_coef);

#     gradk = reshape(gradk, Nx*Ny, 1)
#     return fk, gradk
# end

# function obj_fn_parallel(data0, c, rho, Nx, Ny, Nt, h, dt, source, source_position, receiver_position; pml_len=10, pml_coef=100)
#     x = reshape(c, Nx, Ny)

#     data, u = multi_solver_parallel(x, rho, Nx, Ny, Nt, h, dt, source, source_position, receiver_position; pml_len=pml_len, pml_coef=pml_coef)

#     fk = 0.5 * norm(data - data0) ^ 2 * dt

#     gradk = grad_l2_parallel(data, u, data0, x, rho, Nx, Ny, Nt, h, dt, source_position, receiver_position; pml_len=pml_len, pml_coef=pml_coef);

#     gradk = reshape(gradk, Nx*Ny, 1)
#     return fk, gradk
# end

# function obj_fn_sinkhorn_parallel(data0, u, c, rho, Nx, Ny, Nt, h, dt, source, source_position, receiver_position; pml_len=10, pml_coef=100, lambda=10, numItermax=10, stopThr = 1e-6)
#     x = reshape(c, Nx, Ny)

#     data, u = multi_solver_parallel(x, rho, Nx, Ny, Nt, h, dt, source, source_position, receiver_position; pml_len=pml_len, pml_coef=pml_coef)

#     gradk, fk = grad_sinkhorn_parallel(data, u, data0, x, rho, Nx, Ny, Nt, h, dt, source_position, receiver_position; pml_len=pml_len, pml_coef=pml_coef, lambda=lambda, numItermax=numItermax, stopThr=stopThr)

#     gradk = reshape(gradk, Nx*Ny, 1)
#     return fk, gradk
# end
