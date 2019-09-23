function line_search_backtracking(op_fn, xk, fk, gradk, alpha, min_value, max_value; rho=0.9, c=1e-4, maxSearchTime=30)
    print("Start line search, fk: ", fk)
    pk = -gradk
    
    xkk = xk+alpha*pk
    xkk[findall(ind->ind<min_value,xk)] .= min_value
    xkk[findall(ind->ind>max_value,xk)] .= max_value
    fk1, gradk1 = op_fn(xkk)
    println(". fk1: ", fk1)

    searchTime = 0
    for iter = 1:maxSearchTime
        if fk1 < fk + c*alpha*sum(gradk.*pk)
            break
        end
        alpha = rho * alpha
        
        xkk = xk+alpha*pk
        xkk[findall(ind->ind<min_value,xk)] .= min_value
        xkk[findall(ind->ind>max_value,xk)] .= max_value
        fk1, gradk1 = op_fn(xkk)
        
        searchTime += 1
        println("Search time: ", searchTime, ". alpha: ", alpha, ". fk1: ", fk1)
    end

    if searchTime == maxSearchTime
        println("Line search failed. Search time: ", searchTime, ". Try to decrease search coef alpha, rho, c.")
        xk1 = xk
        fk1 = fk
        gradk1 = gradk
    elseif searchTime < maxSearchTime
        println("Line search succeed. Search time: ", searchTime, ".")
        xk1 = xkk
    end
    
    return xk1, fk1, gradk1
end

function gradient_descent(fn, x0, alpha, iterNum, min_value, max_value; rho=0.9, c=0.9, maxSearchTime=30)
#     fn: returns both value of f and gradient of f
    xk = x0[:]
    fn_value = zeros(iterNum+1)
    
    fk, gradk = fn(xk)
    fn_value[1] = fk
    
    for iter = 1:iterNum
        println("Main iteration: ", iter)
        
        xk1, fk, gradk = line_search_backtracking(fn, xk, fk, gradk, alpha, min_value, max_value; rho=rho, c=c, maxSearchTime=maxSearchTime)
        
        if xk1 == xk
            xk[findall(ind->ind<min_value,xk)] .= min_value
            xk[findall(ind->ind>max_value,xk)] .= max_value
            println("Line search Failed. Try decrease line search coef alpha. Interupt.")
            println("--------------------------------")
            break
        else
            xk[findall(ind->ind<min_value,xk)] .= min_value
            xk[findall(ind->ind>max_value,xk)] .= max_value
            xk = xk1
        end
        
        fn_value[iter+1] = fk
        println("--------------------------------")

    end
    return xk, fn_value
end

function obj_fn(c, rho, Nx, Ny, Nt, h, dt, source, source_position, receiver_position; pml_len=10, pml_coef=100)
    x = reshape(c, Nx, Ny)

    data, u = multi_solver(x, rho, Nx, Ny, Nt, h, dt, source, source_position, receiver_position; pml_len=pml_len, pml_coef=pml_coef)

    fk = 0.5 * norm(data - data0) ^ 2

    gradk = grad_l2(data, u, data0, x, rho, Nx, Ny, Nt, h, dt, source_position, receiver_position; pml_len=pml_len, pml_coef=pml_coef);

    gradk = reshape(gradk, Nx*Ny, 1)
    return fk, gradk
end

function obj_fn_parallel(c, rho, Nx, Ny, Nt, h, dt, source, source_position, receiver_position; pml_len=10, pml_coef=100)
    x = reshape(c, Nx, Ny)

    data, u = multi_solver_parallel(x, rho, Nx, Ny, Nt, h, dt, source, source_position, receiver_position; pml_len=pml_len, pml_coef=pml_coef)

    fk = 0.5 * norm(data - data0) ^ 2

    gradk = grad_l2_parallel(data, u, data0, x, rho, Nx, Ny, Nt, h, dt, source_position, receiver_position; pml_len=pml_len, pml_coef=pml_coef);

    gradk = reshape(gradk, Nx*Ny, 1)
    return fk, gradk
end
