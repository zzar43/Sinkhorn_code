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
#         println("    alpha: $alpha, fk1: $fk1, fk+c*alpha*gradk^2: ", (fk + c*alpha*sum(gradk.*pk)))
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