function adj_source_sinkhorn(data1, data2; lambda=1000, numItermax=1000, stopThr = 1e-7, verbose=false);
    Nt = size(data1,1)
    adj = 0 .* data1;
    M = cost_func_1d(Nt);

    if length(size(data1)) == 2
        for i = 1:size(data1,2)
            f = data1[:,i]
            g = data2[:,i]
            T, adj[:,i], d = sinkhorn_signal_1d(f, g, M; lambda=lambda, numItermax=numItermax, stopThr=stopThr, verbose=verbose);
        end
    elseif length(size(data1)) == 3
        for i = 1:size(data1,2)
            for j = 1:size(data1,3)
                f = data1[:,i,j]
                g = data2[:,i,j]
                T, adj[:,i,j], d = sinkhorn_signal_1d(f, g, M; lambda=lambda, numItermax=numItermax, stopThr=stopThr, verbose=verbose);
            end
        end
    else
        error("Please check the dimension of data1")
    end

    return adj
end

function grad_l2(data, u, data0, c, rho, Nx, Ny, Nt, h, dt, source_position, receiver_position; pml_len=10, pml_coef=100);
#     input:
#     data1: received data
#     c1, rho1: 
    
    adj_source = data - data0
    
#     adjoint wavefield
    vl = backward_solver(c, rho, Nx, Ny, Nt, h, dt, adj_source, source_position, receiver_position; pml_len=pml_len, pml_coef=pml_coef);
    
    uu = 0 .* u;
    uu[:,:,2:end-1,:] = (u[:,:,3:end,:] - 2*u[:,:,2:end-1,:] + u[:,:,1:end-2,:]) / (dt^2);
    gradl = uu[:,:,end:-1:1,:].*vl
    gradl = sum(gradl, dims=[3,4])
    gradl = gradl[:,:,1,1]
    gradl = gradl ./ (maximum(abs.(gradl)))
    
    return gradl
end

function grad_sinkhorn(data, u, data0, c, rho, Nx, Ny, Nt, h, dt, source_position, receiver_position; pml_len=10, pml_coef=100, lambda=1000, numItermax=10, stopThr = 1e-6)
    
    adj_source = adj_source_sinkhorn(data, data0; lambda=lambda, numItermax=numItermax, stopThr=stopThr, verbose=false);
    
#     adjoint wavefield
    v = backward_solver(c, rho, Nx, Ny, Nt, h, dt, adj_source, source_position, receiver_position; pml_len=pml_len, pml_coef=pml_coef);
    
    uu = 0 .* u;
    uu[:,:,2:end-1,:] = (u[:,:,3:end,:] - 2*u[:,:,2:end-1,:] + u[:,:,1:end-2,:]) / (dt^2);
    grad = uu[:,:,end:-1:1,:].*v
    grad = sum(grad, dims=[3,4])
    grad = grad[:,:,1,1] ./ maximum(abs.(grad))
    
    return -grad
end

function grad_l2_parallel(data, u, data0, c, rho, Nx, Ny, Nt, h, dt, source_position, receiver_position; pml_len=10, pml_coef=100);
#     input:
#     data1: received data
#     c1, rho1: 
    
    adj_source = data - data0
    
#     adjoint wavefield
    vl = backward_solver_parallel(c, rho, Nx, Ny, Nt, h, dt, adj_source, source_position, receiver_position; pml_len=pml_len, pml_coef=pml_coef);
    
    uu = 0 .* u;
    uu[:,:,2:end-1,:] = (u[:,:,3:end,:] - 2*u[:,:,2:end-1,:] + u[:,:,1:end-2,:]) / (dt^2);
    gradl = uu[:,:,end:-1:1,:].*vl
    gradl = sum(gradl, dims=[3,4])
    gradl = gradl[:,:,1,1]
    gradl = gradl ./ (maximum(abs.(gradl)))
    
    return gradl
end