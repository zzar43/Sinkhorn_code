using Distributed
using SharedArrays

function adj_source_sinkhorn(data1, data2, M; reg=1e-3, reg_m=1e2, iterMax=100, verbose=false);
    adj = 0 .* data1;
    dist = 0

    if length(size(data1)) == 2
        for i = 1:size(data1,2)
            f = data1[:,i]
            g = data2[:,i]
            T, aaa, d1 = unbalanced_sinkhorn_1d_signal(f, g, M, reg, reg_m; iterMax=iterMax, verbose=verbose)
            adj[:,i] = aaa
            dist += d1
        end
    elseif length(size(data1)) == 3
        for i = 1:size(data1,2)
            for j = 1:size(data1,3)
                f = data1[:,i,j]
                g = data2[:,i,j]
                T, aaa, d1 = unbalanced_sinkhorn_1d_signal(f, g, M, reg, reg_m; iterMax=iterMax, verbose=verbose)
                adj[:,i,j] = aaa
                dist += d1
            end
        end
    else
        error("Please check the dimension of data1")
    end

    return adj
end
        
function adj_source_sinkhorn_parallel_linear(data1, data2, M; reg_p=0, reg=1e-3, reg_m=1e2, iterMax=50, verbose=false);

    Nt = size(data1,1)
    adj = 0 .* data1;
    adj = SharedArray{Float64}(adj);
    dist = zeros(size(data1,2), size(data1,3))
    dist = SharedArray{Float64}(dist);
    

    if length(size(data1)) == 2
        @sync @distributed for i = 1:size(data1,2)
            f = data1[:,i]
            g = data2[:,i]
            T, aaa, d1 = unbalanced_sinkhorn_1d_signal_linear(f, g, M, reg, reg_m; reg_p=reg_p, iterMax=iterMax, verbose=verbose)
            adj[:,i] = aaa
            dist[i] = d1
        end
    elseif length(size(data1)) == 3
        @sync @distributed for i = 1:size(data1,2)
            for j = 1:size(data1,3)
                f = data1[:,i,j]
                g = data2[:,i,j]
                T, aaa, d1 = unbalanced_sinkhorn_1d_signal_linear(f, g, M, reg, reg_m; reg_p=reg_p, iterMax=iterMax, verbose=verbose)
                adj[:,i,j] = aaa
                dist[i,j] = d1
            end
        end
    else
        error("Please check the dimension of data1")
    end
    adj = Array(adj)
    dist = sum(dist)

    return adj, dist
end

function adj_source_sinkhorn_parallel_exp(data1, data2, M;c=1, reg=1e-3, reg_m=1e2, iterMax=50, verbose=false);

    Nt = size(data1,1)
    adj = 0 .* data1;
    adj = SharedArray{Float64}(adj);
    dist = zeros(size(data1,2), size(data1,3))
    dist = SharedArray{Float64}(dist);
    
    if length(size(data1)) == 2
        @sync @distributed for i = 1:size(data1,2)
            f = data1[:,i]
            g = data2[:,i]
            T, aaa, d1 = unbalanced_sinkhorn_1d_signal_exp(f, g, M, reg, reg_m; c=c, iterMax=iterMax, verbose=verbose)
            adj[:,i] = aaa
            dist[i] = d1
        end
    elseif length(size(data1)) == 3
        @sync @distributed for i = 1:size(data1,2)
            for j = 1:size(data1,3)
                f = data1[:,i,j]
                g = data2[:,i,j]
                T, aaa, d1 = unbalanced_sinkhorn_1d_signal_exp(f, g, M, reg, reg_m; c=c, iterMax=iterMax, verbose=verbose)
                adj[:,i,j] = aaa
                dist[i,j] = d1
            end
        end
    else
        error("Please check the dimension of data1")
    end
    adj = Array(adj)
    dist = sum(dist)

    return adj, dist
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
#     gradl = gradl ./ (maximum(abs.(gradl)))
    
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
    grad = grad[:,:,1,1]
    
    return grad
end

function grad_l2_parallel(data, u, data0, c, rho, Nx, Ny, Nt, h, dt, source_position, receiver_position; save_ratio=1, pml_len=10, pml_coef=100);
    c = reshape(c, Nx, Ny)
    
    adj_source = data - data0
    
#     adjoint wavefield
    vl = backward_solver_parallel(c, rho, Nx, Ny, Nt, h, dt, -adj_source, source_position, receiver_position; save_ratio=save_ratio, pml_len=pml_len, pml_coef=pml_coef);
    
    uu = 0 .* u;
    for i = 1:size(uu,4)
        for j = 2:size(uu,3)-1
            uu[:,:,j,i] = (u[:,:,j-1,i] - 2*u[:,:,j,i] + u[:,:,j+1,i]) / (dt^2);
        end
    end
    gradl = zeros(Nx,Ny)
    for i = 1:size(u,3)
        for j = 1:size(u,4)
            gradl += uu[:,:,i,j] .* vl[:,:,end-i+1,j] * dt * save_ratio
        end
    end
    gradl = gradl .* (-2)./(rho.*c.^3)
    gradl = gradl ./ maximum(abs.(gradl))
    
    return gradl
end
        
function grad_sinkhorn_parallel_linear(data, u, data0, c, rho, Nx, Ny, Nt, h, dt, source_position, receiver_position; reg_p=0, pml_len=10, pml_coef=100, reg=5e-3, reg_m=1e2, iterMax=50, verbose=false, save_ratio=1)
    c = reshape(c, Nx, Ny)
    t = range(0,step=dt,length=Nt)
    M = cost_matrix_1d(t,t)
    
    adj_source, fk = adj_source_sinkhorn_parallel_linear(data, data0, M; reg_p=reg_p, reg=reg, reg_m=reg_m, iterMax=iterMax, verbose=verbose);

#     adjoint wavefield
    v = backward_solver_parallel(c, rho, Nx, Ny, Nt, h, dt, adj_source, source_position, receiver_position; pml_len=pml_len, pml_coef=pml_coef, save_ratio=save_ratio);
    
    uu = 0 .* u;
    uu[:,:,2:end-1,:] = (u[:,:,3:end,:] - 2*u[:,:,2:end-1,:] + u[:,:,1:end-2,:]) / (dt^2);
#     grad = uu[:,:,end:-1:1,:].*v
#     grad = sum(grad, dims=[3,4])
#     grad = grad[:,:,1,1]
    grad = zeros(Nx,Ny)
    for i = 1:size(u,3)
        for j = 1:size(u,4)
            grad += uu[:,:,i,j] .* vl[:,:,end-i+1,j] * dt * save_ratio
        end
    end
    gradl = grad .* (-2)./(rho.*c.^3)
    
    return grad, fk
end

function grad_sinkhorn_parallel_exp(data, u, data0, c, rho, Nx, Ny, Nt, h, dt, source_position, receiver_position; reg_p=0, pml_len=10, pml_coef=100, reg=5e-3, reg_m=1e2, iterMax=50, verbose=false, save_ratio=1)
    c = reshape(c, Nx, Ny)
    t = range(0,step=dt,length=Nt)
    M = cost_matrix_1d(t,t)
    
    adj_source, fk = adj_source_sinkhorn_parallel_exp(data, data0, M; reg_p=reg_p, reg=reg, reg_m=reg_m, iterMax=iterMax, verbose=verbose);

#     adjoint wavefield
    v = backward_solver_parallel(c, rho, Nx, Ny, Nt, h, dt, adj_source, source_position, receiver_position; pml_len=pml_len, pml_coef=pml_coef, save_ratio=save_ratio);
    
    uu = 0 .* u;
    uu[:,:,2:end-1,:] = (u[:,:,3:end,:] - 2*u[:,:,2:end-1,:] + u[:,:,1:end-2,:]) / (dt^2);
#     grad = uu[:,:,end:-1:1,:].*v
#     grad = sum(grad, dims=[3,4])
#     grad = grad[:,:,1,1]
    grad = zeros(Nx,Ny)
    for i = 1:size(u,3)
        for j = 1:size(u,4)
            grad += uu[:,:,i,j] .* vl[:,:,end-i+1,j] * dt * save_ratio
        end
    end
    gradl = grad .* (-2)./(rho.*c.^3)
    
    return grad, fk
end

# function grad_sinkhorn_parallel_bal(data, u, data0, c, rho, Nx, Ny, Nt, h, dt, source_position, receiver_position; reg_p=0, pml_len=10, pml_coef=100, reg=5e-3, reg_m=1e2, iterMax=50, verbose=false)
#     c = reshape(c, Nx, Ny)
#     Nt = size(data,1)
#     t = range(0,step=dt,length=Nt)
#     M = cost_matrix_1d(t,t)
    
#     adj_source, fk = adj_source_sinkhorn_parallel_bal(data, data0, M; reg_p=reg_p, reg=reg, iterMax=iterMax, verbose=verbose);
        
# #     adjoint wavefield
#     v = backward_solver_parallel(c, rho, Nx, Ny, Nt, h, dt, adj_source, source_position, receiver_position; pml_len=pml_len, pml_coef=pml_coef);
    
#     uu = 0 .* u;
#     uu[:,:,2:end-1,:] = (u[:,:,3:end,:] - 2*u[:,:,2:end-1,:] + u[:,:,1:end-2,:]) / (dt^2);
#     grad = uu[:,:,end:-1:1,:].*v
#     grad = sum(grad, dims=[3,4])
#     grad = grad[:,:,1,1]
    
#     return grad, fk
# end