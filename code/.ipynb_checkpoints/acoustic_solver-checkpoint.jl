# differential operator
function dx_central(A, Nx, Ny, dx)
    B = zeros(Nx, Ny);
    B[2:end-1, 2:end-1] = (A[3:end,2:end-1] - A[1:end-2,2:end-1])./(2*dx);
    return B
end
function dy_central(A, Nx, Ny, dy)
    B = zeros(Nx, Ny);
    B[2:end-1, 2:end-1] = (A[2:end-1,3:end] - A[2:end-1,1:end-2])./(2*dy);
    return B
end
function dx_forward(A, Nx, Ny, dx)
    B = zeros(Nx, Ny);
    B[2:end-1, 2:end-1] = (A[2:end-1,2:end-1] - A[1:end-2,2:end-1])./(dx);
    return B
end
function dy_forward(A, Nx, Ny, dy)
    B = zeros(Nx, Ny);
    B[2:end-1, 2:end-1] = (A[2:end-1,2:end-1] - A[2:end-1,1:end-2])./(dy);
    return B
end
function laplace_central(A, Nx, Ny, dx, dy)
    B = zeros(Nx, Ny);
    B[2:end-1, 2:end-1] = (A[3:end,2:end-1] - 2*A[2:end-1,2:end-1] + A[1:end-2,2:end-1])./dx^2 + (A[2:end-1,3:end] - 2*A[2:end-1,2:end-1] + A[2:end-1,1:end-2])./dy^2;
    return B
end

# Ricker function for source
function source_ricker(center_fre, center_time, t)
    x = (1 .- 2*pi^2 .* center_fre^2 .* (t.-center_time).^2) .* exp.(-pi^2*center_fre^2 .* (t .- center_time).^2);
end

# extension of the domain
function extend_vel(vel, Nx_pml, Ny_pml, pml_len)
    vel_ex = zeros(Nx_pml, Ny_pml);
    vel_ex[pml_len+1:end-pml_len, pml_len+1:end-pml_len] .= vel;
    for i = 1:pml_len
        vel_ex[i,:] = vel_ex[pml_len+1,:];
        vel_ex[end-i+1,:] = vel_ex[end-pml_len,:];
        vel_ex[:,i] = vel_ex[:,pml_len+1];
        vel_ex[:,end-i+1] = vel_ex[:,end-pml_len];
    end
    return vel_ex
end

# core solver
function acoustic_eq_solver(c, rho, Nx, Ny, Nt, h, dt, source, source_position, receiver_position; pml_len=10, pml_coef=100)

    # prepare PML coef
    if ndims(source_position) == 1
        source_num = 1
    else
        source_num = size(source_position,1)
    end
    if ndims(receiver_position) == 1
        receiver_num = 1;
    else
        receiver_num = size(receiver_position,1)
    end
    pml_value = range(0, stop=pml_coef, length=pml_len);
    Nx_pml = Nx + 2*pml_len;
    Ny_pml = Ny + 2*pml_len;
    source_position_pml = source_position .+ pml_len;
    source_position_pml_vec = zeros(Int, source_num)
    if source_num == 1
        source_position_pml_vec[1] = source_position_pml[1] + (source_position_pml[2]-1)*Nx_pml
    else
        for i = 1:source_num
            source_position_pml_vec[i] = source_position_pml[i,1] + (source_position_pml[i,2]-1)*Nx_pml
        end
    end
    receiver_position_pml = receiver_position .+ pml_len;
    receiver_position_pml_vec = zeros(Int, receiver_num)
    if receiver_num == 1
        receiver_position_pml_vec[1] = receiver_position_pml[1] + (receiver_position_pml[2]-1)*Nx_pml
    else
        for i = 1:receiver_num
            receiver_position_pml_vec[i] = receiver_position_pml[i,1] + (receiver_position_pml[i,2]-1)*Nx_pml
        end
    end
    sigma_x = zeros(Nx_pml, Ny_pml);
    sigma_y = zeros(Nx_pml, Ny_pml);
    for i = 1:pml_len
        sigma_x[pml_len+1-i,:] .= pml_value[i];
        sigma_x[pml_len+Nx+i,:] .= pml_value[i];
        sigma_y[:,pml_len+1-i] .= pml_value[i];
        sigma_y[:,pml_len+Ny+i] .= pml_value[i];
    end

    # extend coef
    c_ex = extend_vel(c, Nx_pml, Ny_pml, pml_len)
    rho_ex = extend_vel(rho, Nx_pml, Ny_pml, pml_len)

    # Anonymous Functions
    dx(A) = dx_central(A, Nx_pml, Ny_pml, h);
    dy(A) = dy_central(A, Nx_pml, Ny_pml, h);

    # initialize
    a = 1 ./ (rho_ex .* c_ex.^2);
    b = 1 ./ rho_ex;
    alpha = 1 * (sigma_x + sigma_y);
    beta = 1 * sigma_x .* sigma_y;
    dx_b = dx(b);
    dy_b = dy(b);
    P = zeros(Nx, Ny, Nt);
    data = zeros(Nt, receiver_num);

    # main algorithm
    # n = 0, initial value
    p0 = zeros(Nx_pml, Ny_pml);
    p1 = zeros(Nx_pml, Ny_pml);
    p2 = zeros(Nx_pml, Ny_pml);
    ux0 = zeros(Nx_pml, Ny_pml);
    ux1 = zeros(Nx_pml, Ny_pml);
    ux2 = zeros(Nx_pml, Ny_pml);
    uy0 = zeros(Nx_pml, Ny_pml);
    uy1 = zeros(Nx_pml, Ny_pml);
    uy2 = zeros(Nx_pml, Ny_pml);
    
    # n = 1
    ux1 = (1 .- 0.5*dt*sigma_x).*ux0 - 0.5*dt*(sigma_x-sigma_y).*(dx(p1)+dx(p0));
    ux1 = ux1 ./ (1 .+ 0.5*dt*sigma_x);
    uy1 = (1 .- 0.5*dt*sigma_y).*uy0 - 0.5*dt*(sigma_y-sigma_x).*(dy(p1)+dy(p0));
    uy1 = uy1 ./ (1 .+ 0.5*dt*sigma_y);
    p1[source_position_pml_vec] .= dt^2*source[1,:]
    P[:,:,1] = p1[pml_len+1:end-pml_len, pml_len+1:end-pml_len];
    data[1,:] = p1[receiver_position_pml_vec]

    # n = 1, main loop
    for iter = 2:Nt
        dx_p1 = dx(p1)
        dy_p1 = dy(p1)
        space = zeros(Nx_pml, Ny_pml);
        space = space + dx_b.*ux1 + dy_b.*uy1
        space = space + b .* (dx(ux1) + dy(uy1))
        space = space + dx_b .* dx_p1 + dy_b .* dy_p1
        space = space + b .* laplace_central(p1, Nx_pml, Ny_pml, h, h)
        
        p2 = (2*a - dt^2*beta).*p1 + (-a+0.5*dt*a.*alpha).*p0 + dt^2*space
        # p2[source_position_pml_vec] .= dt^2*source[iter,:]
        p2 = p2 ./ (a+0.5*dt*a.*alpha)
        p2[source_position_pml_vec] .= source[iter,:]
        
        ux2 = (1 .- 0.5*dt*sigma_x).*ux1 - 0.5*dt*(sigma_x-sigma_y).*(dx(p2)+dx_p1);
        ux2 = ux2 ./ (1 .+ 0.5*dt*sigma_x);
        uy2 = (1 .- 0.5*dt*sigma_y).*uy1 - 0.5*dt*(sigma_y-sigma_x).*(dy(p2)+dy_p1);
        uy2 = uy2 ./ (1 .+ 0.5*dt*sigma_y);
        
        p0[:] = p1[:]; p1[:] = p2[:]
        ux1[:] = ux2[:]
        uy1[:] = uy2[:]
        
        P[:,:,iter] .= p2[pml_len+1:end-pml_len, pml_len+1:end-pml_len];
        data[iter,:] = p2[receiver_position_pml_vec];
    end

    return data, P
end

# solver for multiple sources
function multi_solver(c, rho, Nx, Ny, Nt, h, dt, source, source_position, receiver_position; pml_len=10, pml_coef=100)
    source_num = size(source_position,1)
    receiver_num = size(receiver_position,1)
    
    data = zeros(Nt, receiver_num, source_num)
    P = zeros(Nx, Ny, Nt, source_num)

    for ind = 1:source_num
        source1 = source[:,ind]
        source_position1 = zeros(Int, 1, 2)
        source_position1 = source_position[ind,:]
        d, p = acoustic_eq_solver(c, rho, Nx, Ny, Nt, h, dt, source1, source_position1, receiver_position; pml_len=pml_len, pml_coef=pml_coef)
        data[:,:,ind] = d;
        P[:,:,:,ind] = p;
    end
    return data, P
end

# solver for backward wavefield in adjoint method
function backward_solver(c, rho, Nx, Ny, Nt, h, dt, adj_source, source_position, receiver_position; pml_len=10, pml_coef=100)
    adj_source = adj_source[end:-1:1,:,:]
    source_num = size(adj_source, 3)
    # receiver_num = size(receiver_position,1)

    # data = zeros(Nt, receiver_num, source_num)
    P = zeros(Nx, Ny, Nt, source_num)

    for ind = 1:source_num
        source1 = adj_source[:,:,ind]
        d, p = acoustic_eq_solver(c, rho, Nx, Ny, Nt, h, dt, source1, receiver_position, receiver_position; pml_len=pml_len, pml_coef=pml_coef)
        P[:,:,:,ind] = p;
    end
    return P
end

# Parallel solver for wave equation

using Distributed, SharedArrays

function multi_solver_parallel(c, rho, Nx, Ny, Nt, h, dt, source, source_position, receiver_position; pml_len=10, pml_coef=100)
    source_num = size(source_position,1)
    receiver_num = size(receiver_position,1)
    data = SharedArray{Float64}(Nt, receiver_num, source_num)
    P = SharedArray{Float64}(Nx, Ny, Nt, source_num)

    @sync @distributed for ind = 1:source_num
        source1 = source[:,ind]
        source_position1 = zeros(Int, 1, 2)
        source_position1 = source_position[ind,:]
        d, p = acoustic_eq_solver(c, rho, Nx, Ny, Nt, h, dt, source1, source_position1, receiver_position; pml_len=pml_len, pml_coef=pml_coef)
        data[:,:,ind] = d;
        P[:,:,:,ind] = p;
    end
    
    data = Array(data)
    P = Array(P)
    
    return data, P
end

# solver for backward wavefield in adjoint method with parallel computing
function backward_solver_parallel(c, rho, Nx, Ny, Nt, h, dt, adj_source, source_position, receiver_position; pml_len=10, pml_coef=100)
    
    adj_source = adj_source[end:-1:1,:,:]
    source_num = size(adj_source, 3)
    
    P = SharedArray{Float64}(Nx, Ny, Nt, source_num)

    @sync @distributed for ind = 1:source_num
        source1 = adj_source[:,:,ind]
        d, p = acoustic_eq_solver(c, rho, Nx, Ny, Nt, h, dt, source1, receiver_position, receiver_position; pml_len=pml_len, pml_coef=pml_coef)
        P[:,:,:,ind] = p;
    end
    
    P = Array(P)
    
    return P
end