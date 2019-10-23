using Distributed
using SharedArrays
addprocs(2)

@everywhere include("inverse_prob.jl")
@everywhere include("code/barycenter.jl")

@eval @everywhere begin
    Nx = 101;
    Ny = 101;
    h = 0.01;
    x = range(0,step=h,length=Nx)
    y = range(0,step=h,length=Ny)

    Fs = 500;
    dt = 1/Fs
    Nt = 1501;
    t = range(0, length=Nt, step=dt)

    source = source_ricker(5, 0.2, t)
    source_position = zeros(11,2)
    for i = 1:11
        source_position[i,:] = [5 10*(i-1)+1]
    end
    source = repeat(source, 1, 11)

    receiver_position = zeros(51,2)
    for i = 1:51
        receiver_position[i,:] = [101, (i-1)*2+1]
    end

    c = ones(Nx, Ny)
    rho = ones(Nx, Ny)


    c0 = ones(Nx, Ny)
    rho0 = ones(Nx, Ny)
    for i = 1:Nx
        for j = 1:Ny
            if sqrt((x[i]-0.5).^2 + (y[j]-0.5).^2) < 0.2
                c0[i,j] = 1.05
            end
        end
    end
    # c = imfilter(c, Kernel.gaussian(10));
    # cc = ones(Nx, Ny)
    # rhoc = ones(Nx, Ny)

    pml_len = 20
    pml_coef = 50
end


# Forward modelling
@time data, u = multi_solver_parallel(c, rho, Nx, Ny, Nt, h, dt, source, source_position, receiver_position; pml_len=10, pml_coef=100);
@time data0, u0 = multi_solver_parallel(c0, rho0, Nx, Ny, Nt, h, dt, source, source_position, receiver_position; pml_len=10, pml_coef=100);

# optimization function
# opt_fn(x) = obj_fn_parallel(data0, x, rho, Nx, Ny, Nt, h, dt, source, source_position, receiver_position; pml_len=10, pml_coef=100)
opt_fn(x) = obj_fn_sinkhorn_parallel(data0, x, rho, Nx, Ny, Nt, h, dt, source, source_position, receiver_position; pml_len=10, pml_coef=100, reg=5e-3, reg_m=1e2, reg_p=0, iterMax=100, verbose=false)

x0 = c[:]
xk, fn_value = gradient_descent(opt_fn, x0, 0.5, 1, 1, 1.05; rho=0.5, c=1e-4, maxSearchTime=1, threshold=1e-5)

matwrite("FWI_1.mat", Dict("xx" => xk,"ff" => fn_value,"c"=>c,"c0"=>c0); compress = false)