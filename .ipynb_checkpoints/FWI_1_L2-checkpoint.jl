using Distributed
using SharedArrays
addprocs(11)

@everywhere include("inverse_prob.jl")
@everywhere include("code/barycenter.jl")

@eval @everywhere begin
    Nx = 101;
    Ny = 101;
    h = 0.01;
    x = range(0,step=h,length=Nx)
    y = range(0,step=h,length=Ny)

    Fs = 400;
    dt = 1/Fs
    Nt = 1001;
    t = range(0, length=Nt, step=dt)

    source = source_ricker(5, 0.2, t)
    source_position = zeros(11,2)
    for i = 1:11
        source_position[i,:] = [5 10*(i-1)+1]
    #     source_position[i,:] = [5 51]
    end
    source = repeat(source, 1, 11)

    receiver_position = zeros(51,2)
    for i = 1:51
        receiver_position[i,:] = [1, (i-1)*2+1]
    end

    c = ones(Nx, Ny)
    rho = ones(Nx, Ny)

    c0 = ones(Nx, Ny)
    rho0 = ones(Nx, Ny)
    # c0[50:70,60:80] .= 1.05
    for i = 1:Nx
        for j = 1:Ny
            if sqrt((x[i]-0.5).^2 + (y[j]-0.5).^2) < 0.2
                c0[i,j] = 1.05
            end
        end
    end

    c = imfilter(c0, Kernel.gaussian(20));

    cc = ones(Nx, Ny)
    rhoc = ones(Nx, Ny)

    pml_len = 20
    pml_coef = 50
end


# Forward modelling
println("============================================")
@time data, u = multi_solver_parallel(c, rho, Nx, Ny, Nt, h, dt, source, source_position, receiver_position; pml_len=10, pml_coef=100);
@time data0, u0 = multi_solver_parallel(c0, rho0, Nx, Ny, Nt, h, dt, source, source_position, receiver_position; pml_len=10, pml_coef=100);
u0=[]
println("Forward modelling complete.")
println("============================================")


# optimization function
# L2
opt_fn(x) = obj_fn_parallel(data0, x, rho, Nx, Ny, Nt, h, dt, source, source_position, receiver_position; pml_len=10, pml_coef=100)

x0 = c[:]
xk, fn_value = gradient_descent(opt_fn, x0, 0.05, 20, 1, 1.05; rho=0.5, c=1e-5, maxSearchTime=6, threshold=1e-5)
matwrite("FWI_1_l2_gd.mat", Dict("xx" => xk,"ff" => fn_value,"c"=>c,"c0"=>c0); compress = false)

xk, fn_value = nonlinear_cg(opt_fn, x0, 0.05, 20, 1, 1.05; rho=0.5, c=1e-5, maxSearchTime=6, threshold=1e-5)
matwrite("FWI_1_l2_cg.mat", Dict("xx" => xk,"ff" => fn_value,"c"=>c,"c0"=>c0); compress = false)
