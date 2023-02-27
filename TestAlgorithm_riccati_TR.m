% %%%%%%%%%%%%%%%%%%%%%%% TestAlgorithm_riccati_TR.m %%%%%%%%%%%%%%%%%%%%%%
%
% Stephen Monnnet
% Laboratoire d'Automatique, 2023
%
% This code allows to test our trust-region Riccati recursion solver to any
% non-linear system. The user must provide the expression \Dot{x}= f(x, u)
% for each sub-system as well as the switching-times.
%
% Two tests are possible : 
%       - Comparison with casADi
%         -> Uncomment lines 178-197
%       - Measure evolution of the solving duration w.r.t. the horizon N
%         -> Uncomment lines 203-247

%% Initialisation
close all;
clear;
clc;

% DON'T FORGET TO CORRECT THIS PATH ACCORDING TO YOUR INSTALLATION
addpath('C:/casadi-matlabR2016a-v3.5.5');
import casadi.*;
file = 'figures2/';

%% Miscellaneous variables
N = 50; % Horizon
dt = 0.05; % Sampling time
t0 = 0; % Initial time
tf = t0 + N * dt; % Final time

DELTA_0 = 5; % Initial trust radius
maxIter = 100; % Max number of Newton steps
showIter = true; % If true, plot intermediate solution of x, u
solver = 'TR'; % 'casadi' to use casADi to solve trust region sub-problem

%% Declare System
% Dimensions : Change it according to your system
nx = 2; nu = 1;

% Symbolic variables
x = sym('x', [nx, 1], 'real');
u = sym('u', [nu, 1], 'real');
lambda = sym('lambda', [nx, 1], 'real');

% System dynamic : Change nx and nu according to your system
% f1 = [x(1) + u(1) * sin(x(1)); -x(2) - u(2) * cos(x(2)); x(2) * x(3)];
% f2 = [x(2) + u(2) * sin(x(2)); -x(1) - u(1) * cos(x(1)); x(1) * x(3)];
% f3 = [-x(1) - u(1) * sin(x(1)); -x(2) + u(2) * cos(x(2)); x(1) * x(2)]; % nx = 2, nu = 2
%f = x * u + u^2;
f = [x(1) + u * sin(x(1)) ; -x(2) - u * cos(x(2))];

% Switching Times -> for n systems, n-1 switching times s.t. 0 < t_i < N
%switchTime = [10, 22];
switchTime = [];

% Construct cell array of dynamical models
%f = {f1, f2, f3};
f = {f};
Nsys = length(f);

% Cost function
Q = eye(nx);
R = eye(nu);
QN = eye(nx);
l = 0.5 * x' * Q * x + 0.5 * u' * R * u;
Vf = 0.5 * x' * QN * x;

% Hamiltonian
h = cell(Nsys, 1);

%% %%%%%%%%%%%%%%%%%%%%%%%%% BUILD DERIVATIVES AND FUNCTION HANDLES %%%%%%%%%%%%%%%%%%%%%%%
dfdx = cell(N, 1);
dfdu = cell(N, 1);
dhdx = cell(N, 1);
dhdu = cell(N, 1);
dhdxx = cell(N, 1);
dhduu = cell(N, 1);
dhdxu = cell(N, 1);
dhdux = cell(N, 1);

dynamic.f = cell(N, 1);
dynamic.dfdx = cell(N, 1);
dynamic.dfdu = cell(N, 1);

dynamic.Hamilt.h = cell(N, 1);
dynamic.Hamilt.dhdx = cell(N, 1);
dynamic.Hamilt.dhdu = cell(N, 1);
dynamic.Hamilt.dhdxx = cell(N, 1);
dynamic.Hamilt.dhduu = cell(N, 1);
dynamic.Hamilt.dhdxu = cell(N, 1);
dynamic.Hamilt.dhdux = cell(N, 1);

switchTime = [0 switchTime N];

for n = 1:1:N
    for i = 1:1:Nsys
        if n >= switchTime(i) && n <= switchTime(i+1)
            dfdx{n} = jacobian(f{i}, x);
            dfdu{n} = jacobian(f{i}, u);
            h{n} = l + lambda' * f{i} * dt;
            
            dhdx{n} = jacobian(h{n}, x)';
            dhdu{n} = jacobian(h{n}, u)';
            dhdxx{n} = hessian(h{n}, x);
            dhduu{n} = hessian(h{n}, u);
            dhdxu{n} = jacobian(dhdx{n}, u);
            dhdux{n} = jacobian(dhdu{n}, x);
            
            dynamic.f{n} = matlabFunction(f{i}, 'vars', [{x}, {u}]);
            dynamic.dfdx{n} = matlabFunction(dfdx{n}, 'vars', [{x}, {u}]);
            dynamic.dfdu{n} = matlabFunction(dfdu{n}, 'vars', [{x}, {u}]);
            
            dynamic.Hamilt.h{n} = matlabFunction(h{n}, 'vars', [{x}, {u}, {lambda}]);
            dynamic.Hamilt.dhdx{n} = matlabFunction(dhdx{n}, 'vars', [{x}, {u}, {lambda}]);
            dynamic.Hamilt.dhdu{n} = matlabFunction(dhdu{n}, 'vars', [{x}, {u}, {lambda}]);
            dynamic.Hamilt.dhdxx{n} = matlabFunction(dhdxx{n}, 'vars', [{x}, {u}, {lambda}]);
            dynamic.Hamilt.dhduu{n} = matlabFunction(dhduu{n}, 'vars', [{x}, {u}, {lambda}]);
            dynamic.Hamilt.dhdxu{n} = matlabFunction(dhdxu{n}, 'vars', [{x}, {u}, {lambda}]);
            dynamic.Hamilt.dhdux{n} = matlabFunction(dhdux{n}, 'vars', [{x}, {u}, {lambda}]);
        end
    end
    
end

dldx = jacobian(l, x)';
dldu = jacobian(l, u)';
dldxx = jacobian(dldx, x);
dlduu = jacobian(dldu, u);
dldxu = jacobian(dldx, u);
dldux = jacobian(dldu, x);
dVfdx = jacobian(Vf, x)';
dVfdxx = jacobian(dVfdx, x);

cost.l = matlabFunction(l, 'vars', [{x}, {u}]);
cost.dldx = matlabFunction(dldx, 'vars', [{x}, {u}]);
cost.dldu = matlabFunction(dldu, 'vars', [{x}, {u}]);
cost.dldxx = matlabFunction(dldxx, 'vars', [{x}, {u}]);
cost.dlduu = matlabFunction(dlduu, 'vars', [{x}, {u}]);
cost.dldux = matlabFunction(dldux, 'vars', [{x}, {u}]);
cost.dldxu = matlabFunction(dldxu, 'vars', [{x}, {u}]);
cost.Vf = matlabFunction(Vf, 'vars', {x});
cost.dVfdx = matlabFunction(dVfdx, 'vars', {x});
cost.dVfdxx = matlabFunction(dVfdxx, 'vars', {x});
cost.COST = @(X, U) totalCost(cost, X, U);

%% Initialize primal and dual
primal.time = [0:dt:N*dt];
xbar = 5 * rand([nx, 1]) - 2.5;
primal.xbar = xbar;
primal.x = zeros(nx, N+1);
primal.u = 10 * (rand([nu, N])-0.5); % Initial guess between -1 and +1
dual.lambda = 2e-3*(rand(nx, N+1)); % Initial guess between 2e-3 and 0

primal.dx = zeros(nx, N+1);
primal.du = zeros(nu, N);
dual.dlambda = zeros(nx, N+1);

%% Initialize Parameters
params.DELTA_0 = DELTA_0;
params.DELTA_MAX = DELTA_0 * 2;
params.dt = dt;
params.tol = N * 0.1e-3;
params.maxIter = maxIter;
params.showIter = showIter;
params.solver = solver; % "TR" : Trust region / "casadi"
params.file = file;

%% Forward Propagation to Initalize x as feasible
primal.x(:, 1) = xbar;
for i=1:1:N
    xi = primal.x(:, i);
    ui = primal.u(:, i);
    
    primal.x(:, i+1) = xi + dynamic.f{i}(xi, ui) * dt;
end

%% Plot initial state and inputs trajectory
figure;
legend_str = [];

% Plot the states trajectory
for k = 1:1:nx
    plot(primal.x(k, :));
    legend_str = [legend_str strcat("x_", num2str(k))];
    hold on;
end

% Plot the inputs trajectory
for k = 1:1:nu
    plot(primal.u(k, :));
    legend_str = [legend_str strcat("u_", num2str(k))];
    hold on;
end

title(strcat("Initialization"));
legend(legend_str);
xlabel("Time-step [s]");
ylabel("States / Inputs [-]");
grid on;
hold off;

saveas(gcf, strcat(file, "initialisation.png"));

%% OPTIMIZATION : UNCOMMENT THE TEST THAT YOU WANT TO RUN

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%%%%%%%%%%%%%% COMPARISON WITH CASADI %%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
OCP_TR = NOCP(primal, dual, dynamic, cost, params);
tic
[x, u] = OCP_TR.solve();
toc

params.solver = 'casadi';
OCP_casadi = NOCP(primal, dual, dynamic, cost, params);
tic
[x_, u_] = OCP_casadi.solve();
toc

figure;
plot([0:1:length(OCP_TR.costCurve)-1], OCP_TR.costCurve, 'Linewidth', 2);
hold on;
plot([0:1:length(OCP_casadi.costCurve)-1], OCP_casadi.costCurve, '--', 'Linewidth', 2);
grid on;
xlabel("Iteration [-]");
ylabel("Cost [-]");
title("Objective Function");
legend("TR", "CasADi");
saveas(gcf, strcat(file, "_tr_VS_casadi_obj", ".png"));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%% EVOLUTION OF THE SOLVING DURATION %%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% t = 0;
% tvec = [];
% Nvec = [2:2:N]; % CHOOSE THE HORIZON VALUES TO TEST
% primal.xbar = 5 * rand([nx, 1]) - 2.5;
% params.showIter = false;
% 
% for N = Nvec
%     params.tol = N * 0.1e-3;
%     N
% 
%     % Variables
%     primal.time = [0:dt:N*dt];
%     primal.x = zeros(nx, N+1);
%     primal.u = 2 * (rand([nu, N])-0.5); % Initial guess between -1 and +1
%     dual.lambda = 2e-3*(rand(nx, N+1)-0.5); % Initial guess between 1e-3 and -2e-3
% 
%     primal.dx = zeros(nx, N+1);
%     primal.du = zeros(nu, N);
%     dual.dlambda = zeros(nx, N+1);
%     cost.COST = @(X, U) 0.5 * trace(X(1:N, :) * Q * X(1:N, :)') + 0.5 * trace(U * R * U') + cost.Vf(X(N+1, :)');
%     
%     primal.x(:, 1) = xbar;
%     for i=1:1:N
%         xi = primal.x(:, i);
%         ui = primal.u(:, i);
%     
%         primal.x(:, i+1) = xi + dynamic.f{i}(xi, ui) * dt;
%     end
%     
%     t = 0;
%     for i = 1:1:10
%         OCP_TR = NOCP(primal, dual, dynamic, cost, params);
%         tic;
%         [x, u] = OCP_TR.solve();
%         t = t + toc; 
%     end
%     tvec = [tvec t/10];
% end
% 
% figure;
% plot(Nvec, tvec);
% title("Solving Time : TR & Riccati Recursion");
% xlabel("Horizon [-]");
% ylabel("Duration [s]");
% grid on;

%% Function to compute the total cost (Used to construct a function handle)
function [COST] = totalCost(cost,X, U)
    COST = 0;
    for i = 1:1:length(U)
        COST = COST + cost.l(X(i, :)', U(i, :)');
    end
    COST = COST + cost.Vf(X(end, :)');
end