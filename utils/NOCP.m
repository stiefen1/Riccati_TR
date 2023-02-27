classdef NOCP < handle
    properties
        % primal & dual variables
        primal, dual
        % dynamic, Hamiltonian & cost
        dynamic, Hamilt, cost
        % solver
        riccati_solver
        % Miscellaneous
        dt, N, nx, nu, DELTA, DELTA_MAX, maxIter, tol, costCurve, stopCriterion
        Q_xx, Q_uu, Q_xu, r_x, A
        % Figures
        fig_xu, fig_cost, fig_xu_axis, fig_obj_axis, showIter
        % Lagrangian
        Lagrangian, LagrangianCurve
        % solver = choice of the solver, casadi_opti = optimizer
        solver, casadi_opti
        % Matrices
        H, g, c, M, h
        % File
        file
    end

    methods
        
        %
        % %%%%%%%%%%%%%%%%%%%%%% CONSTRUCTOR %%%%%%%%%%%%%%%%%%%%%%%
        %
        
        function obj = NOCP(primal, dual, dynamic, cost, params)
            obj.primal = primal;
            obj.dual = dual;
            obj.dynamic = dynamic;
            obj.cost = cost;
            obj.dt = params.dt;
            obj.DELTA = params.DELTA_0;
            obj.DELTA_MAX = params.DELTA_MAX;
            obj.showIter = params.showIter;
            obj.maxIter = params.maxIter;
            obj.tol = params.tol;
            obj.solver = params.solver;
            obj.N = size(primal.x, 2) - 1;
            obj.nx = size(primal.x, 1);
            obj.nu = size(primal.u, 1);
            obj.file = params.file;
            
            obj.Hamilt = dynamic.Hamilt;
        end
        
        %
        % %%%%%%%%%%%%%%% SETUP MATRICES & TR-RICCATI SOLVER %%%%%%%%%%%%%%%%
        %
        
        function [] = build_SQP(obj)
            % Initialization of the cells
            obj.Q_xx = cell(obj.N+1, 1);
            obj.Q_xu = cell(obj.N, 1);
            Q_ux = cell(obj.N, 1);
            obj.Q_uu = cell(obj.N, 1);

            obj.r_x = cell(obj.N+1, 1);
            r_u = cell(obj.N, 1);

            x_bar = cell(obj.N, 1);

            obj.A = cell(obj.N, 1);
            B = cell(obj.N, 1);
            
            %
            % %%%%%%%%%% CONSTRUCT MATRICES QXX, QUU, QXU, QUX %%%%%%%%5
            %
            
            % iteration 0, ..., N-1
            for i=1:1:obj.N
                % Get state, input, dual variables at current stage i & i+1
                xi = obj.primal.x(:, i);
                xip1 = obj.primal.x(:, i+1);
                ui = obj.primal.u(:, i);
                li = obj.dual.lambda(:, i); % lambda_i+1
                lip1 = obj.dual.lambda(:, i+1); % lambda_i+1

                % Compute matrices Q_xx, Q_xu, Q_ux, Q_uu, ... at stage i
                obj.Q_xx{i} = obj.Hamilt.dhdxx{i}(xi, ui, lip1);
                obj.Q_xu{i} = obj.Hamilt.dhdxu{i}(xi, ui, lip1);
                obj.Q_uu{i} = obj.Hamilt.dhduu{i}(xi, ui, lip1);
                Q_ux{i} = obj.Hamilt.dhdux{i}(xi, ui, lip1);
                
                obj.r_x{i} = obj.Hamilt.dhdx{i}(xi, ui, lip1) + lip1 - li;
                r_u{i} = obj.Hamilt.dhdu{i}(xi, ui, lip1);

                obj.A{i} = eye(obj.nx) + obj.dynamic.dfdx{i}(xi, ui) * obj.dt;
                B{i} = obj.dynamic.dfdu{i}(xi, ui) * obj.dt;

                x_bar{i} = xi + obj.dynamic.f{i}(xi, ui) * obj.dt - xip1;
            end
            % Iteration N
            xN = obj.primal.x(:, i+1);
            lN = obj.dual.lambda(:, i+1);
            obj.Q_xx{i+1} = obj.cost.dVfdxx(xN);
            obj.r_x{i+1} = obj.cost.dVfdx(xN) - lN;
            
            
            %
            % %%%%%%%%% Construct matrix \Delta x = M * \Delta u + h %%%%%%
            %
            
            obj.M = zeros((obj.N+1)*obj.nx, obj.N*obj.nu);
            obj.h = zeros((obj.N+1) * obj.nx, 1);

            % Go through the matrix M and fill it with appropriate matrices
            for x = 1:1:obj.N
                for y = x+1:1:obj.N+1
                    if(y == x+1)
                        obj.M((y-1)*obj.nx+1:y*obj.nx, (x-1)*obj.nu + 1:x*obj.nu) = B{x};
                    else
                    obj.M((y-1)*obj.nx+1:y*obj.nx, (x-1)*obj.nu + 1:x*obj.nu) = obj.A{y-1} * obj.M((y-2)*obj.nx+1: (y-1)*obj.nx, (x-1)*obj.nu+1:x*obj.nu);
                    end
                end
            end
            
            % Go through vector h and fill it with appropriate values
            obj.h(1:obj.nx) = obj.primal.xbar - obj.primal.x(:, 1);
            for y = 1:1:obj.N
                obj.h(y*obj.nx+1:y*obj.nx+obj.nx) = x_bar{y} + obj.A{y} * obj.h((y-1)*obj.nx+1:(y-1)*obj.nx+obj.nx);
            end
            
            %
            % %%%%%%%%%%%% Setup matrices QXX, QUX, QXU, QUU %%%%%%%%%%%%%%%%
            %
            
            QXX = zeros(obj.nx*(obj.N+1));
            QUX = zeros(obj.nu*obj.N, obj.nx*obj.N);
            QXU = zeros(obj.nx*obj.N, obj.nu*obj.N);
            QUU = zeros(obj.nu*obj.N, obj.nu*obj.N);
            RX = zeros(obj.nx*(obj.N+1), 1);
            RU = zeros(obj.nu*obj.N, 1);

            for i = 0 : 1 : obj.N
                QXX(i*obj.nx + 1 : (i+1)*obj.nx, i*obj.nx + 1 : (i+1)*obj.nx) = obj.Q_xx{i+1};
                RX(i*obj.nx+1:i*obj.nx+obj.nx) = obj.r_x{i+1};
    
                if i < obj.N
                    QUX(i*obj.nu + 1 : (i+1)*obj.nu, i*obj.nx + 1 : (i+1)*obj.nx) = Q_ux{i+1};
                    QXU(i*obj.nx + 1 : (i+1)*obj.nx, i*obj.nu + 1 : (i+1)*obj.nu) = obj.Q_xu{i+1};
                    QUU(i*obj.nu + 1 : (i+1)*obj.nu, i*obj.nu + 1 : (i+1)*obj.nu) = obj.Q_uu{i+1};
                    RU(i*obj.nu + 1 : (i+1)*obj.nu) = r_u{i+1};
                end
            end
            
            %
            % %%%%%%%%%%%%%%%%% Create matrix H %%%%%%%%%%%%%%%%%%
            % 
            
            obj.H = obj.M' * QXX * obj.M + QUX * obj.M(1:end-obj.nx, :) + obj.M(1:end-obj.nx, :)' * QXU + QUU;            
            
            switch obj.solver
                case "TR"
                    
                    %
                    % %%%%%%%%%%%%%%%%%%% Check eigenvalues of H %%%%%%%%%%%%%%%%%%%
                    %
            
                    E = eig(obj.H);
                    LAMBDA1 = min(E);
                    % Initialize LAMBDA as bigger than the smallest
                    % eigevalue of H
                    LAMBDA = (-LAMBDA1 + 8e-1 * abs(LAMBDA1)) * eye(obj.nu);
            
                    % Create an object that handles Riccati recursion with
                    % trust region
                    obj.riccati_solver = riccati_TR(obj.N, LAMBDA, obj.DELTA, obj.A, B, obj.Q_xx, obj.Q_uu, Q_ux, obj.r_x, r_u, x_bar);
                case "casadi"
                    % If casasadi is used, g and c must be provided since
                    % the objective depends on it
                    obj.g = (obj.h' * QXX * obj.M + obj.h' * QXX' * obj.M + obj.h(1:end-obj.nx, :)' * QUX' + obj.h(1:end-obj.nx, :)' * QXU + RX' * obj.M + RU')';
                    obj.c = obj.h' * QXX * obj.h + RX' * obj.h;
                otherwise
                    error("Invalid Solver name != ['TR', 'casadi']");
            end
        end
        
        %
        % %%%%%%%%%%%%%%%%%%% UPDATE PRIMAL AND DUAL %%%%%%%%%%%%%%%%%%%%%
        %
        
        function [x, u, lambda] = updateSol(obj)
            % du and dx are directly updated 
            obj.primal.u = obj.primal.u + obj.primal.du;
            obj.primal.x = obj.primal.x + obj.primal.dx;
            % Update dlambda
            obj.update_dlambda();
            obj.dual.lambda = obj.dual.lambda + obj.dual.dlambda;
            x = obj.primal.x;
            u = obj.primal.u;
            lambda = obj.dual.lambda;
        end
        
        %
        % %%%%%%%%%%%%%%%%%%%%%%% SOLVE THE NOCP %%%%%%%%%%%%%%%%%%%%%%%%%
        %
        
        function [x, u] = solve(obj)
            if obj.showIter
                obj.fig_xu = figure;
            end
            
            obj.costCurve = [];
            obj.LagrangianCurve = [];
            
            % Initialize with first cost at initialization
            obj.costCurve = obj.cost.COST(obj.primal.x', obj.primal.u');
            obj.evalLagrangian(); % update lagrangian value with initial guess
            obj.LagrangianCurve = obj.Lagrangian; 
            obj.stopCriterion = false;

            i = 1;
            while(i <= obj.maxIter && ~obj.stopCriterion)
                obj.build_SQP();
                switch obj.solver
                    case "TR"
                        [obj.primal.dx, obj.primal.du] = obj.riccati_solver.solve();                        
                    case "casadi"
                        obj.casadi_opti = casadi.Opti();
                        p = struct("expand",true, "verbose", 0);
                        s = struct("max_iter",obj.maxIter);
                        obj.casadi_opti.solver('ipopt', p, s);
                        obj.primal.du_casadi = obj.casadi_opti.variable(obj.nu * obj.N, 1);
                        objective = obj.primal.du_casadi' * obj.H * obj.primal.du_casadi + obj.g' * obj.primal.du_casadi + obj.c;
                        obj.casadi_opti.subject_to( obj.primal.du_casadi' * obj.primal.du_casadi <= obj.DELTA^2 );
                        obj.casadi_opti.minimize(objective);
                        sol = obj.casadi_opti.solve();
                        
                        % Get du from casADi
                        du = sol.value(obj.primal.du_casadi);
                        % Get dx
                        dx = obj.M * du + obj.h; 
                        % Reshape dx to be consistent with du
                        for k = 0:1:obj.N-1
                            obj.primal.dx(:, k+1) = dx(k*obj.nx+1:(k+1)*obj.nx);
                            obj.primal.du(:, k+1) = du(k*obj.nu+1:(k+1)*obj.nu);
                        end
                        obj.primal.dx(:, obj.N+1) = dx(obj.N*obj.nx+1:(obj.N+1)*obj.nx);
                    otherwise
                        error("Invalid Solver name != ['TR', 'casadi']");
                end
  
                obj.updateSol(); % Update x = x + dx, u = u + du, lambda = lambda + dlambda
                obj.evalLagrangian(); % Evaluate Langrangian at current guess
                obj.LagrangianCurve = [obj.LagrangianCurve obj.Lagrangian];
                obj.costCurve = [obj.costCurve obj.cost.COST(obj.primal.x', obj.primal.u')]; % Update cost
                obj.updateRadius(); % Update the trust region radius (DELTA)
                
                % Check if the cost decrease is below the tolerance
                obj.stopCriterion = abs((obj.costCurve(end) - obj.costCurve(end-1))) < obj.tol;
                
                if obj.showIter % Plot current solution
                    plotIter(obj, i);
                    %pause;
                end
                
                i = i + 1;
            end
            
            % Returns the optimal solution
            x = obj.primal.x;
            u = obj.primal.u;
        end
        
        %
        % %%%%%%%%%%%%% PLOT EACH NEWTON'S ITERATION RESULT %%%%%%%%%%%%%%
        % 
        
        function [] = plotIter(obj, i)
            % Plot results
            figure(obj.fig_xu);
            set(gcf, 'Position', [100, 100, 1000, 400]);
            subplot(1, 2, 1);
            
            legend_str = [];
            
            % Plot the states trajectory
            for k = 1:1:obj.nx
                plot(obj.primal.x(k, :));
                legend_str = [legend_str strcat("x_", num2str(k))];
                hold on;
            end
            
            % Plot the inputs trajectory
            for k = 1:1:obj.nu
                plot(obj.primal.u(k, :));
                legend_str = [legend_str strcat("u_", num2str(k))];
                hold on;
            end
            
            title(strcat("Newton Step ", num2str(i)));
            legend(legend_str, 'Location' ,"SE");
            xlabel("Time [s]");
            ylabel("States / Inputs");
            grid on;
            hold off;        
            
            % Plot Cost & Lagrangian
            subplot(1, 2, 2);
            plot([0:i], obj.costCurve);
            hold on;
            plot([0:i], obj.LagrangianCurve, '--', 'LineWidth', 2);
            title("Objective Function");
            legend("Approx.", "Lagrangian");
            xlabel("Iteration [-]");
            ylabel("Cost [-]");
            hold off;
            grid on;
            
            saveas(gcf, strcat(obj.file, obj.solver, '_newtonStep_', num2str(i), '.png'));
            
%             if i == 1 
%                 obj.fig_xu_axis = axis;
%             end
%             
%             axis(obj.fig_xu_axis);
                        
%             xmin = 0;
%             xmax = i;
%             xSpan = xmax - xmin;
%             ymax = max([obj.costCurve(:); obj.LagrangianCurve(:)]);
%             ymin = min([obj.costCurve(:); obj.LagrangianCurve(:)]);
%             ySpan = ymax - ymin;
            
            %pos = [xmax - xSpan/8, ymax - ySpan/8, xSpan/8 * obj.DELTA / obj.DELTA_MAX, ySpan/8 * obj.DELTA / obj.DELTA_MAX]
            %rectangle('Position', pos, 'Curvature', [1, 1]);            
                        
            if i == obj.maxIter || obj.stopCriterion
                
                figure(obj.fig_xu);
                for k = 1:1:obj.nx
                    plot(obj.primal.time, obj.primal.x(k, :), 'LineWidth', 2);
                    hold on;
                end
                for k = 1:1:obj.nu
                    plot(obj.primal.time(1:end-1), obj.primal.u(k, :), 'LineWidth', 2);
                    hold on;
                end
                
                title(strcat(num2str(i), " Newton steps, ", obj.solver, " solver"));
                xlabel("Time-step [s]");
                ylabel("State / Input [-]");
                legend(legend_str);
                grid on;
                hold off;

                
                obj.fig_cost = figure;
                plot([0:1:i], obj.costCurve, 'LineWidth', 2);
                hold on;
                plot([0:1:i], obj.LagrangianCurve,'--', 'LineWidth', 2);
                xlabel("Iteration [-]");
                ylabel("Cost [-]");
                title(strcat("Objective function, ", obj.solver, " solver"));
                legend("Cost", "Lagrangian");
                grid on;  
                saveas(gcf, strcat(obj.file, obj.solver, "_obj", ".png"));
            end
        end
        
        %
        % %%%%%%%%%%%%%%%%%%%%%% EVALUATE THE CURRENT LAGRANGIAN %%%%%%%%%%%%%%%%%%%%%%%
        %        
        
        function [L] = evalLagrangian(obj)
            obj.Lagrangian = -obj.dual.lambda(:, 1)' * (obj.primal.x(:, 1) - obj.primal.xbar);
            
            for i = 1 : 1: obj.N
                xi = obj.primal.x(:, i); % x_i
                xip1 = obj.primal.x(:, i+1); % x_i+1
                ui = obj.primal.u(:, i); % ui
                lip1 = obj.dual.lambda(:, i+1); % lambda_i+1
                
                % Compute the sum
                obj.Lagrangian = obj.Lagrangian + ...
                        obj.cost.l(xi, ui) + ...
                        lip1' * (xi + obj.dynamic.f{i}(xi, ui) * obj.dt - xip1);
            end
            % Add sum at iteration N
            xN = obj.primal.x(:, i+1);
            obj.Lagrangian = obj.Lagrangian + obj.cost.Vf(xN);
            L = obj.Lagrangian;
        end
        
        %
        % %%%%%%%%%%%%%%%%%%%%%% UPDATE THE CURRENT TRUST RADIUS %%%%%%%%%%%%%%%%%%%%%%%
        %
        
        function [DELTA] = updateRadius(obj)
            % Compute a measure of confidence that we have in our
            % approximation
            rho = (obj.LagrangianCurve(end-1) - obj.LagrangianCurve(end))/(obj.costCurve(end-1) - obj.costCurve(end));
            
            % Update the trust radius depending on the confidence we have
            if rho < 0.25
                obj.DELTA = 0.25 * obj.DELTA;
            elseif rho > 0.75
                obj.DELTA = min(2 * obj.DELTA, obj.DELTA_MAX);
            end
            DELTA = obj.DELTA;
        end
        
        %
        % %%%%%%%%%%%%%%%%%%%%%% UPDATE THE NEWTON STEP OF LAMBDA %%%%%%%%%%%%%%%%%%%%%%%
        %

        function [dlambda] = update_dlambda(obj)
            obj.dual.dlambda = zeros(obj.nx, obj.N+1);
            obj.dual.dlambda(:, obj.N+1) = obj.r_x{obj.N+1} + obj.Q_xx{obj.N+1} * obj.primal.dx(:, obj.N+1);
            
            % Iteratively find the Newton step dlambda to find lambda +=
            % dlambda
            for i = obj.N : -1 : 1
                obj.dual.dlambda(:, i) = obj.r_x{i} + obj.A{i} * obj.dual.dlambda(:, i+1) + obj.Q_xx{i} * obj.primal.dx(:, i) + obj.Q_xu{i} * obj.primal.du(:, i);
            end
            dlambda = obj.dual.dlambda;
        end
    end
end