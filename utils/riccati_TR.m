classdef riccati_TR < handle
    properties
        % Objective variables
        Q, R, S, q, r, b
        % Linearized dynamic
        A, B
        % Inner variables
        R_bar, P, Lambda, L, l, p
        % Horizon
        N
        % u-dimensions, x-dimensions
        nx, nu
        % Newton steps
        du, dx
        % Lambda and TR radius
        LAMBDA, DELTA
    end
    methods
        
        %
        % %%%%%%%%%%%%%%%%%% CONSTRUCTOR %%%%%%%%%%%%%%%%%%%
        %
        
        function obj = riccati_TR(N, LAMBDA, DELTA, A, B, Q, R, S, q, r, b)
            obj.N = N;
            obj.LAMBDA = LAMBDA;
            obj.DELTA = DELTA;
            obj.A = A;
            obj.B = B;
            obj.nx = size(A{1}, 2);
            obj.nu = size(B{1}, 2);
            obj.Q = Q;
            obj.R = R;
            obj.S = S;
            obj.q = q;
            obj.r = r;
            obj.b = b;
            obj.R_bar = cell(N, 1);
            obj.P = cell(N+1, 1);
            obj.Lambda = cell(N, 1);
            obj.L = cell(N, 1);
            obj.l = cell(N, 1);
            obj.p = cell(N+1, 1);
        end
        
        %
        % %%%%%%%%%%%%%%%%%% BACKWARD RICCATI RECURSION %%%%%%%%%%%%%%%%%%%
        %
      
        function [] = backward(obj)
            obj.P{obj.N+1} = obj.Q{obj.N+1};
            obj.p{obj.N+1} = obj.q{obj.N+1};
            
            % Backward recursion
            for k = obj.N : -1 : 1
                obj.R_bar{k} = obj.R{k} + obj.LAMBDA + obj.B{k}' * obj.P{k+1} * obj.B{k};
                obj.Lambda{k} = chol(obj.R_bar{k});
                obj.L{k} = obj.Lambda{k}' \ (obj.S{k} + obj.B{k}' * obj.P{k+1} * obj.A{k});
                obj.P{k} = obj.Q{k} + obj.A{k}' * obj.P{k+1} * obj.A{k} - obj.L{k}' * obj.L{k};
                obj.l{k} = obj.Lambda{k}' \ (obj.r{k} + obj.B{k}' * (obj.P{k+1} * obj.b{k} + obj.p{k+1}));
                obj.p{k} = obj.q{k} + obj.A{k}' * (obj.P{k+1} * obj.b{k} + obj.p{k+1}) - obj.L{k}' * obj.l{k};
            end
        end
        
        %
        % %%%%%%%%%%%%%%%%%% FORWARD RICCATI RECURSION %%%%%%%%%%%%%%%%%%%%
        %
      
        function [dx, du] = forward(obj)
            du = zeros(obj.nu, obj.N);
            dx = zeros(obj.nx, obj.N+1);
            
            % Forward propagation
            for k = 1 : 1 : obj.N
                du(:, k) = - obj.Lambda{k} \ (obj.L{k} * dx(:, k) + obj.l{k});
                dx(:, k+1) = obj.A{k} * dx(:, k) + obj.B{k} * du(:, k) + obj.b{k};
            end
            
            obj.du = du;
            obj.dx = dx;
        end
        
        % 
        % %%%%%%%%%%%%%%%%%%% Q-UPDATE & LAMBDA UPDATE %%%%%%%%%%%%%%%%%%%% 
        %
        
        function [LAMBDA] = updateLambda(obj)
            % Initialize beta and q
            beta = zeros(obj.nx, 1);
            q_norm_squared = 0;
            q_temp = cell(obj.N, 1);
            
            % q-update -> Used to update LAMBDA
            for k = obj.N : -1 : 1
                alpha = obj.du(:, k) + obj.B{k}' * beta;
                q_temp{k} = obj.Lambda{k}' \ alpha;
                beta = obj.A{k}' * beta - obj.L{k}' * (obj.Lambda{k}' \ alpha);
                q_norm_squared = q_norm_squared + q_temp{k}' * q_temp{k};
            end
            
            % LAMBDA-update           
            obj.LAMBDA = obj.LAMBDA + eye(obj.nu) .* norm(obj.du)^2 * (1 * norm(obj.du) - obj.DELTA) / (q_norm_squared * obj.DELTA);
            LAMBDA = obj.LAMBDA;
        end
        
        %
        % %%%%%%%%%%%%%%%%%%%%%%%% SOLVE TR-RICCATI %%%%%%%%%%%%%%%%%%%%%%%
        %
        
        function [dx, du] = solve(obj)
            obj.backward();
            obj.forward();
            
            % While norm > trust radius -> update LAMBDA & solve
            while norm(obj.du, 2) > obj.DELTA
                obj.updateLambda();
                obj.backward();
                obj.forward();
            end
            % At this point Newton step du is inside the trust region
            du = obj.du;
            dx = obj.dx;           
        end     
    end
end



