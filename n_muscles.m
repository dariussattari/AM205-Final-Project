classdef n_muscles
    % n_muscles: Muscle redundancy problem definition
    %
    % The muscle redundancy problem: find forces f that minimize J(f)
    % subject to:
    %   - Equality: sum(r_i * f_i) = tau_d  (torque constraint)
    %   - Inequality: 0 <= f_i <= f_max_i   (force bounds)
    %
    % This class defines the problem and provides MATLAB built-in solvers
    % for comparison with custom implementations.
    
    properties
        n          % number of muscles
        r          % moment arms (m) [n x 1]
        tau_d      % desired torque (N-m)
        f_max      % max forces (N) [n x 1]
        name       % problem name/description
    end
    
    methods
        function obj = n_muscles(n, tau_d, varargin)
            % Constructor: create n-muscle problem
            %
            % Usage:
            %   prob = n_muscles(n, tau_d)              % random problem
            %   prob = n_muscles(n, tau_d, 'hard')      % constraints active
            %   prob = n_muscles(n, tau_d, 'custom', r, f_max)
            
            if nargin < 1, n = 10; end
            if nargin < 2, tau_d = 20; end
            
            obj.n = n;
            obj.tau_d = tau_d;
            
            % Parse optional arguments
            if nargin >= 3 && strcmp(varargin{1}, 'hard')
                % Create problem where constraints ARE active at solution
                obj = obj.create_hard_problem();
                obj.name = sprintf('%d-muscle (constrained)', n);
            elseif nargin >= 5 && strcmp(varargin{1}, 'custom')
                obj.r = varargin{2};
                obj.f_max = varargin{3};
                obj.name = sprintf('%d-muscle (custom)', n);
            else
                % Default: random problem (may or may not hit bounds)
                obj.r = 0.02 + 0.03*rand(n, 1);      % moment arms 2-5 cm
                obj.f_max = 800 + 400*rand(n, 1);    % max forces 800-1200 N
                obj.name = sprintf('%d-muscle (random)', n);
            end
        end
        
        function obj = create_hard_problem(obj)
            % Create a problem where the solution hits bounds
            % Strategy: make tau_d large enough that some muscles must
            % operate at or near their limits
            
            n = obj.n;
            
            % Create moment arms with variation
            obj.r = 0.03 + 0.02*rand(n, 1);  % 3-5 cm
            
            % Create max forces with variation
            obj.f_max = 200 + 100*rand(n, 1);  % 200-300 N (smaller to force saturation)
            
            % Set tau_d to require ~70-80% of total capacity
            % This ensures some muscles hit limits
            max_torque = sum(obj.r .* obj.f_max);
            obj.tau_d = 0.75 * max_torque;
        end
        
        function [f, fval, exitflag] = solve_quadratic(obj)
            % Solve with quadprog: minimize sum(f_i^2)
            %
            % This is the reference solution for quadratic objective
            
            H = 2*eye(obj.n);
            f_lin = zeros(obj.n, 1);
            Aeq = obj.r';
            beq = obj.tau_d;
            lb = zeros(obj.n, 1);
            ub = obj.f_max;
            
            options = optimoptions('quadprog', 'Display', 'off');
            [f, fval, exitflag] = quadprog(H, f_lin, [], [], ...
                                           Aeq, beq, lb, ub, [], options);
        end
        
        function [f, fval, exitflag, output] = solve_fmincon(obj, objective_type, f0)
            % Solve with fmincon SQP for various objectives
            %
            % objective_type: 'quadratic', 'cubic', or 'minimax'
            
            if nargin < 2, objective_type = 'quadratic'; end
            if nargin < 3, f0 = 0.5 * obj.f_max; end
            
            % Select objective function
            switch objective_type
                case 'quadratic'
                    obj_fun = @(f) sum(f.^2);
                case 'cubic'
                    obj_fun = @(f) sum(f.^3);
                case 'minimax'
                    % Reformulate: min t s.t. f_i <= t for all i
                    % Add auxiliary variable t
                    obj_fun = @(x) x(end);  % minimize t
                    f0 = [f0; max(f0)];     % append initial t
                otherwise
                    error('Unknown objective: %s', objective_type);
            end
            
            % Set up constraints
            if strcmp(objective_type, 'minimax')
                % Extended problem: x = [f; t]
                Aeq = [obj.r', 0];
                beq = obj.tau_d;
                lb = [zeros(obj.n, 1); 0];
                ub = [obj.f_max; Inf];
                % Add inequality: f_i - t <= 0 for all i
                A_ineq = [eye(obj.n), -ones(obj.n, 1)];
                b_ineq = zeros(obj.n, 1);
            else
                Aeq = obj.r';
                beq = obj.tau_d;
                lb = zeros(obj.n, 1);
                ub = obj.f_max;
                A_ineq = [];
                b_ineq = [];
            end
            
            options = optimoptions('fmincon', 'Algorithm', 'sqp', ...
                                   'Display', 'off', ...
                                   'OptimalityTolerance', 1e-8);
            
            [x_opt, fval, exitflag, output] = fmincon(obj_fun, f0, ...
                                                       A_ineq, b_ineq, ...
                                                       Aeq, beq, lb, ub, ...
                                                       [], options);
            
            % Extract force solution
            if strcmp(objective_type, 'minimax')
                f = x_opt(1:obj.n);
            else
                f = x_opt;
            end
        end
        
        function feasible = check_feasibility(obj, f, tol)
            % Check if solution is feasible
            if nargin < 3, tol = 1e-6; end
            
            eq_satisfied = abs(obj.r' * f - obj.tau_d) < tol;
            lb_satisfied = all(f >= -tol);
            ub_satisfied = all(f <= obj.f_max + tol);
            
            feasible = eq_satisfied && lb_satisfied && ub_satisfied;
        end
        
        function [active_lb, active_ub] = get_active_constraints(obj, f, tol)
            % Identify which bound constraints are active
            if nargin < 3, tol = 1e-6; end
            
            active_lb = find(abs(f) < tol);
            active_ub = find(abs(f - obj.f_max) < tol);
        end
        
        function display_solution(obj, f, method_name)
            % Pretty-print a solution
            if nargin < 3, method_name = 'Solution'; end
            
            fprintf('\n%s for %s:\n', method_name, obj.name);
            fprintf('  Torque constraint: r''*f = %.4f (target: %.4f)\n', ...
                    obj.r' * f, obj.tau_d);
            fprintf('  Objective (sum f^2): %.4f\n', sum(f.^2));
            fprintf('  Forces: [');
            for i = 1:min(5, obj.n)
                if i > 1, fprintf(', '); end
                fprintf('%.2f', f(i));
            end
            if obj.n > 5, fprintf(', ...'); end
            fprintf(']\n');
            
            [active_lb, active_ub] = obj.get_active_constraints(f);
            if ~isempty(active_lb)
                fprintf('  Active lower bounds: muscles %s\n', mat2str(active_lb'));
            end
            if ~isempty(active_ub)
                fprintf('  Active upper bounds: muscles %s\n', mat2str(active_ub'));
            end
        end
    end
    
    methods (Static)
        function prob = create_2muscle_demo()
            % Create simple 2-muscle problem for visualization
            r = [0.03; 0.05];       % moment arms
            f_max = [300; 200];     % max forces
            tau_d = 18;             % requires both muscles near limits
            prob = n_muscles(2, tau_d, 'custom', r, f_max);
            prob.name = '2-muscle demo';
        end
        
        function prob = create_ill_conditioned(ratio)
            % Create 2-muscle problem with specified moment arm ratio
            % Higher ratio = worse conditioning
            if nargin < 1, ratio = 10; end
            
            r1 = 0.03;
            r2 = ratio * r1;
            r = [r1; r2];
            f_max = [500; 500];
            tau_d = 10;
            
            prob = n_muscles(2, tau_d, 'custom', r, f_max);
            prob.name = sprintf('2-muscle (r2/r1 = %.1f)', ratio);
        end
    end
end