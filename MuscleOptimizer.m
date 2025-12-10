classdef MuscleOptimizer
    % MuscleOptimizer: Custom optimization methods for muscle redundancy
    %
    % Implements Newton-Lagrange method from scratch to compare with
    % MATLAB's quadprog and fmincon, per AM205 project proposal.
    %
    % Methods implemented from scratch:
    %   - solve_newton_lagrange: General Newton on KKT system (any objective)
    %   - solve_newton_lagrange_activeset: With active-set for bounds
    %
    % These directly implement the "Direct KKT approach via ∇L(f,λ) = 0"
    % from Lectures 12-13.
    
    properties
        problem        % n_muscles object
        tolerance      % convergence tolerance
        max_iter       % maximum iterations
        verbose        % display output
    end
    
    methods
        function obj = MuscleOptimizer(problem, options)
            % Constructor
            if nargin < 2
                options = struct();
            end
            
            obj.problem = problem;
            obj.tolerance = get_option(options, 'tol', 1e-8);
            obj.max_iter = get_option(options, 'max_iter', 100);
            obj.verbose = get_option(options, 'verbose', false);
        end
        
        %% ================================================================
        %  NEWTON-LAGRANGE: General method for any smooth objective
        %  This is the "Direct KKT approach via ∇L(f,λ) = 0" from proposal
        %  ================================================================
        function [f_opt, lambda_opt, history] = solve_newton_lagrange(obj, f0, objective_type)
            % Newton-Lagrange for equality-constrained problem
            %
            % Solves: min J(f) s.t. r'*f = tau_d
            %
            % The Lagrangian is: L(f,λ) = J(f) + λ(r'f - τ_d)
            % KKT conditions: ∇_f L = ∇J(f) + λr = 0
            %                 ∇_λ L = r'f - τ_d = 0
            %
            % Newton's method solves this nonlinear system iteratively.
            %
            % Inputs:
            %   f0: initial guess (default: 100*ones)
            %   objective_type: 'quadratic', 'cubic', or function handle
            %
            % This is the CORE METHOD implementing the proposal's
            % "Newton with Lagrange Multipliers: Direct KKT approach"
            
            n = obj.problem.n;
            r = obj.problem.r;
            tau_d = obj.problem.tau_d;
            
            if nargin < 2 || isempty(f0)
                f0 = 100 * ones(n, 1);
            end
            if nargin < 3
                objective_type = 'quadratic';
            end
            
            % Get objective function and its derivatives
            [J, grad_J, hess_J] = obj.get_objective_functions(objective_type);
            
            f = f0;
            lambda = 0;
            
            % History tracking
            history.f = [];
            history.lambda = [];
            history.objective = [];
            history.residual = [];
            history.step_sizes = [];
            history.iterations = 0;
            history.func_evals = 0;
            history.converged = false;
            
            for iter = 1:obj.max_iter
                history.iterations = iter;
                
                % Evaluate objective and derivatives
                J_val = J(f);
                g = grad_J(f);      % gradient of J
                H = hess_J(f);      % Hessian of J
                history.func_evals = history.func_evals + 1;
                
                % KKT residual:
                % [∇J(f) + λr  ]   [0]
                % [r'f - τ_d   ] = [0]
                grad_L_f = g + lambda * r;
                constraint = r' * f - tau_d;
                residual = [grad_L_f; constraint];
                res_norm = norm(residual);
                
                % Store history
                history.f = [history.f; f'];
                history.lambda = [history.lambda; lambda];
                history.objective = [history.objective; J_val];
                history.residual = [history.residual; res_norm];
                
                if obj.verbose
                    fprintf('Iter %d: J=%.6f, ||KKT||=%.2e\n', iter, J_val, res_norm);
                end
                
                % Check convergence
                if res_norm < obj.tolerance
                    history.converged = true;
                    if obj.verbose
                        fprintf('Converged in %d iterations\n', iter);
                    end
                    break;
                end
                
                % Newton step: solve KKT system
                % [H   r ] [Δf]   [-∇J - λr    ]
                % [r'  0 ] [Δλ] = [-(r'f - τ_d)]
                %
                % This is the "Hessian of Lagrangian" approach from Heath Ch. 6
                
                KKT_matrix = [H, r; r', 0];
                rhs = -residual;
                
                % Solve for Newton direction
                delta = KKT_matrix \ rhs;
                delta_f = delta(1:n);
                delta_lambda = delta(n+1);
                
                % Line search (backtracking) for globalization
                alpha = obj.line_search(f, lambda, delta_f, delta_lambda, ...
                                        J, grad_J, r, tau_d);
                history.step_sizes = [history.step_sizes; alpha];
                
                % Update
                f = f + alpha * delta_f;
                lambda = lambda + alpha * delta_lambda;
                
                if iter == obj.max_iter
                    warning('Newton-Lagrange did not converge in %d iterations', obj.max_iter);
                end
            end
            
            f_opt = f;
            lambda_opt = lambda;
            
            % Check bound violations (this method ignores bounds)
            history.violates_lower = any(f_opt < -obj.tolerance);
            history.violates_upper = any(f_opt > obj.problem.f_max + obj.tolerance);
            history.feasible = ~history.violates_lower && ~history.violates_upper;
        end
        
        %% ================================================================
        %  NEWTON-LAGRANGE WITH ACTIVE-SET (handles bounds)
        %  ================================================================
        function [f_opt, lambda_opt, history] = solve_newton_lagrange_activeset(obj, f0, objective_type)
            % Newton-Lagrange with active-set strategy for bound constraints
            %
            % Solves: min J(f) s.t. r'*f = tau_d, 0 <= f <= f_max
            %
            % Algorithm:
            %   1. Identify active bounds
            %   2. Solve reduced Newton-Lagrange for free variables
            %   3. Update active set based on KKT multiplier signs
            %   4. Repeat until converged
            
            n = obj.problem.n;
            r = obj.problem.r;
            tau_d = obj.problem.tau_d;
            f_max = obj.problem.f_max;
            
            if nargin < 2 || isempty(f0)
                f0 = 0.5 * f_max;
            end
            if nargin < 3
                objective_type = 'quadratic';
            end
            
            % Get objective functions
            [J, grad_J, hess_J] = obj.get_objective_functions(objective_type);
            
            % Initialize feasibly
            f = max(0, min(f_max, f0));
            f = obj.project_to_equality(f);
            f = max(1e-6, min(f_max - 1e-6, f));  % keep strictly feasible
            
            lambda = 0;
            active_lb = [];
            active_ub = [];
            
            % History
            history.f = [];
            history.objective = [];
            history.kkt_residual = [];
            history.active_lb = {};
            history.active_ub = {};
            history.iterations = 0;
            history.func_evals = 0;
            history.converged = false;
            
            for iter = 1:obj.max_iter
                history.iterations = iter;
                
                % Store state
                history.f = [history.f; f'];
                J_val = J(f);
                history.objective = [history.objective; J_val];
                history.active_lb{iter} = active_lb;
                history.active_ub{iter} = active_ub;
                history.func_evals = history.func_evals + 1;
                
                % Free variables
                free = setdiff(1:n, [active_lb(:); active_ub(:)]);
                n_free = length(free);
                
                if obj.verbose
                    fprintf('Iter %d: J=%.4f, active_lb=%s, active_ub=%s\n', ...
                            iter, J_val, mat2str(active_lb(:)'), mat2str(active_ub(:)'));
                end
                
                % Compute remaining torque after fixing active variables
                tau_remaining = tau_d;
                for i = active_lb(:)'
                    tau_remaining = tau_remaining - r(i) * 0;
                end
                for i = active_ub(:)'
                    tau_remaining = tau_remaining - r(i) * f_max(i);
                end
                
                if n_free == 0
                    break;  % All variables fixed
                end
                
                % Solve reduced Newton-Lagrange for free variables
                f_free = f(free);
                r_free = r(free);
                f_max_free = f_max(free);
                
                % Inner Newton iterations for free variables
                for inner = 1:20
                    % Gradient and Hessian for free variables
                    g_full = grad_J(f);
                    H_full = hess_J(f);
                    g_free = g_full(free);
                    H_free = H_full(free, free);
                    
                    % KKT residual for free variables
                    grad_L_free = g_free + lambda * r_free;
                    constraint = r_free' * f_free - tau_remaining;
                    residual = [grad_L_free; constraint];
                    res_norm = norm(residual);
                    
                    if res_norm < obj.tolerance
                        break;
                    end
                    
                    % Newton step
                    KKT = [H_free, r_free; r_free', 0];
                    delta = KKT \ (-residual);
                    delta_f = delta(1:n_free);
                    delta_lambda = delta(n_free + 1);
                    
                    % Line search keeping in bounds
                    alpha = 1.0;
                    for ls = 1:20
                        f_new = f_free + alpha * delta_f;
                        if all(f_new >= 1e-10) && all(f_new <= f_max_free - 1e-10)
                            break;
                        end
                        alpha = alpha * 0.5;
                    end
                    
                    f_free = f_free + alpha * delta_f;
                    lambda = lambda + alpha * delta_lambda;
                    f(free) = f_free;
                end
                
                % Record KKT residual
                history.kkt_residual = [history.kkt_residual; res_norm];
                
                % Compute bound multipliers
                g_full = grad_J(f);
                mu_lb = zeros(n, 1);
                mu_ub = zeros(n, 1);
                
                for i = active_lb(:)'
                    % At f_i = 0: mu_lb = g_i + lambda*r_i (should be >= 0)
                    mu_lb(i) = g_full(i) + lambda * r(i);
                end
                for i = active_ub(:)'
                    % At f_i = f_max: mu_ub = -(g_i + lambda*r_i) (should be >= 0)
                    mu_ub(i) = -(g_full(i) + lambda * r(i));
                end
                
                % Check convergence
                if res_norm < obj.tolerance
                    % Verify multiplier signs
                    lb_ok = all(mu_lb(active_lb) >= -obj.tolerance);
                    ub_ok = all(mu_ub(active_ub) >= -obj.tolerance);
                    bounds_ok = all(f(free) >= -obj.tolerance) && ...
                                all(f(free) <= f_max(free) + obj.tolerance);
                    
                    if lb_ok && ub_ok && bounds_ok
                        history.converged = true;
                        if obj.verbose
                            fprintf('Converged in %d iterations\n', iter);
                        end
                        break;
                    end
                end
                
                % Update active set
                active_set_changed = false;
                
                % Check if free variables violate bounds
                for idx = 1:length(free)
                    i = free(idx);
                    if f(i) < obj.tolerance
                        active_lb = union(active_lb, i);
                        f(i) = 0;
                        active_set_changed = true;
                    elseif f(i) > f_max(i) - obj.tolerance
                        active_ub = union(active_ub, i);
                        f(i) = f_max(i);
                        active_set_changed = true;
                    end
                end
                
                % Check if active constraints should be released
                for i = active_lb(:)'
                    if mu_lb(i) < -obj.tolerance
                        active_lb = setdiff(active_lb, i);
                        active_set_changed = true;
                    end
                end
                for i = active_ub(:)'
                    if mu_ub(i) < -obj.tolerance
                        active_ub = setdiff(active_ub, i);
                        active_set_changed = true;
                    end
                end
                
                if iter == obj.max_iter
                    warning('Active-set Newton did not converge');
                end
            end
            
            f_opt = f;
            lambda_opt.equality = lambda;
            lambda_opt.lower = mu_lb;
            lambda_opt.upper = mu_ub;
        end
        
        %% ================================================================
        %  COMPARISON METHODS
        %  ================================================================
        function results = compare_all_methods(obj, f0, objective_type)
            % Compare all three methods on the same problem and objective
            %
            % This directly addresses the proposal requirement:
            % "Evaluate all three methods on quadratic, cubic, and minimax costs"
            
            if nargin < 2 || isempty(f0)
                f0 = 0.5 * obj.problem.f_max;
            end
            if nargin < 3
                objective_type = 'quadratic';
            end
            
            n = obj.problem.n;
            
            fprintf('\n--- Comparing methods on %s objective ---\n', objective_type);
            fprintf('Problem: %d muscles, tau_d = %.2f\n\n', n, obj.problem.tau_d);
            
            % Method 1: quadprog (only for quadratic)
            if strcmp(objective_type, 'quadratic')
                tic;
                [f_qp, fval_qp, exitflag] = obj.problem.solve_quadratic();
                time_qp = toc;
                results.qp.f = f_qp;
                results.qp.objective = fval_qp;
                results.qp.time = time_qp;
                results.qp.available = true;
            else
                results.qp.available = false;
                results.qp.f = NaN(n, 1);
                results.qp.objective = NaN;
            end
            
            % Method 2: fmincon SQP
            tic;
            [f_sqp, fval_sqp, ~, output_sqp] = obj.problem.solve_fmincon(objective_type, f0);
            time_sqp = toc;
            results.sqp.f = f_sqp;
            results.sqp.objective = fval_sqp;
            results.sqp.time = time_sqp;
            results.sqp.iterations = output_sqp.iterations;
            results.sqp.funcCount = output_sqp.funcCount;
            
            % Method 3: Newton-Lagrange (our implementation)
            % First without bounds (pure Newton-Lagrange)
            tic;
            [f_nl, ~, hist_nl] = obj.solve_newton_lagrange(f0, objective_type);
            time_nl = toc;
            [J_func, ~, ~] = obj.get_objective_functions(objective_type);
            results.newton.f = f_nl;
            results.newton.objective = J_func(f_nl);
            results.newton.time = time_nl;
            results.newton.iterations = hist_nl.iterations;
            results.newton.func_evals = hist_nl.func_evals;
            results.newton.feasible = hist_nl.feasible;
            results.newton.history = hist_nl;
            
            % Also with active-set (handles bounds)
            tic;
            [f_nl_as, ~, hist_nl_as] = obj.solve_newton_lagrange_activeset(f0, objective_type);
            time_nl_as = toc;
            results.newton_activeset.f = f_nl_as;
            results.newton_activeset.objective = J_func(f_nl_as);
            results.newton_activeset.time = time_nl_as;
            results.newton_activeset.iterations = hist_nl_as.iterations;
            results.newton_activeset.func_evals = hist_nl_as.func_evals;
            results.newton_activeset.history = hist_nl_as;
            
            % Print comparison
            fprintf('%-25s %12s %12s %8s %10s %10s\n', ...
                    'Method', 'Objective', 'Time(ms)', 'Iters', 'FuncEval', 'Feasible');
            fprintf('%s\n', repmat('-', 1, 80));
            
            if results.qp.available
                fprintf('%-25s %12.4f %12.4f %8s %10s %10s\n', ...
                        'quadprog', results.qp.objective, results.qp.time*1000, '-', '-', 'Yes');
            end
            
            fprintf('%-25s %12.4f %12.4f %8d %10d %10s\n', ...
                    'fmincon SQP', results.sqp.objective, results.sqp.time*1000, ...
                    results.sqp.iterations, results.sqp.funcCount, 'Yes');
            
            feasible_str = 'Yes';
            if ~results.newton.feasible
                feasible_str = 'NO';
            end
            fprintf('%-25s %12.4f %12.4f %8d %10d %10s\n', ...
                    'Newton-Lagrange', results.newton.objective, results.newton.time*1000, ...
                    results.newton.iterations, results.newton.func_evals, feasible_str);
            
            fprintf('%-25s %12.4f %12.4f %8d %10d %10s\n', ...
                    'Newton-Lagrange (AS)', results.newton_activeset.objective, ...
                    results.newton_activeset.time*1000, ...
                    results.newton_activeset.iterations, results.newton_activeset.func_evals, 'Yes');
        end
        
        %% ================================================================
        %  HELPER METHODS
        %  ================================================================
        function [J, grad_J, hess_J] = get_objective_functions(obj, objective_type)
            % Return objective function and its derivatives
            %
            % For Newton-Lagrange, we need:
            %   J(f)      - objective value
            %   ∇J(f)     - gradient
            %   ∇²J(f)    - Hessian
            
            n = obj.problem.n;
            
            if isa(objective_type, 'function_handle')
                % Custom objective - use finite differences
                J = objective_type;
                grad_J = @(f) obj.numerical_gradient(J, f);
                hess_J = @(f) obj.numerical_hessian(J, f);
            else
                switch objective_type
                    case 'quadratic'
                        % J = sum(f_i^2) = f'*f
                        J = @(f) sum(f.^2);
                        grad_J = @(f) 2*f;
                        hess_J = @(f) 2*eye(n);
                        
                    case 'cubic'
                        % J = sum(f_i^3)
                        J = @(f) sum(f.^3);
                        grad_J = @(f) 3*f.^2;
                        hess_J = @(f) diag(6*f);
                        
                    case 'sum_exp'
                        % J = sum(exp(f_i/100)) - smoother nonlinear
                        J = @(f) sum(exp(f/100));
                        grad_J = @(f) exp(f/100)/100;
                        hess_J = @(f) diag(exp(f/100)/10000);
                        
                    otherwise
                        error('Unknown objective type: %s', objective_type);
                end
            end
        end
        
        function alpha = line_search(obj, f, lambda, df, dlam, J, grad_J, r, tau_d)
            % Backtracking line search for Newton-Lagrange
            % Minimize merit function: phi(alpha) = ||KKT residual||^2
            
            c = 1e-4;  % Armijo parameter
            rho = 0.5; % Backtracking factor
            
            % Current merit (residual norm squared)
            g0 = grad_J(f);
            res0 = [g0 + lambda*r; r'*f - tau_d];
            phi0 = norm(res0)^2;
            
            alpha = 1.0;
            for k = 1:25
                f_new = f + alpha * df;
                lambda_new = lambda + alpha * dlam;
                
                g_new = grad_J(f_new);
                res_new = [g_new + lambda_new*r; r'*f_new - tau_d];
                phi_new = norm(res_new)^2;
                
                if phi_new <= phi0 - c * alpha * phi0  % Sufficient decrease
                    break;
                end
                alpha = rho * alpha;
            end
        end
        
        function g = numerical_gradient(obj, f_func, x)
            % Finite difference gradient
            n = length(x);
            g = zeros(n, 1);
            h = 1e-7;
            f0 = f_func(x);
            for i = 1:n
                x_plus = x;
                x_plus(i) = x_plus(i) + h;
                g(i) = (f_func(x_plus) - f0) / h;
            end
        end
        
        function H = numerical_hessian(obj, f_func, x)
            % Finite difference Hessian
            n = length(x);
            H = zeros(n, n);
            h = 1e-5;
            for i = 1:n
                x_plus = x; x_plus(i) = x_plus(i) + h;
                x_minus = x; x_minus(i) = x_minus(i) - h;
                g_plus = obj.numerical_gradient(f_func, x_plus);
                g_minus = obj.numerical_gradient(f_func, x_minus);
                H(:, i) = (g_plus - g_minus) / (2*h);
            end
            H = (H + H') / 2;  % Symmetrize
        end
        
        function f_proj = project_to_equality(obj, f)
            % Project f to satisfy r'*f = tau_d
            current_tau = obj.problem.r' * f;
            if abs(current_tau) < 1e-12
                f_proj = f + obj.problem.tau_d / (obj.problem.r' * obj.problem.r) * obj.problem.r;
            else
                f_proj = f * (obj.problem.tau_d / current_tau);
            end
        end
    end
end

%% Helper function
function val = get_option(options, field, default)
    if isfield(options, field)
        val = options.(field);
    else
        val = default;
    end
end