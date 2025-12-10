%% AM205 Final Project: Muscle Redundancy Optimization Comparison
% Darius Sattari
%
% Comparing three optimization methods per project proposal:
%   1. Quadratic Programming (QP): MATLAB's quadprog (quadratic obj only)
%   2. Sequential Quadratic Programming (SQP): MATLAB's fmincon with SQP
%   3. Newton with Lagrange Multipliers: Custom implementation (from scratch)
%
% Experiments (from proposal):
%   1. Convergence: iteration counts, function evals, sensitivity to initial guess (2-muscle)
%   2. Scaling: computation time vs problem size (n = 2 to 20 muscles)
%   3. Nonlinear objectives: all three methods on quadratic, cubic, minimax
%   4. Conditioning: vary moment arm ratios to test robustness

clear; close all; clc;

%% Setup
fprintf('========================================\n');
fprintf('AM205 Final Project\n');
fprintf('Muscle Redundancy Optimization\n');
fprintf('Comparing QP, SQP, and Newton-Lagrange\n');
fprintf('========================================\n');

if ~exist('figures', 'dir')
    mkdir('figures');
end

rng(42);  % Reproducibility

%% ========================================================================
%  EXPERIMENT 1: Convergence Analysis (2-muscle case)
%  ========================================================================
fprintf('\n\n========================================\n');
fprintf('EXPERIMENT 1: Convergence Analysis\n');
fprintf('========================================\n');
fprintf('2-muscle problem, different initial guesses\n');
fprintf('Tracking: iterations, function evaluations, time\n\n');

% Create 2-muscle problem where bounds are active
prob1 = n_muscles.create_2muscle_demo();
opt1 = MuscleOptimizer(prob1, struct('tol', 1e-10, 'verbose', false));

fprintf('Problem: %s\n', prob1.name);
fprintf('Moment arms: r = [%.3f, %.3f] m\n', prob1.r(1), prob1.r(2));
fprintf('Max forces: f_max = [%.0f, %.0f] N\n', prob1.f_max(1), prob1.f_max(2));
fprintf('Desired torque: tau_d = %.1f N-m\n\n', prob1.tau_d);

% Get reference solution
[f_ref, ~] = prob1.solve_quadratic();
fprintf('Reference solution (quadprog): [%.2f, %.2f]\n\n', f_ref(1), f_ref(2));

% Initial guesses to test
initial_guesses = {
    [50; 50],       'Low uniform';
    [150; 150],     'Mid uniform';
    [250; 250],     'High uniform';
    [50; 200],      'Asymmetric 1';
    [200; 50],      'Asymmetric 2';
    [10; 10],       'Very low';
    f_ref + [20; -20], 'Near optimal';
    [300; 300],     'Above bounds';
};
n_guesses = size(initial_guesses, 1);

% Storage for all three methods
exp1_qp_time = zeros(n_guesses, 1);
exp1_sqp_iters = zeros(n_guesses, 1);
exp1_sqp_fevals = zeros(n_guesses, 1);
exp1_sqp_time = zeros(n_guesses, 1);
exp1_newton_iters = zeros(n_guesses, 1);
exp1_newton_fevals = zeros(n_guesses, 1);
exp1_newton_time = zeros(n_guesses, 1);
exp1_newton_residuals = cell(n_guesses, 1);

fprintf('%-15s | %-20s | %-25s | %-25s\n', '', 'QP (quadprog)', 'SQP (fmincon)', 'Newton-Lagrange');
fprintf('%-15s | %6s %6s %6s | %6s %6s %6s %6s | %6s %6s %6s %6s\n', ...
        'Initial Guess', 'time', '', '', 'iters', 'feval', 'time', '', 'iters', 'feval', 'time', '');
fprintf('%s\n', repmat('-', 1, 95));

for i = 1:n_guesses
    f0 = initial_guesses{i, 1};
    name = initial_guesses{i, 2};
    
    % QP (quadprog) - not sensitive to initial guess
    tic;
    [f_qp, ~] = prob1.solve_quadratic();
    exp1_qp_time(i) = toc;
    
    % SQP (fmincon)
    tic;
    [f_sqp, ~, ~, output_sqp] = prob1.solve_fmincon('quadratic', f0);
    exp1_sqp_time(i) = toc;
    exp1_sqp_iters(i) = output_sqp.iterations;
    exp1_sqp_fevals(i) = output_sqp.funcCount;
    
    % Newton-Lagrange (our implementation)
    tic;
    [f_newton, ~, hist_newton] = opt1.solve_newton_lagrange_activeset(f0, 'quadratic');
    exp1_newton_time(i) = toc;
    exp1_newton_iters(i) = hist_newton.iterations;
    exp1_newton_fevals(i) = hist_newton.func_evals;
    exp1_newton_residuals{i} = hist_newton.kkt_residual;
    
    fprintf('%-15s | %6.2f %6s %6s | %6d %6d %6.2f %6s | %6d %6d %6.2f %6s\n', ...
            name, exp1_qp_time(i)*1000, '', '', ...
            exp1_sqp_iters(i), exp1_sqp_fevals(i), exp1_sqp_time(i)*1000, '', ...
            exp1_newton_iters(i), exp1_newton_fevals(i), exp1_newton_time(i)*1000, '');
end

% Plot Experiment 1
fig1 = figure('Position', [100, 100, 1400, 500]);

% Panel 1: Iterations comparison (SQP vs Newton)
subplot(1, 3, 1);
bar_data = [exp1_sqp_iters, exp1_newton_iters];
bar(bar_data);
set(gca, 'XTickLabel', cellfun(@(x) x, initial_guesses(:,2), 'UniformOutput', false));
xtickangle(45);
ylabel('Iterations', 'FontSize', 12);
title('Iterations vs Initial Guess (2-muscle)', 'FontSize', 14);
legend('SQP (fmincon)', 'Newton-Lagrange', 'Location', 'best');
grid on;

% Panel 2: Function evaluations
subplot(1, 3, 2);
bar_data = [exp1_sqp_fevals, exp1_newton_fevals];
bar(bar_data);
set(gca, 'XTickLabel', cellfun(@(x) x, initial_guesses(:,2), 'UniformOutput', false));
xtickangle(45);
ylabel('Function Evaluations', 'FontSize', 12);
title('Function Evaluations vs Initial Guess', 'FontSize', 14);
legend('SQP (fmincon)', 'Newton-Lagrange', 'Location', 'best');
grid on;

% Panel 3: Newton convergence trajectories
subplot(1, 3, 3);
colors = lines(n_guesses);
hold on;
for i = 1:n_guesses
    if ~isempty(exp1_newton_residuals{i}) && length(exp1_newton_residuals{i}) > 1
        semilogy(1:length(exp1_newton_residuals{i}), exp1_newton_residuals{i}, ...
                 'o-', 'LineWidth', 1.5, 'Color', colors(i,:), ...
                 'DisplayName', initial_guesses{i, 2});
    end
end
xlabel('Iteration', 'FontSize', 12);
ylabel('KKT Residual', 'FontSize', 12);
title('Newton-Lagrange Convergence', 'FontSize', 14);
legend('Location', 'best', 'FontSize', 8);
grid on;
hold off;

sgtitle('Experiment 1: Convergence Analysis (2-muscle problem)', 'FontSize', 16);
saveas(fig1, 'figures/exp1_convergence_analysis.png');
fprintf('\nFigure saved: figures/exp1_convergence_analysis.png\n');

%% ========================================================================
%  EXPERIMENT 2: Scaling Behavior (n = 2 to 20)
%  ========================================================================
fprintf('\n\n========================================\n');
fprintf('EXPERIMENT 2: Scaling Behavior\n');
fprintf('========================================\n');
fprintf('Problem size n = 2 to 20 muscles\n\n');

n_values = 2:2:20;
n_sizes = length(n_values);

% Storage
exp2_qp_time = zeros(n_sizes, 1);
exp2_sqp_iters = zeros(n_sizes, 1);
exp2_sqp_fevals = zeros(n_sizes, 1);
exp2_sqp_time = zeros(n_sizes, 1);
exp2_newton_iters = zeros(n_sizes, 1);
exp2_newton_fevals = zeros(n_sizes, 1);
exp2_newton_time = zeros(n_sizes, 1);
exp2_accuracy_sqp = zeros(n_sizes, 1);
exp2_accuracy_newton = zeros(n_sizes, 1);

fprintf('%-5s | %-15s | %-25s | %-25s\n', 'n', 'QP', 'SQP', 'Newton-Lagrange');
fprintf('%-5s | %7s %7s | %6s %6s %7s %6s | %6s %6s %7s %6s\n', ...
        '', 'time', 'err', 'iters', 'feval', 'time', 'err', 'iters', 'feval', 'time', 'err');
fprintf('%s\n', repmat('-', 1, 95));

for i = 1:n_sizes
    n = n_values(i);
    
    % Create problem with active constraints
    prob = n_muscles(n, 0, 'hard');
    opt = MuscleOptimizer(prob, struct('tol', 1e-8, 'verbose', false));
    f0 = 0.5 * prob.f_max;
    
    % QP (quadprog) - reference
    tic;
    [f_qp, ~] = prob.solve_quadratic();
    exp2_qp_time(i) = toc;
    
    % SQP (fmincon)
    tic;
    [f_sqp, ~, ~, output_sqp] = prob.solve_fmincon('quadratic', f0);
    exp2_sqp_time(i) = toc;
    exp2_sqp_iters(i) = output_sqp.iterations;
    exp2_sqp_fevals(i) = output_sqp.funcCount;
    exp2_accuracy_sqp(i) = norm(f_sqp - f_qp);
    
    % Newton-Lagrange (our implementation)
    tic;
    [f_newton, ~, hist_newton] = opt.solve_newton_lagrange_activeset(f0, 'quadratic');
    exp2_newton_time(i) = toc;
    exp2_newton_iters(i) = hist_newton.iterations;
    exp2_newton_fevals(i) = hist_newton.func_evals;
    exp2_accuracy_newton(i) = norm(f_newton - f_qp);
    
    fprintf('%-5d | %7.2f %7.1e | %6d %6d %7.2f %6.1e | %6d %6d %7.2f %6.1e\n', ...
            n, exp2_qp_time(i)*1000, 0, ...
            exp2_sqp_iters(i), exp2_sqp_fevals(i), exp2_sqp_time(i)*1000, exp2_accuracy_sqp(i), ...
            exp2_newton_iters(i), exp2_newton_fevals(i), exp2_newton_time(i)*1000, exp2_accuracy_newton(i));
end

% Plot Experiment 2
fig2 = figure('Position', [100, 100, 1400, 500]);

% Panel 1: Computation time
subplot(1, 3, 1);
semilogy(n_values, exp2_qp_time*1000, 'o-', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'QP (quadprog)');
hold on;
semilogy(n_values, exp2_sqp_time*1000, 's-', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'SQP (fmincon)');
semilogy(n_values, exp2_newton_time*1000, '^-', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'Newton-Lagrange');
xlabel('Number of Muscles (n)', 'FontSize', 12);
ylabel('Computation Time (ms)', 'FontSize', 12);
title('Computation Time vs Problem Size', 'FontSize', 14);
legend('Location', 'northwest', 'FontSize', 10);
grid on;
hold off;

% Panel 2: Iterations
subplot(1, 3, 2);
plot(n_values, exp2_sqp_iters, 's-', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'SQP (fmincon)');
hold on;
plot(n_values, exp2_newton_iters, '^-', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'Newton-Lagrange');
xlabel('Number of Muscles (n)', 'FontSize', 12);
ylabel('Iterations', 'FontSize', 12);
title('Iterations vs Problem Size', 'FontSize', 14);
legend('Location', 'best', 'FontSize', 10);
grid on;
hold off;

% Panel 3: Solution accuracy
subplot(1, 3, 3);
semilogy(n_values, exp2_accuracy_sqp, 's-', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'SQP (fmincon)');
hold on;
semilogy(n_values, exp2_accuracy_newton, '^-', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'Newton-Lagrange');
xlabel('Number of Muscles (n)', 'FontSize', 12);
ylabel('||f - f_{QP}||', 'FontSize', 12);
title('Solution Accuracy vs Problem Size', 'FontSize', 14);
legend('Location', 'best', 'FontSize', 10);
grid on;
hold off;

sgtitle('Experiment 2: Scaling Behavior', 'FontSize', 16);
saveas(fig2, 'figures/exp2_scaling_behavior.png');
fprintf('\nFigure saved: figures/exp2_scaling_behavior.png\n');

%% ========================================================================
%  EXPERIMENT 3: Nonlinear Objectives
%  "Evaluate all three methods on quadratic, cubic, and minimax costs"
%  ========================================================================
fprintf('\n\n========================================\n');
fprintf('EXPERIMENT 3: Nonlinear Objectives\n');
fprintf('========================================\n');
fprintf('All three methods on quadratic, cubic, minimax\n\n');

% Use 5-muscle problem
prob3 = n_muscles(5, 0, 'hard');
opt3 = MuscleOptimizer(prob3, struct('tol', 1e-8, 'verbose', false));
f0 = 0.5 * prob3.f_max;

fprintf('Problem: %d muscles, tau_d = %.2f N-m\n\n', prob3.n, prob3.tau_d);

objectives = {'quadratic', 'cubic'};  % minimax handled separately
n_obj = length(objectives);

% Storage
exp3_results = struct();

for j = 1:n_obj
    obj_type = objectives{j};
    fprintf('\n--- %s objective: J = sum(f^%d) ---\n', obj_type, j+1);
    
    % QP (only for quadratic)
    if strcmp(obj_type, 'quadratic')
        tic;
        [f_qp, fval_qp] = prob3.solve_quadratic();
        time_qp = toc;
        exp3_results.(obj_type).qp.f = f_qp;
        exp3_results.(obj_type).qp.objective = fval_qp;
        exp3_results.(obj_type).qp.time = time_qp;
        exp3_results.(obj_type).qp.available = true;
        fprintf('  QP:     J = %10.2f, time = %.2f ms\n', fval_qp, time_qp*1000);
    else
        exp3_results.(obj_type).qp.available = false;
        fprintf('  QP:     N/A (only for quadratic)\n');
    end
    
    % SQP (fmincon)
    tic;
    [f_sqp, fval_sqp, ~, output_sqp] = prob3.solve_fmincon(obj_type, f0);
    time_sqp = toc;
    exp3_results.(obj_type).sqp.f = f_sqp;
    exp3_results.(obj_type).sqp.objective = fval_sqp;
    exp3_results.(obj_type).sqp.time = time_sqp;
    exp3_results.(obj_type).sqp.iters = output_sqp.iterations;
    exp3_results.(obj_type).sqp.fevals = output_sqp.funcCount;
    fprintf('  SQP:    J = %10.2f, iters = %d, feval = %d, time = %.2f ms\n', ...
            fval_sqp, output_sqp.iterations, output_sqp.funcCount, time_sqp*1000);
    
    % Newton-Lagrange (our implementation)
    tic;
    [f_newton, ~, hist_newton] = opt3.solve_newton_lagrange_activeset(f0, obj_type);
    time_newton = toc;
    [J_func, ~, ~] = opt3.get_objective_functions(obj_type);
    fval_newton = J_func(f_newton);
    exp3_results.(obj_type).newton.f = f_newton;
    exp3_results.(obj_type).newton.objective = fval_newton;
    exp3_results.(obj_type).newton.time = time_newton;
    exp3_results.(obj_type).newton.iters = hist_newton.iterations;
    exp3_results.(obj_type).newton.fevals = hist_newton.func_evals;
    exp3_results.(obj_type).newton.history = hist_newton;
    fprintf('  Newton: J = %10.2f, iters = %d, feval = %d, time = %.2f ms\n', ...
            fval_newton, hist_newton.iterations, hist_newton.func_evals, time_newton*1000);
end

% Minimax (special handling - reformulated)
fprintf('\n--- minimax objective: J = max(f_i) ---\n');
tic;
[f_minimax_sqp, fval_minimax] = prob3.solve_fmincon('minimax', f0);
time_minimax = toc;
exp3_results.minimax.sqp.f = f_minimax_sqp;
exp3_results.minimax.sqp.objective = fval_minimax;
exp3_results.minimax.sqp.time = time_minimax;
fprintf('  SQP:    J = %10.2f (max force), time = %.2f ms\n', fval_minimax, time_minimax*1000);
fprintf('  Newton: N/A (minimax is non-smooth, requires reformulation)\n');
exp3_results.minimax.newton.available = false;
exp3_results.minimax.qp.available = false;

% Get force solutions for plotting
f_quad_qp = exp3_results.quadratic.qp.f;
f_quad_newton = exp3_results.quadratic.newton.f;
f_cubic_sqp = exp3_results.cubic.sqp.f;
f_cubic_newton = exp3_results.cubic.newton.f;
f_minimax_sqp = exp3_results.minimax.sqp.f;

% Plot Experiment 3
fig3 = figure('Position', [100, 100, 1400, 500]);

% Panel 1: Force distributions for quadratic
subplot(1, 3, 1);
x_pos = 1:5;
width = 0.25;
bar(x_pos - width, f_quad_qp, width, 'DisplayName', 'QP');
hold on;
bar(x_pos, f_quad_newton, width, 'DisplayName', 'Newton');
bar(x_pos + width, exp3_results.quadratic.sqp.f, width, 'DisplayName', 'SQP');
plot(x_pos, prob3.f_max, 'k--', 'LineWidth', 2, 'DisplayName', 'f_{max}');
xlabel('Muscle Index', 'FontSize', 12);
ylabel('Force (N)', 'FontSize', 12);
title('Quadratic Objective: sum(f^2)', 'FontSize', 14);
legend('Location', 'best', 'FontSize', 9);
grid on;
hold off;

% Panel 2: Force distributions for cubic
subplot(1, 3, 2);
bar(x_pos - width/2, f_cubic_sqp, width, 'DisplayName', 'SQP');
hold on;
bar(x_pos + width/2, f_cubic_newton, width, 'DisplayName', 'Newton');
plot(x_pos, prob3.f_max, 'k--', 'LineWidth', 2, 'DisplayName', 'f_{max}');
xlabel('Muscle Index', 'FontSize', 12);
ylabel('Force (N)', 'FontSize', 12);
title('Cubic Objective: sum(f^3)', 'FontSize', 14);
legend('Location', 'best', 'FontSize', 9);
grid on;
hold off;

% Panel 3: Compare all solutions
subplot(1, 3, 3);
bar(x_pos - width, f_quad_qp, width, 'DisplayName', 'Quadratic');
hold on;
bar(x_pos, f_cubic_sqp, width, 'DisplayName', 'Cubic');
bar(x_pos + width, f_minimax_sqp, width, 'DisplayName', 'Minimax');
plot(x_pos, prob3.f_max, 'k--', 'LineWidth', 2, 'DisplayName', 'f_{max}');
xlabel('Muscle Index', 'FontSize', 12);
ylabel('Force (N)', 'FontSize', 12);
title('Comparison: Different Objectives', 'FontSize', 14);
legend('Location', 'best', 'FontSize', 9);
grid on;
hold off;

sgtitle('Experiment 3: Nonlinear Objectives (5 muscles)', 'FontSize', 16);
saveas(fig3, 'figures/exp3_nonlinear_objectives.png');
fprintf('\nFigure saved: figures/exp3_nonlinear_objectives.png\n');

%% ========================================================================
%  EXPERIMENT 4: Conditioning Analysis
%  ========================================================================
fprintf('\n\n========================================\n');
fprintf('EXPERIMENT 4: Conditioning Analysis\n');
fprintf('========================================\n');
fprintf('Varying moment arm ratios r2/r1\n\n');

r2_ratios = [1, 1.5, 2, 3, 5, 7, 10, 15, 20];
n_ratios = length(r2_ratios);

exp4_cond = zeros(n_ratios, 1);
exp4_qp_time = zeros(n_ratios, 1);
exp4_sqp_iters = zeros(n_ratios, 1);
exp4_sqp_time = zeros(n_ratios, 1);
exp4_newton_iters = zeros(n_ratios, 1);
exp4_newton_time = zeros(n_ratios, 1);
exp4_accuracy_sqp = zeros(n_ratios, 1);
exp4_accuracy_newton = zeros(n_ratios, 1);

fprintf('%-8s | %-10s | %-15s | %-20s | %-20s\n', ...
        'r2/r1', 'cond(KKT)', 'QP', 'SQP', 'Newton-Lagrange');
fprintf('%s\n', repmat('-', 1, 85));

for i = 1:n_ratios
    ratio = r2_ratios(i);
    
    % Create ill-conditioned problem
    prob = n_muscles.create_ill_conditioned(ratio);
    opt = MuscleOptimizer(prob, struct('tol', 1e-10, 'verbose', false));
    
    % Condition number of KKT matrix
    n = prob.n;
    KKT = [2*eye(n), prob.r; prob.r', 0];
    exp4_cond(i) = cond(KKT);
    
    f0 = 0.5 * prob.f_max;
    
    % QP (reference)
    tic;
    [f_qp, ~] = prob.solve_quadratic();
    exp4_qp_time(i) = toc;
    
    % SQP
    tic;
    [f_sqp, ~, ~, output_sqp] = prob.solve_fmincon('quadratic', f0);
    exp4_sqp_time(i) = toc;
    exp4_sqp_iters(i) = output_sqp.iterations;
    exp4_accuracy_sqp(i) = norm(f_sqp - f_qp);
    
    % Newton-Lagrange
    tic;
    [f_newton, ~, hist_newton] = opt.solve_newton_lagrange_activeset(f0, 'quadratic');
    exp4_newton_time(i) = toc;
    exp4_newton_iters(i) = hist_newton.iterations;
    exp4_accuracy_newton(i) = norm(f_newton - f_qp);
    
    fprintf('%-8.1f | %10.2e | %7.2f ms %5s | %5d iters %7.2f ms | %5d iters %7.2f ms  err=%.1e\n', ...
            ratio, exp4_cond(i), exp4_qp_time(i)*1000, '', ...
            exp4_sqp_iters(i), exp4_sqp_time(i)*1000, ...
            exp4_newton_iters(i), exp4_newton_time(i)*1000, exp4_accuracy_newton(i));
end

% Plot Experiment 4
fig4 = figure('Position', [100, 100, 1200, 500]);

% Panel 1: Iterations vs conditioning
subplot(1, 2, 1);
semilogx(exp4_cond, exp4_sqp_iters, 's-', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'SQP');
hold on;
semilogx(exp4_cond, exp4_newton_iters, '^-', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'Newton-Lagrange');
xlabel('Condition Number of KKT Matrix', 'FontSize', 12);
ylabel('Iterations', 'FontSize', 12);
title('Convergence vs Problem Conditioning', 'FontSize', 14);
legend('Location', 'best', 'FontSize', 10);
grid on;
hold off;

% Panel 2: Accuracy vs conditioning
subplot(1, 2, 2);
loglog(exp4_cond, exp4_accuracy_sqp, 's-', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'SQP');
hold on;
loglog(exp4_cond, exp4_accuracy_newton, '^-', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'Newton-Lagrange');
xlabel('Condition Number of KKT Matrix', 'FontSize', 12);
ylabel('||f - f_{QP}||', 'FontSize', 12);
title('Solution Accuracy vs Conditioning', 'FontSize', 14);
legend('Location', 'best', 'FontSize', 10);
grid on;
hold off;

sgtitle('Experiment 4: Sensitivity to Problem Conditioning', 'FontSize', 16);
saveas(fig4, 'figures/exp4_conditioning_analysis.png');
fprintf('\nFigure saved: figures/exp4_conditioning_analysis.png\n');

%% ========================================================================
%  SUMMARY FIGURE
%  ========================================================================
fig5 = figure('Position', [100, 100, 1500, 900]);

% (a) Exp 1: Iterations
subplot(2, 3, 1);
bar([exp1_sqp_iters, exp1_newton_iters]);
set(gca, 'XTickLabel', {'Low', 'Mid', 'High', 'Asym1', 'Asym2', 'VLow', 'Near', 'Above'});
xtickangle(45);
ylabel('Iterations');
title('(a) Exp 1: Convergence (2-muscle)');
legend('SQP', 'Newton', 'Location', 'best', 'FontSize', 8);
grid on;

% (b) Exp 2: Scaling time
subplot(2, 3, 2);
semilogy(n_values, exp2_qp_time*1000, 'o-', 'LineWidth', 2, 'DisplayName', 'QP');
hold on;
semilogy(n_values, exp2_sqp_time*1000, 's-', 'LineWidth', 2, 'DisplayName', 'SQP');
semilogy(n_values, exp2_newton_time*1000, '^-', 'LineWidth', 2, 'DisplayName', 'Newton');
xlabel('Muscles (n)');
ylabel('Time (ms)');
title('(b) Exp 2: Scaling');
legend('Location', 'best', 'FontSize', 8);
grid on;
hold off;

% (c) Exp 2: Iterations
subplot(2, 3, 3);
plot(n_values, exp2_sqp_iters, 's-', 'LineWidth', 2, 'DisplayName', 'SQP');
hold on;
plot(n_values, exp2_newton_iters, '^-', 'LineWidth', 2, 'DisplayName', 'Newton');
xlabel('Muscles (n)');
ylabel('Iterations');
title('(c) Exp 2: Iterations vs n');
legend('Location', 'best', 'FontSize', 8);
grid on;
hold off;

% (d) Exp 3: Force distributions
subplot(2, 3, 4);
x_pos = 1:5;
width = 0.25;
bar(x_pos - width, f_quad_qp, width, 'DisplayName', 'Quadratic');
hold on;
bar(x_pos, f_cubic_sqp, width, 'DisplayName', 'Cubic');
bar(x_pos + width, f_minimax_sqp, width, 'DisplayName', 'Minimax');
xlabel('Muscle');
ylabel('Force (N)');
title('(d) Exp 3: Different Objectives');
legend('Location', 'best', 'FontSize', 8);
grid on;
hold off;

% (e) Exp 4: Conditioning
subplot(2, 3, 5);
semilogx(exp4_cond, exp4_sqp_iters, 's-', 'LineWidth', 2, 'DisplayName', 'SQP');
hold on;
semilogx(exp4_cond, exp4_newton_iters, '^-', 'LineWidth', 2, 'DisplayName', 'Newton');
xlabel('Condition Number');
ylabel('Iterations');
title('(e) Exp 4: Conditioning');
legend('Location', 'best', 'FontSize', 8);
grid on;
hold off;

% (f) Summary table
subplot(2, 3, 6);
axis off;
text(0.1, 0.95, 'Summary: Method Comparison', 'FontSize', 12, 'FontWeight', 'bold');
text(0.1, 0.80, sprintf('Exp 1 (2-muscle, avg iters): SQP=%.1f, Newton=%.1f', ...
     mean(exp1_sqp_iters), mean(exp1_newton_iters)), 'FontSize', 10);
text(0.1, 0.65, sprintf('Exp 2 (n=20): QP=%.1fms, SQP=%.1fms, Newton=%.1fms', ...
     exp2_qp_time(end)*1000, exp2_sqp_time(end)*1000, exp2_newton_time(end)*1000), 'FontSize', 10);
text(0.1, 0.50, sprintf('Exp 3: Newton matches SQP on quad/cubic'), 'FontSize', 10);
text(0.1, 0.35, sprintf('Exp 4: Newton stable across cond %.0e to %.0e', ...
     min(exp4_cond), max(exp4_cond)), 'FontSize', 10);
text(0.1, 0.15, 'QP fastest but only for quadratic objective', 'FontSize', 10);

sgtitle('AM205 Project: QP vs SQP vs Newton-Lagrange Comparison', 'FontSize', 14, 'FontWeight', 'bold');
saveas(fig5, 'figures/summary_all_experiments.png');
fprintf('\nSummary figure saved: figures/summary_all_experiments.png\n');

%% Comparison Table
prob = n_muscles(10, 0, 'hard');
opt = MuscleOptimizer(prob, struct('tol', 1e-8, 'verbose', false));
results = opt.compare_all_methods(0.5*prob.f_max, 'quadratic');