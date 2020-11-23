import mdptoolbox
from mdptoolbox import example
import matplotlib.pyplot as plt
import cereal
import taxi
import numpy as np


def run_and_average(alg, transitions, reward, initial_state, discount, env_creator = None, param=0.1, param_key='iter', param2=None, starting_idx=1):
    verbose = False
    times = []
    iters = []
    opt_values = []
    for x in range(starting_idx, 1000):
        print(x)
        if env_creator != None:
            #print('creating')
            initial_state, transitions, reward = env_creator()
        if x == 999:
            verbose = True
        results = alg(transitions, reward, initial_state, discount, verbose, param, param2)
        times.append(results.time)
        iters.append(getattr(results, param_key))
        opt_values.append(results.V)

    avg_time = np.mean(np.array(times))
    avg_iters = np.mean(np.array(iters))
    opt_values = np.average(opt_values, axis=0)

    
    return avg_time, avg_iters, opt_values.tolist(), results


def value_iteration(transitions, reward, initial_state, discount, verbose, epsilon=0.1, param2=None):
    alg = mdptoolbox.mdp.ValueIteration(epsilon=epsilon, max_iter=9000000000000000000000000000000, transitions=transitions, reward=reward, discount=discount, initial_value=initial_state)
    if verbose == True:
        alg.setVerbose()
    alg.run()
    return alg

def policy_iteration(transitions, rewards, initial_state, discount, verbose, eval_type, param2=None):
    alg = mdptoolbox.mdp.PolicyIteration(max_iter=9000000000000000000000000000000, transitions=transitions, reward=rewards, discount=discount, eval_type=eval_type)
    if verbose == True:
        alg.setVerbose()
    alg.run()
    return alg

def q_learning(transitions, reward, param, discount, verbose, n_iter, learning_rate=None):
    alg = mdptoolbox.mdp.QLearning(transitions=transitions, reward=reward, discount=discount, n_iter=n_iter, learning_rate=learning_rate)
    if verbose == True:
        alg.setVerbose()
    alg.run()
    return alg



def plot_avg(dataName, avgArray, xAxisName, yAxisName, xArray):
    plt.plot(xArray, avgArray)
    plt.ylabel(yAxisName)
    plt.xlabel(xAxisName)
    plt.title(dataName + ": " +  xAxisName + " vs " + yAxisName)
    plt.show()

def plot_optimal_values(dataName, values, xArray, num_states = 4, step=1, by = "Discount value"):
    fig=plt.figure()
    ax=fig.add_subplot(111)
    # print(values)
    for x in range(0, len(values)):
        val = values[x]
        ax.plot(np.arange(len(val)), val,label=by + " " + str(xArray[x]), c=np.random.rand(3,))
    
    plt.legend(loc=2)
    plt.xticks(np.arange(0, num_states, step=step))
    plt.title(dataName + ": Optimal Values by" + by)
    plt.ylabel("Value")
    plt.xlabel("State")
    plt.show()

def run_cereal_val_iteration():
    print("Starting Cereal Problem, Value Iteration")
    times = []
    iters = []
    discounts = []
    values = []
    for i in range(1, 10):
        discount = (i) / 10
        discounts.append(discount)
        avg_time, avg_iters, opt_values, results = run_and_average(value_iteration, cereal.transitions, cereal.reward, cereal.initial_state, discount)
        times.append(avg_time)
        iters.append(avg_iters)
        values.append(opt_values)
    ## plot
    plot_avg("Cereal Loyalty", times, "Discount Value", "Avg Time to Run", discounts)
    plot_avg("Cereal Loyalty", iters, "Discount Value", "Avg Iterations", discounts)
    plot_optimal_values("Cereal Loyalty", values, discounts)

    print("**********************************")
    print("Running value iteration experiment")
    print("**********************************")
    avg_time, avg_iters, opt_values, results = run_and_average(value_iteration, cereal.transitions, cereal.reward_experiment, cereal.initial_state, discount)

def run_taxi_value_iteration():
    print("Starting Taxi Problem, Value Iteration")
    times = []
    iters = []
    tenths = []
    values = []

    # analyze various discounts
    for i in range(1, 10):
        discount = (i) / 10
        tenths.append(discount)
        avg_time, avg_iters, opt_values, results = run_and_average(value_iteration, None, None, None, discount, taxi.create_problem, 0.5)
        times.append(avg_time)
        iters.append(avg_iters)
        values.append(opt_values)
     ## plot
    plot_avg("Taxi Cab", times, "Discount Value", "Avg Time to Run", tenths)
    plot_avg("Taxi Cab", iters, "Discount Value", "Avg Iterations", tenths)
    plot_optimal_values("Taxi Cab", values, tenths, 500, 100)

    # analyze various epsilon values
    times = []
    iters = []
    values = []
    for i in range(1, 10):
        episilon = (i) / 10
        avg_time, avg_iters, opt_values, results = run_and_average(value_iteration, None, None, None, 0.8, taxi.create_problem, episilon)
        times.append(avg_time)
        iters.append(avg_iters)
        values.append(opt_values)
     ## plot
    plot_avg("Taxi Cab", times, "Epsilon", "Avg Time to Run", tenths)
    plot_avg("Taxi Cab", iters, "Episilon Value", "Avg Iterations", tenths)
    plot_optimal_values("Taxi Cab", values, tenths, 500, 100, "episilon")


    # find best discount for best epsilon
    times = []
    iters = []
    values = []
    for i in range(1, 10):
        discount = (i) / 10
        avg_time, avg_iters, opt_values, results = run_and_average(value_iteration, None, None, None, discount, taxi.create_problem, 0.9)
        times.append(avg_time)
        iters.append(avg_iters)
        values.append(opt_values)
     ## plot
    plot_avg("Taxi Cab", times, "Discount Value", "Avg Time to Run", tenths)
    plot_avg("Taxi Cab", iters, "Discount Value", "Avg Iterations", tenths)
    plot_optimal_values("Taxi Cab", values, tenths, 500, 100)
        

def run_cereal_policy_iteration():
    print("Starting Cereal Problem, Policy Iteration")
    times = []
    iters = []
    discounts = []
    for i in range(1, 10):
        discount = (i) / 10
        discounts.append(discount)
        # eval_type iterative
        avg_time, avg_iters, opt_values, results = run_and_average(policy_iteration, cereal.transitions, cereal.reward, None, discount, None, 1)
        times.append(avg_time)
        iters.append(avg_iters)
    ## plot
    plot_avg("Cereal Loyalty: Iterative", times, "Discount Value", "Avg Time to Run", discounts)
    plot_avg("Cereal Loyalty: Iterative", iters, "Discount Value", "Avg Iterations", discounts)

    times = []
    iters = []
    discounts = []
    for i in range(1, 10):
        discount = (i) / 10
        discounts.append(discount)
        # eval_type linear equations
        avg_time, avg_iters, opt_values, results = run_and_average(policy_iteration, cereal.transitions, cereal.reward, None, discount, None, 0)
        times.append(avg_time)
        iters.append(avg_iters)
    ## plot
    plot_avg("Cereal Loyalty: Linear Eq", times, "Discount Value", "Avg Time to Run", discounts)
    plot_avg("Cereal Loyalty: Linear Eq", iters, "Discount Value", "Avg Iterations", discounts)

def run_taxi_policy_iteration():
    print("Starting Taxi Problem, Policy Iteration")
    times = []
    iters = []
    tenths = []
    values = []

    # analyze various discounts
    for i in range(1, 10):
        discount = (i) / 10
        tenths.append(discount)
        # iterative solver
        avg_time, avg_iters, opt_values, results = run_and_average(policy_iteration, None, None, None, discount, taxi.create_problem, 1)
        times.append(avg_time)
        iters.append(avg_iters)
     ## plot
    plot_avg("Taxi Cab", times, "Discount Value", "Avg Time to Run", tenths)
    plot_avg("Taxi Cab", iters, "Discount Value", "Avg Iterations", tenths)

    # analyze various discounts
    times = []
    iters = []
    for i in range(1, 10):
        discount = (i) / 10
        print("discount")
        print(discount)
        # linear equation solver
        avg_time, avg_iters, opt_values, results = run_and_average(policy_iteration, None, None, None, discount, taxi.create_problem, 0)
        times.append(avg_time)
        iters.append(avg_iters)
     ## plot
    plot_avg("Taxi Cab: Linear Eq", times, "Discount Value", "Avg Time to Run", tenths)
    plot_avg("Taxi Cab: Linear Eq", iters, "Discount Value", "Avg Iterations", tenths)


def run_cereal_q_learning():
    print("Starting Cereal Problem, Q Learning")
    n_iteration = 10000
    times = []
    tenths = []
    mean_discrepancy = []
    values = []
    n_iterations = []
    print("Experimenting with discount values")
    for i in range(1, 10):
        print("discount")
        discount = (i) / 10
        print(discount)
        tenths.append(discount)
        avg_time, avg_mean_discrepancy, opt_values, results = run_and_average(q_learning, cereal.transitions, cereal.reward, None, discount, None, n_iteration, 'mean_discrepancy')
        times.append(avg_time)
        values.append(opt_values)
        mean_discrepancy.append(avg_mean_discrepancy)

    # plot
    plot_avg("Cereal Loyalty: Q Learning", times, "Discount Value", "Avg Time to Run", tenths)
    plot_avg("Cereal Loyalty: Q Learning", mean_discrepancy, "Discount Value", "Mean Discrepancy", tenths)
    plot_optimal_values("Cereal Loyalty: Q Learning", values, tenths)

    discount = 0.3
    times = []
    mean_discrepancy = []
    values = []
    print("Experimenting with n_iterations")
    for i in range(10000, 100000, 10000):
        n_iteration = i
        n_iterations.append(n_iteration)
        print("n_iteration")
        print(n_iteration)
        avg_time, avg_mean_discrepancy, opt_values, results = run_and_average(q_learning, cereal.transitions, cereal.reward, None, discount, None, n_iteration, 'mean_discrepancy', None, 900)
        times.append(avg_time)
        values.append(opt_values)
        mean_discrepancy.append(avg_mean_discrepancy)
        # print("Policy")
        # print(results.policy)
        
    plot_avg("Cereal Loyalty: Q Learning", times, "N Iterations", "Avg Time to Run", n_iterations)
    plot_avg("Cereal Loyalty: Q Learning", mean_discrepancy, "N Iterations", "Mean Discrepancy", n_iterations)
    plot_optimal_values("Cereal Loyalty: Q Learning", values, n_iterations, num_states = 4, step=1, by = "Number of Iterations")

    print("Experimenting with Learning Rates")
    n_iteration = 30000
    times = []
    mean_discrepancy = []
    values = []
    for i in range(1, 10):
        print("Learning Rate")
        lr = (i) / 10
        print(lr)
        avg_time, avg_mean_discrepancy, opt_values, results = run_and_average(q_learning, cereal.transitions, cereal.reward, None, discount, None, n_iteration, 'mean_discrepancy', lr, 900)
        times.append(avg_time)
        values.append(opt_values)
        mean_discrepancy.append(avg_mean_discrepancy)

    ## plot
    plot_avg("Cereal Loyalty: Q Learning", times, "Learning Rate", "Avg Time to Run", tenths)
    plot_avg("Cereal Loyalty: Q Learning", mean_discrepancy, "Learning Rate", "Mean Discrepancy", tenths)
    plot_optimal_values("Cereal Loyalty: Q Learning", values, tenths, num_states = 4, step=1, by = "Learning Rate")

def run_taxi_q_learning():
    print("Starting Cereal Problem, Q Learning")
    n_iteration = 10000
    times = []
    tenths = []
    mean_discrepancy = []
    values = []
    print("Experimenting with discount values")
    for i in range(1, 10):
        print("discount")
        discount = (i) / 10
        print(discount)
        tenths.append(discount)
        avg_time, avg_mean_discrepancy, opt_values, results = run_and_average(q_learning, None, None, None, discount, taxi.create_problem, n_iteration, 'mean_discrepancy', None, 900)
        times.append(avg_time)
        values.append(opt_values)
        mean_discrepancy.append(avg_mean_discrepancy)
    

    # plot
    plot_avg("Taxi Cab: Q Learning", times, "Discount Value", "Avg Time to Run", tenths)
    plot_avg("Taxi Cab: Q Learning", mean_discrepancy, "Discount Value", "Mean Discrepancy", tenths)
    plot_optimal_values("Taxi Cab: Q Learning", values, tenths, 500, 100)

    print("Experimenting with n_iterations")
    times = []
    mean_discrepancy = []
    values = []
    n_iterations = []
    discount = 0.3
    for i in range(10000, 100000, 10000):
        n_iteration = i
        n_iterations.append(n_iteration)
        print("n_iteration")
        print(n_iteration)
        avg_time, avg_mean_discrepancy, opt_values, results = run_and_average(q_learning, None, None, None, discount, taxi.create_problem, n_iteration, 'mean_discrepancy', None, 900)
        times.append(avg_time)
        values.append(opt_values)
        mean_discrepancy.append(avg_mean_discrepancy)
        print("Policy")
        print(results.policy)
        
    plot_avg("Taxi Problem: Q Learning", times, "N Iterations", "Avg Time to Run", n_iterations)
    plot_avg("Taxi Problemy: Q Learning", mean_discrepancy, "N Iterations", "Mean Discrepancy", n_iterations)
    plot_optimal_values("Taxi Problem: Q Learning", values, n_iterations, num_states = 4, step=1, by = "Number of Iterations")

    print("Experimenting with Learning Rates")
    n_iteration = 30000
    times = []
    mean_discrepancy = []
    values = []
    for i in range(1, 10):
        print("Learning Rate")
        lr = (i) / 10
        print(lr)
        avg_time, avg_mean_discrepancy, opt_values, results = run_and_average(q_learning, None, None, None, discount, taxi.create_problem, n_iteration, 'mean_discrepancy', lr, 900)
        times.append(avg_time)
        values.append(opt_values)
        mean_discrepancy.append(avg_mean_discrepancy)
        # print('policy')
        # print(results.policy)

    ## plot
    plot_avg("Taxi Problem: Q Learning", times, "Learning Rate", "Avg Time to Run", tenths)
    plot_avg("Taxi Problem: Q Learning", mean_discrepancy, "Learning Rate", "Mean Discrepancy", tenths)
    plot_optimal_values("Taxi Problem: Q Learning", values, tenths, num_states = 500, step=100, by = "Learning Rate")

## Execute project code
### Value Iteration
print("Exploring value iteration")
run_cereal_val_iteration()
run_taxi_value_iteration()

### Policy Iteration
print("Exploring policy iteration")
run_cereal_policy_iteration()
run_taxi_policy_iteration()

### QLearning
print("Exploring Q Learning")
run_cereal_q_learning()
run_taxi_q_learning()
