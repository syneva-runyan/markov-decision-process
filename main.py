import mdptoolbox
from mdptoolbox import example
import matplotlib.pyplot as plt
import cereal
import taxi
import numpy as np


def run_and_average(alg, transitions, reward, initial_state, discount, env_creator = None, param=0.1, param_key='iter'):
    verbose = False
    times = []
    iters = []
    opt_values = []
    for x in range(0, 1000):
        if env_creator != None:
            #print('creating')
            initial_state, transitions, reward = env_creator()
        if x == 999:
            verbose = True
        results = alg(transitions, reward, initial_state, discount, verbose, param)
        times.append(results.time)
        iters.append(getattr(results, param_key))
        opt_values.append(results.V)
        # print("policy")
        # print(results.policy)
        # print("v")
        # print(results.V)
    avg_time = np.mean(np.array(times))
    avg_iters = np.mean(np.array(iters))
    opt_values = np.average(opt_values, axis=0)

    
    return avg_time, avg_iters, opt_values.tolist(), results


def value_iteration(transitions, reward, initial_state, discount, verbose, epsilon=0.1):
    alg = mdptoolbox.mdp.ValueIteration(epsilon=epsilon, max_iter=9000000000000000000000000000000, transitions=transitions, reward=reward, discount=discount, initial_value=initial_state)
    if verbose == True:
        alg.setVerbose()
    alg.run()
    return alg

def policy_iteration(transitions, rewards, initial_state, discount, verbose, eval_type):
    alg = mdptoolbox.mdp.PolicyIteration(max_iter=9000000000000000000000000000000, transitions=transitions, reward=rewards, discount=discount, eval_type=eval_type)
    if verbose == True:
        alg.setVerbose()
    alg.run()
    return alg

def q_learning(transitions, reward, param, discount, verbose, n_iter):
    alg = mdptoolbox.mdp.QLearning(transitions=transitions, reward=reward, discount=discount, n_iter=10000)
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
    times = []
    iters = []
    discounts = []
    values = []
    for i in range(1, 10):
        discount = (i) / 10
        discounts.append(discount)
        avg_time, avg_iters, opt_values = run_and_average(value_iteration, cereal.transitions, cereal.reward, cereal.initial_state, discount)
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
    avg_time, avg_iters, opt_values = run_and_average(value_iteration, cereal.transitions, cereal.reward_experiment, cereal.initial_state, discount)

def run_taxi_value_iteration():
    times = []
    iters = []
    tenths = []
    values = []

    # analyze various discounts
    for i in range(1, 10):
        discount = (i) / 10
        tenths.append(discount)
        avg_time, avg_iters, opt_values = run_and_average(value_iteration, None, None, None, discount, taxi.create_problem, 0.5)
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
        avg_time, avg_iters, opt_values = run_and_average(value_iteration, None, None, None, 0.8, taxi.create_problem, episilon)
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
        avg_time, avg_iters, opt_values = run_and_average(value_iteration, None, None, None, discount, taxi.create_problem, 0.9)
        times.append(avg_time)
        iters.append(avg_iters)
        values.append(opt_values)
     ## plot
    plot_avg("Taxi Cab", times, "Discount Value", "Avg Time to Run", tenths)
    plot_avg("Taxi Cab", iters, "Discount Value", "Avg Iterations", tenths)
    plot_optimal_values("Taxi Cab", values, tenths, 500, 100)
        

def run_cereal_policy_iteration():
    times = []
    iters = []
    discounts = []
    for i in range(1, 10):
        discount = (i) / 10
        discounts.append(discount)
        # eval_type iterative
        avg_time, avg_iters, opt_values = run_and_average(policy_iteration, cereal.transitions, cereal.reward, None, discount, None, 1)
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
        avg_time, avg_iters, opt_values = run_and_average(policy_iteration, cereal.transitions, cereal.reward, None, discount, None, 0)
        times.append(avg_time)
        iters.append(avg_iters)
    ## plot
    plot_avg("Cereal Loyalty: Linear Eq", times, "Discount Value", "Avg Time to Run", discounts)
    plot_avg("Cereal Loyalty: Linear Eq", iters, "Discount Value", "Avg Iterations", discounts)

def run_taxi_value_iteration():
    times = []
    iters = []
    tenths = []
    values = []

    # analyze various discounts
    for i in range(1, 10):
        discount = (i) / 10
        tenths.append(discount)
        # iterative solver
        avg_time, avg_iters, opt_values = run_and_average(policy_iteration, None, None, None, discount, taxi.create_problem, 1)
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
        tenths.append(discount)
        print("discount")
        print(discount)
        # linear equation solver
        avg_time, avg_iters, opt_values = run_and_average(policy_iteration, None, None, None, discount, taxi.create_problem, 0)
        times.append(avg_time)
        iters.append(avg_iters)
     ## plot
    plot_avg("Taxi Cab: Linear Eq", times, "Discount Value", "Avg Time to Run", tenths)
    plot_avg("Taxi Cab: Linear Eq", iters, "Discount Value", "Avg Iterations", tenths)


def run_cereal_q_learning():
    print("Starting Cereal Problem, Q Learning")
    n_iterations = 10000
    times = []
    tenths = []
    mean_discrepancy = []
    values = []
    for i in range(1, 10):
        print("discount")
        discount = (i) / 10
        print(discount)
        tenths.append(discount)
        avg_time, avg_mean_discrepancy, opt_values, results = run_and_average(q_learning, cereal.transitions, cereal.reward, None, discount, None, n_iterations, 'mean_discrepancy')
        times.append(avg_time)
        values.append(opt_values)
        mean_discrepancy.append(avg_mean_discrepancy)
        # print("Q")
        # print(alg.Q)
        # # print("V")
        # # print(alg.V)
        # print("Policy")
        # print(alg.policy)
    ## plot
    # print(mean_discrepancy)
    # print(times)
    # print(values)
    plot_avg("Cereal Loyalty: Q Learning", times, "Discount Value", "Avg Time to Run", tenths)
    plot_avg("Cereal Loyalty: Q Learning", mean_discrepancy, "Discount Value", "Mean Discrepancy", tenths)
    plot_optimal_values("Cereal Loyalty: Q Learning", values, tenths)


## Execute project code
### Value Iteration
#run_cereal_val_iteration()
#run_taxi_value_iteration()

### Policy Iteration
#run_cereal_policy_iteration()
# run_taxi_value_iteration()

### QLearning
run_cereal_q_learning()
