from taxicode import index as taxicab
import numpy as np 

def create_matries(P):
    # hardcoded for taxicab problem.
    num_states = 500
    num_actions = 6
    init_array_t = []
    init_array_r = []
    for x in range(0, num_actions):
        init_array_t.append([])
        # init_array_r.append(np.zeros(num_states))
        for y in range(0, num_states):
            init_array_t[x].append(np.zeros(num_states, dtype=int))
    
    for y in range(0, num_states):
        init_array_r.append([])
        for x in range(0, num_actions):
            init_array_r[y].append([])

    transitions = init_array_t
    rewards =  init_array_r
    for state in P:
        for action in P[state]:
            probAndRewards = P[state][action][0]
            sPrime = probAndRewards[1]
            reward = probAndRewards[2]
            # cab will take this action with 100% certainty
            transitions[action][state][sPrime] = 1
            # set reward associated with this action/state
            rewards[state][action] = reward
    return np.array(transitions), np.array(rewards)

def create_problem():
    taxi = taxicab.TaxiEnv()
    # taxi_row, taxi_col, pass_idx, dest_idx = (taxi.decode(taxi.s))
    # taxi.P
    ## For every state there is an action
    ### For every action there is
    #### 1.0?
    #### s'
    #### updated passenger index
    #### updated destination index

    initial_state = np.zeros(500)
    initial_state[taxi.s] = 1
    transitions, rewards = create_matries(taxi.P)
    # taxi.render()
    # taxi_row, taxi_col, pass_idx, dest_idx = taxi.decode(taxi.s)

    return list(initial_state), transitions, rewards
    

# initial_state, transitions, rewards = create_problem()
