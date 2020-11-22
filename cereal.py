import numpy as np
## transition probabilities as affected by
# Discount Scrunchy
# Scrunchy Commercial
# Free Scrunchy
# Scrunchy sponsoring a Nascar driver
transitions = np.array([
    [ # User has been offered a discount Scrunchy Cereal Coupon
     [1,0,0,0],  # User will stay with Crispy
     [0,1,0,0],  # User will stay with Crunchy
     [0.4,0.35,0,0.25],  # OG transition probs edited to show user is more likely to purchase Scrunchy
     [0.05,0.15,0.25,0.55]],  # OG transition probs probs edited to show user is more likely to purchase Scrunchy
    [ # User sees a Scrunchy commercial
     [1, 0,0,0],
     [0,1,0,0],
     [0.4,0.35,0.25,0],
     [0.05,0.15,0.5,0.3]
    ],
    [ # User has been offered a free Scrunchy Cereal
     [1, 0,0,0],
     [0,1,0,0],
     [0.35,0.3,0.35,0],
     [0,0.1,0.6,0.3]
    ],
    [ # User sees that Scrunchy sponsors a Nascar driver
     [1, 0,0,0],
     [0,1,0,0],
     [0.3,0.25,0.45,0],
     [0,0.05,0.75,0.2]
    ]
])

# Rewards for 
# User offered Discount Scrunchy
# User sees Scrunchy Commercial
# User offered Free Scrunchy
# User see Scrunchy sponsoring a Nascar driver
reward = np.array([
    [1, 1, 1 , 0], # On brand cereals taste good, but scrunchy seems like it could too! 
    [1, 1, 1 , 1], # Let's try some offbrand cereals!
    [1, 1, 2 , 0], # Who says no to free food?!
    [1, 1, 3 , 0] # Nascar - niiccee
])

# Manipulate rewards to show a world where
# the value of choosing Scrunchy cereal is always neutral.
reward_experiment = np.array([
    [1, 1, 0 , 0], # On brand cereals taste good, but scrunchy seems like it could too! 
    [1, 1, 0 , 1], # Let's try some offbrand cereals!
    [1, 1, 0 , 0], # Who says no to free food?!
    [1, 1, 0 , 0] # Nascar - niiccee
])

initial_state = [0.2, 0.3, 0.3, 0.2]  # Current cereal market shares