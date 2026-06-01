import copy
import gymnasium as gym
import numpy as np
from masa.prob_shield.helpers import build_successor_states_matrix
import types
from typing import Callable

def synthesise_shield(env: gym.Env, transition_matrix_fn: Callable, label_fn: Callable, cost_fn: Callable) -> np.ndarray:
    """
    Synthesise a safety shield for the given environment using MASA's prob_shield helper function.
    
    Args:
    - env: the environment for which to synthesise the shield. Must have a discrete state and action space, and support the interface expected by MASA's prob_shield helper function (see below).
    - transition_matrix_fn: a function that takes the environment as input and returns the transition matrix in the shape that MASA expects: (n_states, n_states, n_actions), with P[s_next, s, a] = Pr(s_next | s, a).
    - label_fn: a function that takes an observation as input and returns a set of labels for that observation.
    - cost_fn: a function that takes a set of labels as input and returns a cost (float) for that set of labels, where higher cost indicates more unsafe. The shield synthesis will aim to keep the agent in states with low cost (i.e. safe states) with high probability.
    Returns:
    - safe_state_action_pairs: a binary matrix of shape (n_states, n_actions), where safe_state_action_pairs[s, a] = 1 indicates that action a in state s is considered safe by the synthesized shield, and 0 indicates that it is not. 
      The shield can then be implemented by masking the agent's action choices with this matrix, e.g. by only allowing actions a in state s for which safe_state_action_pairs[s, a] = 1. 
    """


    ### Prepare the env API for MASA's prob_shield helper function.
    masa_env = copy.deepcopy(env)

    # Add the interface expected by MASA's prob_shield helper.
    if not hasattr(masa_env.unwrapped, 'has_transition_matrix'):
        masa_env.unwrapped.has_transition_matrix = True # type: ignore
    if not hasattr(masa_env.unwrapped, 'has_successor_states_dict'):
        masa_env.unwrapped.has_successor_states_dict = False # type: ignore
    if not hasattr(masa_env.unwrapped, 'get_transition_matrix'):
        masa_env.unwrapped.get_transition_matrix = types.MethodType( # type: ignore
            lambda self: transition_matrix_fn(self),
            masa_env.unwrapped,
        )

    # Optional but useful: MASA checks initial state feasibility if _start_state exists.
    masa_env.unwrapped._start_state = 0 # type: ignore

    ### Build the successor states matrix and winning set using MASA's helper function.
    output = build_successor_states_matrix(env=masa_env, label_fn=label_fn, cost_fn=cost_fn)

    ### Synthesize the shield from the output of the helper function.
    n_actions = int(masa_env.action_space.n) # type: ignore
    safe_state_action_pairs = np.zeros((output[0].shape[1], n_actions), dtype=int)
    W = set(output[5]) # type: ignore # winning set / safe states, starting from which we can provably stay safe with some policy
    for s in list(W):
        succs_s = output[0][:, s] # possible successor states from s which would keep us safe, i.e. in W
        for a in range(n_actions):
            p_sa = output[1][:, s, a] # transition probabilities to those successor states when taking action a in state s
            mask = (p_sa > 0.0) & (succs_s != -1)
            support = succs_s[mask]
            if all(int(sp) in W for sp in support):
                safe_state_action_pairs[s, a] = 1

    return safe_state_action_pairs