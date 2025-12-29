function valueIteration(mdp, theta = 0.01, gamma = 0.9) {
  if (!mdp || !mdp.states || !mdp.actions || !mdp.transitions) {
    throw new Error('invalid MDP');
  }

  const values = {};
  const policy = {};
  const steps = [];
  // initialize all state values equall zero
  mdp.states.forEach(state => {
    values[state.id] = 0;
  });

  steps.push({
    iteration: 0,
    values: { ...values },
    policy: { ...policy },
    delta: 0
  });

  let iterations = 0;
  let delta;

  do {
    delta = 0;
    iterations++;

    for (const state of mdp.states) {
      if (state.isTerminal) continue;

      const v = values[state.id];
      let maxValue = -Infinity;
      let bestAction = '';

      for (const action of mdp.actions) {
        let actionValue = 0;
        // get all possible transitions for (S,a)
        const transitions = mdp.transitions.filter(
          t => t.fromState === state.id && t.action === action.id
        );

        if (transitions.length === 0) continue;
         // expected return for this action
        for (const transition of transitions) {
          actionValue += transition.probability *
            (transition.reward + gamma * values[transition.toState]);
        }

        if (actionValue > maxValue) {
          maxValue = actionValue;
          bestAction = action.id;
        }
      }

      if (maxValue > -Infinity) {
        values[state.id] = maxValue;
        policy[state.id] = bestAction;
        // check max value change for convergence
        delta = Math.max(delta, Math.abs(v - values[state.id]));
      }
    }

    steps.push({
      iteration: iterations,
      values: { ...values },
      policy: { ...policy },
      delta: delta
    });
    
  } while (delta > theta && iterations < 1000);

  return {
    values,
    policy,
    iterations,
    steps,
    algorithm: 'value-iteration'
  };
}


function computeValueFunction(policy, mdp, gamma, theta = 0.01) {
  const values = {};
  mdp.states.forEach(state => {
    values[state.id] = 0;
  });

  const steps = [];
  let iterations = 0;

  steps.push({
    iteration: 0,
    values: { ...values },
    policy: { ...policy },
    delta: 0,
    phase: 'evaluation'
  });

  while (true) {
    const oldValues = { ...values };
    let delta = 0;
    iterations++;

    for (const state of mdp.states) {
      if (state.isTerminal) continue;

      const action = policy[state.id];
      let actionValue = 0;

      const transitions = mdp.transitions.filter(
        t => t.fromState === state.id && t.action === action
      );
       // bellman expectation equation
      for (const transition of transitions) {
        actionValue += transition.probability *
          (transition.reward + gamma * values[transition.toState]);
      }

      values[state.id] = actionValue;
      delta = Math.max(delta, Math.abs(oldValues[state.id] - values[state.id]));
    }

    steps.push({
      iteration: iterations,
      values: { ...values },
      policy: { ...policy },
      delta: delta,
      phase: 'evaluation'
    });

    if (delta < theta || iterations >= 100) break;
  }

  return { values, iterations, steps };
}
// policy improvement to extract greedy policy from value function
function extractPolicy(mdp, values, gamma) {
  const policy = {};
  let delta = 0;

  for (const state of mdp.states) {
    if (state.isTerminal) {
      policy[state.id] = null;
      continue;
    }

    let bestAction = null;
    let bestValue = -Infinity;

    for (const action of mdp.actions) {
      let actionValue = 0;

      const transitions = mdp.transitions.filter(
        t => t.fromState === state.id && t.action === action.id
      );

      if (transitions.length === 0) continue;
      // choose action that maximize expected return
      for (const transition of transitions) {
        actionValue += transition.probability *
          (transition.reward + gamma * values[transition.toState]);
      }

      if (actionValue > bestValue) {
        bestValue = actionValue;
        bestAction = action.id;
      }
    }

    if (bestAction !== null) {
      policy[state.id] = bestAction;
      delta = Math.max(delta, Math.abs(bestValue - (values[state.id] || 0)));
    }
  }

  return { policy, delta };
}

function policyIteration(mdp, theta = 0.01, gamma = 0.9, max_iter = 1000) {
  if (!mdp || !mdp.states || !mdp.actions || !mdp.transitions) {
    throw new Error('invalid MDP');
  }

  // initialize random policy
  const policy = {};
  mdp.states.forEach(state => {
    if (!state.isTerminal) {
      const randomAction = mdp.actions[Math.floor(Math.random() * mdp.actions.length)].id;
      policy[state.id] = randomAction;
    } else {
      policy[state.id] = null;
    }
  });

  const allSteps = [];
  let iterationCount = 0;
  let policyStable = false;

  // add initial step
  allSteps.push({
    iteration: 0,
    values: {},
    policy: { ...policy },
    delta: 0,
    phase: 'initial'
  });

  while (!policyStable && iterationCount < max_iter) {
    iterationCount++;
    
    // policy evaluation
    const evalResult = computeValueFunction(policy, mdp, gamma, theta);
    const values = evalResult.values;
    
    // add evaluation steps
    allSteps.push(...evalResult.steps.map(step => ({
      ...step,
      iteration: iterationCount,
      phase: 'evaluation'
    })));
    
    // policy improvement
    const improvementResult = extractPolicy(mdp, values, gamma);
    const newPolicy = improvementResult.policy;
    
    // add improvement step
    allSteps.push({
      iteration: iterationCount,
      values: { ...values },
      policy: { ...newPolicy },
      delta: improvementResult.delta,
      phase: 'improvement'
    });
    
    // Check if policy changed
    policyStable = true;
    for (const stateId in policy) {
      if (policy[stateId] !== newPolicy[stateId]) {
        policyStable = false;
        break;
      }
    }
    
    // update policy
    Object.assign(policy, newPolicy);
  }
  
  // final policy evaluation 
  const finalEval = computeValueFunction(policy, mdp, gamma, theta / 10);
  const finalValues = finalEval.values;
  
  // add final evaluation steps
  allSteps.push(...finalEval.steps.map(step => ({
    ...step,
    iteration: iterationCount + 1,
    phase: 'final-evaluation'
  })));
  
  return {
    values: finalValues,
    policy,
    iterations: iterationCount,
    steps: allSteps,
    algorithm: 'policy-iteration'
  };
}

// select to explore or exploite for epsilon greedy action
function get_action(Q, stateId, epsilon) {
  const actions = Object.keys(Q[stateId]);
  
  // random action if explore
  if (Math.random() < epsilon) {
    const randomIndex = Math.floor(Math.random() * actions.length);
    return actions[randomIndex];
  }
  
  // action with max q-value if exploite
  let bestAction = null;
  let bestValue = -Infinity;
  
  for (const actionId of actions) {
    if (Q[stateId][actionId] > bestValue) {
      bestValue = Q[stateId][actionId];
      bestAction = actionId;
    }
  }
  
  return bestAction || actions[0];
}
// sample next state based on transition probabilities
function get_next_state(mdp, stateId, actionId) {
  const transitions = mdp.transitions.filter(
    t => t.fromState === stateId && t.action === actionId
  );

  if (transitions.length === 0) {
    return stateId; 
  }

  // select transition based on probability
  const rand = Math.random();
  let cumulativeProb = 0;
  let chosenTransition = null;

  for (const transition of transitions) {
    cumulativeProb += transition.probability;
    if (rand <= cumulativeProb) {
      chosenTransition = transition;
      break;
    }
  }

  if (!chosenTransition) {
    chosenTransition = transitions[0];
  }

  return chosenTransition.toState;
}

function get_reward(mdp, stateId, actionId, nextStateId) {
  const transition = mdp.transitions.find(
    t => t.fromState === stateId && t.action === actionId && t.toState === nextStateId
  );

  if (transition) {
    return transition.reward;
  }
  return 0;
}


function generate_episode(mdp, Q, epsilon, max_steps = 1000) {
  const episode = [];
  
  let currentState = mdp.states.find(s => s.isStart) || mdp.states[0];
  let currentStateId = currentState.id;
  let steps = 0;
  
  while (!currentState.isTerminal && steps < max_steps) {
    const action = get_action(Q, currentStateId, epsilon);
    
    const nextStateId = get_next_state(mdp, currentStateId, action);
    const reward = get_reward(mdp, currentStateId, action, nextStateId);
    
    episode.push({
      state: currentStateId,
      action: action,
      reward: reward,
      nextState: nextStateId
    });
    
    currentStateId = nextStateId;
    currentState = mdp.states.find(s => s.id === nextStateId) || currentState;
    steps++;
    
    if (currentState.isTerminal) {
      break;
    }
  }
  
  return episode;
}

function extractCurrentValuesAndPolicy(Q, mdp) {
  const stateIds = mdp.states.map(s => s.id);
  const actionIds = mdp.actions.map(a => a.id);
  
  const currentValues = {};
  const currentPolicy = {};
  
  stateIds.forEach(stateId => {
    const stateObj = mdp.states.find(s => s.id === stateId);
    
    if (stateObj && (stateObj.isTerminal || stateObj.isGoal || stateObj.isHole)) {
      currentValues[stateId] = 0;
      currentPolicy[stateId] = null;
      return;
    }

    let bestAction = null;
    let bestValue = -Infinity;
    
    for (const actionId of actionIds) {
      if (Q[stateId][actionId] > bestValue) {
        bestValue = Q[stateId][actionId];
        bestAction = actionId;
      }
    }

    currentValues[stateId] = bestValue;
    currentPolicy[stateId] = bestAction;
  });
  
  return { values: currentValues, policy: currentPolicy };
}

function sarsa(mdp, episodes = 1000, gamma = 0.95, alpha = 0.1, epsilon = 0.1, epsilon_decay = 0.999, min_epsilon = 0.01) {
  if (!mdp || !mdp.states || !mdp.actions || !mdp.transitions) {
    throw new Error('invalid MDP');
  }

  // initialize Q-table
  const Q = {};
  const stateIds = mdp.states.map(s => s.id);
  const actionIds = mdp.actions.map(a => a.id);
  
  stateIds.forEach(stateId => {
    Q[stateId] = {};
    actionIds.forEach(actionId => {
      Q[stateId][actionId] = Math.random() * 0.1;
    });
  });

  const steps = [];
  let successfulEpisodes = 0;
  let totalReward = 0;
  let episodeRewards = [];

  let current_epsilon = epsilon;

  // initial step with extracted values/policy
  const initialExtract = extractCurrentValuesAndPolicy(Q, mdp);
  steps.push({
    iteration: 0,
    Q: JSON.parse(JSON.stringify(Q)),
    values: initialExtract.values,
    policy: initialExtract.policy,
    epsilon: current_epsilon,
    successfulEpisodes: 0,
    avgReward: 0,
    algorithm: 'sarsa'
  });

  for (let episode = 1; episode <= episodes; episode++) {
    let currentState = mdp.states.find(s => s.isStart) || mdp.states[0];
    let currentStateId = currentState.id;
    
    let actionId = get_action(Q, currentStateId, current_epsilon);
    
    let episodeReward = 0;
    let stepCount = 0;
    let done = false;

    while (!done && stepCount < 1000) {
      const nextStateId = get_next_state(mdp, currentStateId, actionId);
      const reward = get_reward(mdp, currentStateId, actionId, nextStateId);
      
      episodeReward += reward;

      const nextStateObj = mdp.states.find(s => s.id === nextStateId);
      if (nextStateObj && (nextStateObj.isTerminal || nextStateObj.isGoal || nextStateObj.isHole)) {
        done = true;
        
        if (nextStateObj.isGoal) {
          successfulEpisodes++;
        }
      }

      const nextActionId = get_action(Q, nextStateId, current_epsilon);

      // SARSA update using next action from same policy
      const oldQ = Q[currentStateId][actionId];
      let nextQ;
      if (done) {
           nextQ = 0;
      } else {
          nextQ = Q[nextStateId][nextActionId];
      }
      
      Q[currentStateId][actionId] = oldQ + alpha * (reward + gamma * nextQ - oldQ);

      currentStateId = nextStateId;
      actionId = nextActionId;
      stepCount++;
    }

    // decay epsilon
    current_epsilon = Math.max(min_epsilon, current_epsilon * epsilon_decay);
    
    totalReward += episodeReward;
    episodeRewards.push(episodeReward);
    
    // calc moving average of rewards
    const window = Math.min(100, episode);
    const startIdx = Math.max(0, episode - window);
    const avgReward = episodeRewards.slice(startIdx, episode).reduce((a, b) => a + b, 0) / window;

    // record progress every 100 episodes
    if (episode % 100 === 0 || episode === episodes) {
      const currentExtract = extractCurrentValuesAndPolicy(Q, mdp);
      steps.push({
        iteration: episode,
        Q: JSON.parse(JSON.stringify(Q)),
        values: currentExtract.values,
        policy: currentExtract.policy,
        epsilon: current_epsilon,
        successfulEpisodes: successfulEpisodes,
        avgReward: avgReward,
        episodeReward: episodeReward,
        algorithm: 'sarsa'
      });
    }
  }

  // extract final policy and values from Q-table
  const finalExtract = extractCurrentValuesAndPolicy(Q, mdp);
  const values = finalExtract.values;
  const policy = finalExtract.policy;

  const successRate = (successfulEpisodes / episodes * 100).toFixed(1);
  console.log(`SARSA: ${successfulEpisodes}/${episodes} successful episodes (${successRate}%)`);
  console.log(`Final epsilon: ${current_epsilon.toFixed(4)}`);

  return {
    values,
    policy,
    Q: Q,
    iterations: episodes,
    steps,
    successfulEpisodes,
    successRate,
    avgReward: totalReward / episodes,
    finalEpsilon: current_epsilon,
    algorithm: 'sarsa'
  };
}

function q_learning(mdp, episodes = 1000, gamma = 0.95, alpha = 0.1, epsilon = 0.1, epsilon_decay = 0.999, min_epsilon = 0.01) {
  if (!mdp || !mdp.states || !mdp.actions || !mdp.transitions) {
    throw new Error('invalid MDP');
  }

  // initialize Q-table
  const Q = {};
  const stateIds = mdp.states.map(s => s.id);
  const actionIds = mdp.actions.map(a => a.id);
  
  stateIds.forEach(stateId => {
    Q[stateId] = {};
    actionIds.forEach(actionId => {
      Q[stateId][actionId] = Math.random() * 0.1;
    });
  });

  const steps = [];
  let successfulEpisodes = 0;
  let totalReward = 0;
  let episodeRewards = [];

  let current_epsilon = epsilon;

  function get_max_q(stateId) {
    // q-learning uses max q(s',a') regardless behavior policy
    let maxQ = -Infinity;
    for (const actionId of actionIds) {
      if (Q[stateId][actionId] > maxQ) {
        maxQ = Q[stateId][actionId];
      }
    }
    return maxQ;
  }

  // initial step with extracted values/policy
  const initialExtract = extractCurrentValuesAndPolicy(Q, mdp);
  steps.push({
    iteration: 0,
    Q: JSON.parse(JSON.stringify(Q)),
    values: initialExtract.values,
    policy: initialExtract.policy,
    epsilon: current_epsilon,
    successfulEpisodes: 0,
    avgReward: 0,
    algorithm: 'q-learning'
  });

  for (let episode = 1; episode <= episodes; episode++) {
    let currentState = mdp.states.find(s => s.isStart) || mdp.states[0];
    let currentStateId = currentState.id;
    
    let episodeReward = 0;
    let stepCount = 0;
    let done = false;

    while (!done && stepCount < 1000) {
      const actionId = get_action(Q, currentStateId, current_epsilon);
      
      const nextStateId = get_next_state(mdp, currentStateId, actionId);
      const reward = get_reward(mdp, currentStateId, actionId, nextStateId);
      
      episodeReward += reward;

      const nextStateObj = mdp.states.find(s => s.id === nextStateId);
      if (nextStateObj && (nextStateObj.isTerminal || nextStateObj.isGoal || nextStateObj.isHole)) {
        done = true;
        
        if (nextStateObj.isGoal) {
          successfulEpisodes++;
        }
      }

      // q-Learning update
      const oldQ = Q[currentStateId][actionId];
      let maxNextQ;
      if (done) {
         maxNextQ = 0;
      } else {
         maxNextQ = get_max_q(nextStateId);
      }
      
      Q[currentStateId][actionId] = oldQ + alpha * (reward + gamma * maxNextQ - oldQ);

      currentStateId = nextStateId;
      stepCount++;
    }

    current_epsilon = Math.max(min_epsilon, current_epsilon * epsilon_decay);
    
    totalReward += episodeReward;
    episodeRewards.push(episodeReward);
    
    // calc moving average of rewards
    const window = Math.min(100, episode);
    const startIdx = Math.max(0, episode - window);
    const avgReward = episodeRewards.slice(startIdx, episode).reduce((a, b) => a + b, 0) / window;

    if (episode % 100 === 0 || episode === episodes) {
      const currentExtract = extractCurrentValuesAndPolicy(Q, mdp);
      steps.push({
        iteration: episode,
        Q: JSON.parse(JSON.stringify(Q)),
        values: currentExtract.values,
        policy: currentExtract.policy,
        epsilon: current_epsilon,
        successfulEpisodes: successfulEpisodes,
        avgReward: avgReward,
        episodeReward: episodeReward,
        algorithm: 'q-learning'
      });
    }
  }

  // extract final policy and values from Q-table
  const finalExtract = extractCurrentValuesAndPolicy(Q, mdp);
  const values = finalExtract.values;
  const policy = finalExtract.policy;

  const successRate = (successfulEpisodes / episodes * 100).toFixed(1);
  console.log(`Q-Learning: ${successfulEpisodes}/${episodes} successful episodes (${successRate}%)`);
  console.log(`Final epsilon: ${current_epsilon.toFixed(4)}`);

  return {
    values,
    policy,
    Q: Q,
    iterations: episodes,
    steps,
    successfulEpisodes,
    successRate,
    avgReward: totalReward / episodes,
    finalEpsilon: current_epsilon,
    algorithm: 'q-learning'
  };
}

function monte_carlo_control_first_visit(mdp, episodes = 10000, gamma = 0.95, epsilon = 0.1, epsilon_decay = 0.9999, min_epsilon = 0.01) {
  if (!mdp || !mdp.states || !mdp.actions || !mdp.transitions) {
    throw new Error('invalid MDP');
  }

  // initialize Q-table and returns
  const Q = {};
  const returns_sum = {};
  const returns_count = {};
  
  const stateIds = mdp.states.map(s => s.id);
  const actionIds = mdp.actions.map(a => a.id);
  
  stateIds.forEach(stateId => {
    Q[stateId] = {};
    returns_sum[stateId] = {};
    returns_count[stateId] = {};
    
    actionIds.forEach(actionId => {
      Q[stateId][actionId] = Math.random() * 0.1;
      returns_sum[stateId][actionId] = 0;
      returns_count[stateId][actionId] = 0;
    });
  });

  const steps = [];
  let successfulEpisodes = 0;
  let totalReward = 0;
  let episodeRewards = [];

  let current_epsilon = epsilon;

  // initial step with extracted values/policy
  const initialExtract = extractCurrentValuesAndPolicy(Q, mdp);
  steps.push({
    iteration: 0,
    Q: JSON.parse(JSON.stringify(Q)),
    values: initialExtract.values,
    policy: initialExtract.policy,
    epsilon: current_epsilon,
    successfulEpisodes: 0,
    avgReward: 0,
    algorithm: 'monte-carlo-control-first-visit'
  });

  for (let episode_num = 1; episode_num <= episodes; episode_num++) {
    // generate episode using current epsilon-greedy policy
    const episode = generate_episode(mdp, Q, current_epsilon, 1000);
    
    // track if episode reach goal
    if (episode.length > 0) {
      const lastStep = episode[episode.length - 1];
      const finalState = mdp.states.find(s => s.id === lastStep.nextState);
      if (finalState && finalState.isGoal) {
        successfulEpisodes++;
      }
    }
    
    // calc episode reward
    const episodeReward = episode.reduce((sum, step) => sum + step.reward, 0);
    totalReward += episodeReward;
    episodeRewards.push(episodeReward);
    
    const visited = new Set();
    
    // calc returns and update Q-values backwards
    let G = 0;
    for (let t = episode.length - 1; t >= 0; t--) {
      const { state, action, reward } = episode[t];
      G = gamma * G + reward;
      
      const visitKey = `${state}-${action}`;
      if (!visited.has(visitKey)) {
        visited.add(visitKey);
        
        // update returns statistics
        returns_sum[state][action] += G;
        returns_count[state][action] += 1;
        
        // update Q-value using sample average
        Q[state][action] = returns_sum[state][action] / returns_count[state][action];
      }
    }
    
    current_epsilon = Math.max(min_epsilon, current_epsilon * epsilon_decay);
    
    const window = Math.min(100, episode_num);
    const startIdx = Math.max(0, episode_num - window);
    const avgReward = episodeRewards.slice(startIdx, episode_num).reduce((a, b) => a + b, 0) / window;

    if (episode_num % 100 === 0 || episode_num === episodes) {
      // extract current values and policy from Q-table
      const currentExtract = extractCurrentValuesAndPolicy(Q, mdp);
      
      steps.push({
        iteration: episode_num,
        Q: JSON.parse(JSON.stringify(Q)),
        values: currentExtract.values,
        policy: currentExtract.policy,
        epsilon: current_epsilon,
        successfulEpisodes: successfulEpisodes,
        avgReward: avgReward,
        episodeReward: episodeReward,
        algorithm: 'monte-carlo-control-first-visit'
      });
    }
  }

  // extract final policy and values from Q-table
  const finalExtract = extractCurrentValuesAndPolicy(Q, mdp);
  const values = finalExtract.values;
  const policy = finalExtract.policy;

  const successRate = (successfulEpisodes / episodes * 100).toFixed(1);
  console.log(`Monte Carlo Control (First Visit): ${successfulEpisodes}/${episodes} successful episodes (${successRate}%)`);
  console.log(`Final epsilon: ${current_epsilon.toFixed(4)}`);

  return {
    values,
    policy,
    Q: Q,
    iterations: episodes,
    steps,
    successfulEpisodes,
    successRate,
    avgReward: totalReward / episodes,
    finalEpsilon: current_epsilon,
    algorithm: 'monte-carlo-control-first-visit'
  };
}

function monte_carlo_control_every_visit(mdp, episodes = 10000, gamma = 0.95, epsilon = 0.1, epsilon_decay = 0.9999, min_epsilon = 0.01) {
  if (!mdp || !mdp.states || !mdp.actions || !mdp.transitions) {
    throw new Error('invalid MDP');
  }

  const Q = {};
  const returns_sum = {};
  const returns_count = {};
  
  const stateIds = mdp.states.map(s => s.id);
  const actionIds = mdp.actions.map(a => a.id);
  
  stateIds.forEach(stateId => {
    Q[stateId] = {};
    returns_sum[stateId] = {};
    returns_count[stateId] = {};
    
    actionIds.forEach(actionId => {
      Q[stateId][actionId] = Math.random() * 0.1;
      returns_sum[stateId][actionId] = 0;
      returns_count[stateId][actionId] = 0;
    });
  });

  const steps = [];
  let successfulEpisodes = 0;
  let totalReward = 0;
  let episodeRewards = [];

  let current_epsilon = epsilon;

  // initial step with extracted values/policy
  const initialExtract = extractCurrentValuesAndPolicy(Q, mdp);
  steps.push({
    iteration: 0,
    Q: JSON.parse(JSON.stringify(Q)),
    values: initialExtract.values,
    policy: initialExtract.policy,
    epsilon: current_epsilon,
    successfulEpisodes: 0,
    avgReward: 0,
    algorithm: 'monte-carlo-control-every-visit'
  });

  for (let episode_num = 1; episode_num <= episodes; episode_num++) {
    const episode = generate_episode(mdp, Q, current_epsilon, 1000);
    
    if (episode.length > 0) {
      const lastStep = episode[episode.length - 1];
      const finalState = mdp.states.find(s => s.id === lastStep.nextState);
      if (finalState && finalState.isGoal) {
        successfulEpisodes++;
      }
    }
    
    // calc episode reward
    const episodeReward = episode.reduce((sum, step) => sum + step.reward, 0);
    totalReward += episodeReward;
    episodeRewards.push(episodeReward);
    
    // calc returns and update Q-values for every visit backwards
    let G = 0;
    for (let t = episode.length - 1; t >= 0; t--) {
      const { state, action, reward } = episode[t];
      G = gamma * G + reward;
      
      returns_sum[state][action] += G;
      returns_count[state][action] += 1;
      
      // update Q-value using sample average
      Q[state][action] = returns_sum[state][action] / returns_count[state][action];
    }
    
    current_epsilon = Math.max(min_epsilon, current_epsilon * epsilon_decay);
    
    const window = Math.min(100, episode_num);
    const startIdx = Math.max(0, episode_num - window);
    const avgReward = episodeRewards.slice(startIdx, episode_num).reduce((a, b) => a + b, 0) / window;

    if (episode_num % 100 === 0 || episode_num === episodes) {
      // extract current values and policy from Q-table
      const currentExtract = extractCurrentValuesAndPolicy(Q, mdp);
      
      steps.push({
        iteration: episode_num,
        Q: JSON.parse(JSON.stringify(Q)),
        values: currentExtract.values,
        policy: currentExtract.policy,
        epsilon: current_epsilon,
        successfulEpisodes: successfulEpisodes,
        avgReward: avgReward,
        episodeReward: episodeReward,
        algorithm: 'monte-carlo-control-every-visit'
      });
    }
  }

  // extract final policy and values from Q-table
  const finalExtract = extractCurrentValuesAndPolicy(Q, mdp);
  const values = finalExtract.values;
  const policy = finalExtract.policy;

  const successRate = (successfulEpisodes / episodes * 100).toFixed(1);
  console.log(`Monte Carlo Control (Every Visit): ${successfulEpisodes}/${episodes} successful episodes (${successRate}%)`);
  console.log(`Final epsilon: ${current_epsilon.toFixed(4)}`);

  return {
    values,
    policy,
    Q: Q,
    iterations: episodes,
    steps,
    successfulEpisodes,
    successRate,
    avgReward: totalReward / episodes,
    finalEpsilon: current_epsilon,
    algorithm: 'monte-carlo-control-every-visit'
  };
}

window.valueIteration = valueIteration;
window.policyIteration = policyIteration;
window.sarsa = sarsa;
window.q_learning = q_learning;
window.monte_carlo_control_first_visit = monte_carlo_control_first_visit;
window.monte_carlo_control_every_visit = monte_carlo_control_every_visit;