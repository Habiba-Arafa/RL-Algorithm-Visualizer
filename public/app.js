const actionArrows = { up: '↑', down: '↓', left: '←', right: '→' };
const actionNames = { up: 'Up', down: 'Down', left: 'Left', right: 'Right' };

const elements = {
  envSelect: document.getElementById('envSelect'),
  algorithmSelect: document.getElementById('algorithmSelect'),
  modeSelect: document.getElementById('modeSelect'),
  gammaInput: document.getElementById('gammaInput'),
  thetaInput: document.getElementById('thetaInput'),
  episodesInput: document.getElementById('episodesInput'),
  epsilonInput: document.getElementById('epsilonInput'),
  alphaInput: document.getElementById('alphaInput'),
  alphaLabel: document.getElementById('alphaLabel'),
  mcParams: document.getElementById('mcParams'),
  loadBtn: document.getElementById('loadBtn'),
  runBtn: document.getElementById('runBtn'),
  stepBtn: document.getElementById('stepBtn'),
  animateBtn: document.getElementById('animateBtn'),
  stopBtn: document.getElementById('stopBtn'),
  playBtn: document.getElementById('playBtn'),
  resetBtn: document.getElementById('resetBtn'),
  gridContainer: document.getElementById('gridContainer'),
  graphContainer: document.getElementById('graphContainer'),
  valuesPre: document.getElementById('valuesPre'),
  policyPre: document.getElementById('policyPre'),
  log: document.getElementById('log'),
  stepDisplay: document.getElementById('stepDisplay'),
  progressFill: document.getElementById('progressFill'),
  iterationDisplay: document.getElementById('iterationDisplay'),
  deltaDisplay: document.getElementById('deltaDisplay'),
  totalIterationsDisplay: document.getElementById('totalIterationsDisplay'),
  trainingStatus: document.getElementById('trainingStatus'),
  envName: document.getElementById('envName'),
  currentAlgorithm: document.getElementById('currentAlgorithm'),
  currentMode: document.getElementById('currentMode'),
  currentGamma: document.getElementById('currentGamma'),
  inferenceStats: document.getElementById('inferenceStats'),
  agentStep: document.getElementById('agentStep'),
  currentState: document.getElementById('currentState'),
  currentAction: document.getElementById('currentAction'),
  currentReward: document.getElementById('currentReward'),
  totalReward: document.getElementById('totalReward'),
  agentStatus: document.getElementById('agentStatus'),
  algoDesc: document.getElementById('algoDesc'),
  algoSteps: document.getElementById('algoSteps'),
  currentAlgoInfo: document.getElementById('currentAlgoInfo')
};

let currentMDP = null;
let runResult = null;
let trainedPolicy = null;
let trainedValues = null;
let stepIndex = 0;
let animHandle = null;
let inferenceHandle = null;
let agentState = null;
let agentHistory = [];
let totalAgentReward = 0;
let inferenceStep = 0;

// algorithm explanations
const algorithmExplanations = {
  'value-iteration': {
    desc: 'Iteratively improves value estimates until convergence. Guaranteed to find optimal policy.',
    steps: [
      'Initialize all state values to 0',
      'For each state, compute expected return for all actions',
      'Update state value with maximum expected return',
      'Repeat until maximum change (delta) < theta'
    ],
    info: 'Best for: Small to medium MDPs. Convergence: Guaranteed.'
  },
  'policy-iteration': {
    desc: 'Alternates between policy evaluation and policy improvement. Often converges faster than value iteration.',
    steps: [
      'Initialize random policy',
      'Policy Evaluation: Compute value function for current policy',
      'Policy Improvement: Update policy to be greedy with respect to computed values',
      'Repeat until policy no longer changes'
    ],
    info: 'Phases: Evaluation → Improvement → Repeat. Often faster convergence.'
  },
  'sarsa': {
    desc: 'On-policy Temporal Difference learning. Updates Q-values using the actual next action taken. Uses epsilon-greedy exploration.',
    steps: [
      'Initialize Q-table with small random values',
      'For each episode: Start from initial state',
      'Choose action with epsilon-greedy policy',
      'Take action, observe reward and next state',
      'Choose next action with epsilon-greedy policy',
      'Update: Q(s,a) ← Q(s,a) + α[r + γQ(s\',a\') - Q(s,a)]',
      'Repeat until convergence'
    ],
    info: 'On-policy TD control. Learns while following the policy.'
  },
  'q-learning': {
    desc: 'Off-policy Temporal Difference learning. Updates Q-values using the maximum Q-value of the next state.',
    steps: [
      'Initialize Q-table with small random values',
      'For each episode: Start from initial state',
      'Choose action with epsilon-greedy policy',
      'Take action, observe reward and next state',
      'Update: Q(s,a) ← Q(s,a) + α[r + γ max_a\' Q(s\',a\') - Q(s,a)]',
      'Repeat until convergence'
    ],
    info: 'Off-policy TD control. More aggressive updates than SARSA.'
  },
  'monte-carlo-control-first-visit': {
    desc: 'Model-free control using complete episodes. First-visit MC with epsilon-greedy exploration. Only first occurrence of each state-action in episode is used.',
    steps: [
      'Initialize Q-table and returns counters',
      'Generate episodes using epsilon-greedy policy',
      'For each state-action pair visited for the FIRST time in episode',
      'Calculate return from that point to end of episode',
      'Update Q(s,a) as average of returns from first visits',
      'Decay epsilon over time'
    ],
    info: 'First-visit MC control. Less variance, slower learning.'
  },
  'monte-carlo-control-every-visit': {
    desc: 'Model-free control using complete episodes. Every-visit MC with epsilon-greedy exploration. All occurrences of state-action in episode are used.',
    steps: [
      'Initialize Q-table and returns counters',
      'Generate episodes using epsilon-greedy policy',
      'For EACH state-action occurrence in episode',
      'Calculate return from that point to end of episode',
      'Update Q(s,a) as average of returns from all visits',
      'Decay epsilon over time'
    ],
    info: 'Every-visit MC control. More variance, faster learning.'
  }
};

// environment explanations
const envExplanations = {
  'gridworld': {
    name: 'GridWorld 3x3',
    desc: 'Simple grid with deterministic transitions. Agent must reach goal with minimum steps.',
    details: 'Start: S1 (top-left). Goal: S9 (bottom-right, +100 reward). Step penalty: -1.'
  },
  'frozenlake': {
    name: 'Frozen Lake 4x4',
    desc: 'Navigate frozen lake without falling in holes. Episodic task with terminal states.',
    details: 'Start: S0. Goal: S15 (+1). Holes: S5, S7, S11, S12 (-1). Ice: Deterministic moves.'
  }
};

function log(msg, type = 'info') {
  const d = document.createElement('div');
  d.className = 'log-entry';
  
  let prefix = 'Info: '; 
  if (type === 'error') {
    prefix = 'Error: ';
  } else if (type === 'success') {
   prefix = 'Success: ';
  } else if (type === 'warning') {
   prefix = 'Warning: ';
  }
  d.textContent = prefix + msg;
  elements.log.prepend(d);
  
  if (elements.log.children.length > 50) {
    elements.log.removeChild(elements.log.lastChild);
  }
}

function updateAlgorithmExplanation() {
  const algorithm = elements.algorithmSelect.value;
  const explanation = algorithmExplanations[algorithm];
  
  if (explanation) {
    elements.algoDesc.textContent = explanation.desc;
    elements.currentAlgoInfo.innerHTML = `<strong>${algorithm.replace(/-/g, ' ').toUpperCase()}:</strong> ${explanation.info}`;
    
    elements.algoSteps.innerHTML = '';
    explanation.steps.forEach(step => {
      const li = document.createElement('li');
      li.textContent = step;
      elements.algoSteps.appendChild(li);
    });
  }
}

function updateEnvInfo() {
  if (!currentMDP) return;
  elements.envName.textContent = currentMDP.name || 'Custom';
  elements.currentAlgorithm.textContent = elements.algorithmSelect.value;
  elements.currentMode.textContent = elements.modeSelect.value;
  elements.currentGamma.textContent = currentMDP.gamma || elements.gammaInput.value;
}

function updateTrainingStatus() {
  if (trainedPolicy) {
    elements.trainingStatus.textContent = 'Trained';
    elements.trainingStatus.style.color = '#38a169';
  } else {
    elements.trainingStatus.textContent = 'Not trained';
    elements.trainingStatus.style.color = '#e53e3e';
  }
}

function updateInferenceStats() {
  elements.agentStep.textContent = inferenceStep;
  elements.totalReward.textContent = totalAgentReward.toFixed(2);
  elements.currentReward.textContent = agentHistory.length > 0 ? 
    agentHistory[agentHistory.length - 1].reward.toFixed(2) : '0';
  
  if (agentState) {
    const stateObj = currentMDP.states.find(s => s.id === agentState);
    elements.currentState.textContent = agentState;
    elements.currentAction.textContent = agentHistory.length > 0 ? 
      actionNames[agentHistory[agentHistory.length - 1].action] || 
      agentHistory[agentHistory.length - 1].action : '-';
    
    if (stateObj) {
      if (stateObj.isTerminal) {
        elements.agentStatus.textContent = stateObj.isGoal ? 'Goal!' : 'Terminal';
        elements.agentStatus.style.color = stateObj.isGoal ? '#38a169' : '#e53e3e';
      } else {
        elements.agentStatus.textContent = 'Playing';
        elements.agentStatus.style.color = '#2b6cb0';
      }
    }
  }
}

function loadSelectedEnvironment() {
  try {
    const envType = elements.envSelect.value;
    
    if (envType === 'frozenlake') {
      currentMDP = JSON.parse(JSON.stringify(window.EXAMPLE_MDPS.frozenlake_4x4));
    } else {
      currentMDP = JSON.parse(JSON.stringify(window.EXAMPLE_MDPS.gridworld));
    }
    
    currentMDP.gamma = parseFloat(elements.gammaInput.value) || 0.9;
    
    runResult = null;
    stepIndex = 0;
    trainedPolicy = null;
    trainedValues = null;
    stopInference();
    stopAnimation();
    
    elements.stopBtn.disabled = true;
    elements.playBtn.style.display = 'none';
    elements.inferenceStats.style.display = 'none';
    
    log(`Loaded: ${currentMDP.name}`);
    log(`Mode: ${elements.modeSelect.value}, Algorithm: ${elements.algorithmSelect.value}`);
    
    renderGrid(currentMDP);
    renderGraph(currentMDP);
    renderTextViews({}, {});
    updateProgressAndStats();
    updateEnvInfo();
    updateAlgorithmExplanation();
    updateTrainingStatus();
    
  } catch (error) {
    console.error('Error:', error);
    log(`Error: ${error.message}`, 'error');
  }
}

function computeGridSize(mdp) {
  if (!mdp || !mdp.states) return { width: 3, height: 3 };
  const xs = mdp.states.map(s => s.x);
  const ys = mdp.states.map(s => s.y);
  return { width: Math.max(...xs) + 1, height: Math.max(...ys) + 1 };
}

function renderGrid(mdp, values = {}, policy = {}, activeState = null, agentPosition = null) {
  if (!mdp || !mdp.states) {
    elements.gridContainer.innerHTML = '<div>No environment</div>';
    return;
  }

  const { width, height } = computeGridSize(mdp);
  elements.gridContainer.style.gridTemplateColumns = `repeat(${width}, 1fr)`;
  elements.gridContainer.style.gridTemplateRows = `repeat(${height}, 90px)`;
  elements.gridContainer.innerHTML = '';

  const cellMap = {};
  mdp.states.forEach(s => { cellMap[`${s.x},${s.y}`] = s; });

  for (let row = 0; row < height; row++) {
    for (let col = 0; col < width; col++) {
      const cell = document.createElement('div');
      cell.className = 'cell';
      const s = cellMap[`${col},${row}`];
      if (!s) {
        cell.style.visibility = 'hidden';
        elements.gridContainer.appendChild(cell);
        continue;
      }
      
      if (s.isTerminal) cell.classList.add('terminal');
      if (s.isStart) cell.classList.add('start');
      if (s.isGoal) cell.classList.add('goal');
      if (s.isHole) cell.classList.add('hole');
      if (s.id === activeState) cell.classList.add('active');
      if (agentPosition && s.id === agentPosition) cell.classList.add('agent');

      const name = document.createElement('div');
      name.className = 'name';
      if (agentPosition && s.id === agentPosition) {
        name.textContent = '🤖';
      } else if (s.isStart) {
        name.textContent = '🏁';
      } else if (s.isGoal) {
        name.textContent = '🎯';
      } else if (s.isHole) {
        name.textContent = '🌊';
      } else {
        name.textContent = s.name;
      }

      const val = document.createElement('div');
      val.className = 'value';
      val.textContent = (values[s.id] !== undefined) ? Number(values[s.id]).toFixed(2) : '0.00';

      const act = document.createElement('div');
      act.className = 'action';
      act.textContent = policy[s.id] ? actionArrows[policy[s.id]] || policy[s.id] : '';

      cell.appendChild(name);
      cell.appendChild(val);
      cell.appendChild(act);
      elements.gridContainer.appendChild(cell);
    }
  }
}

function renderGraph(mdp, currentValues = {}, policy = {}, activeState = null, agentPosition = null) {
  const container = elements.graphContainer;
  container.innerHTML = '<h3 style="margin-top: 0;">State Transition Graph</h3>';
  
  if (!mdp || !mdp.states || mdp.states.length === 0) {
    container.innerHTML += '<div>No environment</div>';
    return;
  }

  const svg = d3.select(container).append('svg')
    .attr('width', 800)
    .attr('height', 500)
    .style('background', 'white');

  const margin = { top: 20, right: 20, bottom: 20, left: 20 };
  const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

  const gridWidth = 800 - margin.left - margin.right;
  const gridHeight = 500 - margin.top - margin.bottom;
  
  const { width, height } = computeGridSize(mdp);
  const cellSize = Math.min(gridWidth / width, gridHeight / height) * 0.9;

  mdp.states.forEach(state => {
    const value = currentValues[state.id] || 0;
    const isActive = state.id === activeState;
    const isAgent = agentPosition && state.id === agentPosition;
    const isTerminal = state.isTerminal;
    const isGoal = state.isGoal;
    const isStart = state.isStart;
    const isHole = state.isHole;
    
    let color = '#667eea';
    if (isAgent) {
      color = '#9f7aea';
    } else if (isStart) {
      color = '#f6e05e';
    } else if (isGoal) {
      color = '#9ae6b4';
    } else if (isHole) {
      color = '#fc8181';
    }
    
    const stateGroup = g.append('g')
      .attr('transform', `translate(${state.x * cellSize}, ${state.y * cellSize})`);

    const circle = stateGroup.append('circle')
      .attr('cx', cellSize/2)
      .attr('cy', cellSize/2)
      .attr('r', isAgent ? 45 : (isActive ? 42 : 40))
      .attr('fill', color)
      .attr('stroke', isAgent ? '#d6bcfa' : (isActive ? '#63b3ed' : '#fff'))
      .attr('stroke-width', isAgent ? 4 : (isActive ? 4 : 3));

    let label = state.name;
    if (isAgent) {
      label = '🤖';
    } else if (isStart) {
      label = '🏁';
    } else if (isGoal) {
      label = '🎯';
    } else if (isHole) {
      label = '🌊';
    }
    
    stateGroup.append('text')
      .attr('x', cellSize/2)
      .attr('y', cellSize/2)
      .attr('text-anchor', 'middle')
      .attr('dy', (isAgent || isStart || isGoal || isHole) ? '0.1em' : '-0.3em')
      .attr('fill', (isStart || isGoal || isHole) ? '#000' : 'white')
      .attr('font-weight', 'bold')
      .attr('font-size', (isAgent || isStart || isGoal || isHole) ? '24px' : '16px')
      .text(label);

    if (!isTerminal && !isAgent) {
      stateGroup.append('text')
        .attr('x', cellSize/2)
        .attr('y', cellSize/2)
        .attr('text-anchor', 'middle')
        .attr('dy', '1.2em')
        .attr('fill', 'white')
        .attr('font-size', '12px')
        .text(`V: ${value.toFixed(2)}`);
    }
  });
}

function renderTextViews(values, policy) {
  const formattedValues = {};
  Object.keys(values).forEach(key => {
    formattedValues[key] = Number(values[key]).toFixed(4);
  });
  elements.valuesPre.textContent = JSON.stringify(formattedValues, null, 2);
  
  const formattedPolicy = {};
  Object.keys(policy).forEach(key => {
    formattedPolicy[key] = policy[key] ? actionArrows[policy[key]] || policy[key] : 'TERMINAL';
  });
  elements.policyPre.textContent = JSON.stringify(formattedPolicy, null, 2);
}

function updateProgressAndStats() {
  if (!runResult) {
    elements.stepDisplay.textContent = 'Step 0 / 0';
    elements.progressFill.style.width = '0%';
    elements.iterationDisplay.textContent = '0';
    elements.deltaDisplay.textContent = '0.0000';
    elements.totalIterationsDisplay.textContent = '0';
    return;
  }
  
  const totalSteps = runResult.steps.length;
  const progress = totalSteps > 1 ? ((stepIndex + 1) / totalSteps) * 100 : 0;
  elements.stepDisplay.textContent = `Step ${stepIndex + 1} / ${totalSteps}`;
  elements.progressFill.style.width = `${progress}%`;
  
  const currentStep = runResult.steps[stepIndex] || { iteration: 0, delta: 0 };
  elements.iterationDisplay.textContent = currentStep.iteration || 0;
  elements.deltaDisplay.textContent = currentStep.delta ? currentStep.delta.toFixed(4) : '0.0000';
  elements.totalIterationsDisplay.textContent = runResult.iterations;
}

// inference functions

function startInference() {
  if (!currentMDP || !trainedPolicy) {
    log('Error: No trained policy found. Train first!', 'error');
    return;
  }
  
  stopInference();
  
  agentState = currentMDP.states.find(s => s.isStart)?.id || currentMDP.states[0].id;
  agentHistory = [];
  totalAgentReward = 0;
  inferenceStep = 0;
  
  elements.inferenceStats.style.display = 'block';
  elements.playBtn.style.display = 'inline-block';
  updateInferenceStats();
  
  log('Starting agent inference with trained policy');
  log(`Agent starts at: ${agentState}`);
  
  const valuesToShow = trainedValues || {};
  renderGrid(currentMDP, valuesToShow, trainedPolicy, null, agentState);
  renderGraph(currentMDP, valuesToShow, trainedPolicy, null, agentState);
}

function stepInference() {
  if (!currentMDP || !trainedPolicy || !agentState) {
    log('Error: Agent not initialized', 'error');
    return;
  }
  
  const currentStateObj = currentMDP.states.find(s => s.id === agentState);
  if (!currentStateObj || currentStateObj.isTerminal) {
    log(`Agent reached terminal state: ${agentState}`, 'success');
    return;
  }
  
  const action = trainedPolicy[agentState];
  
  if (!action) {
    log(`No policy for state ${agentState}, choosing random`, 'warning');
    const randomAction = currentMDP.actions[Math.floor(Math.random() * currentMDP.actions.length)].id;
    takeAction(randomAction);
    return;
  }
  
  const stateValue = trainedValues ? trainedValues[agentState].toFixed(2) : 'N/A';
  log(`Step ${inferenceStep + 1}: State ${agentState} (V=${stateValue}) → ${actionNames[action]} (${actionArrows[action]})`);
  
  takeAction(action);
}

function takeAction(action) {
  const transitions = currentMDP.transitions.filter(
    t => t.fromState === agentState && t.action === action
  );
  
  if (transitions.length === 0) {
    log(`No valid transition for action ${action}`, 'warning');
    inferenceStep++;
    updateInferenceStats();
    return;
  }
  
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
  
  const oldState = agentState;
  agentState = chosenTransition.toState;
  const reward = chosenTransition.reward;
  totalAgentReward += reward;
  
  agentHistory.push({
    step: inferenceStep + 1,
    fromState: oldState,
    action: action,
    toState: agentState,
    reward: reward
  });
  
  inferenceStep++;
  
  const newStateObj = currentMDP.states.find(s => s.id === agentState);
  const isTerminal = newStateObj && (newStateObj.isTerminal || newStateObj.isGoal || newStateObj.isHole);
  
  log(`Transition: ${oldState} → ${agentState}, Reward: ${reward.toFixed(2)}`);
  
  if (isTerminal) {
    const outcome = newStateObj.isGoal ? 'Goal reached!' : 'Fell in hole!';
    log(`${outcome} Total reward: ${totalAgentReward.toFixed(2)}`, newStateObj.isGoal ? 'success' : 'error');
  }
  
  const valuesToShow = trainedValues || {};
  renderGrid(currentMDP, valuesToShow, trainedPolicy, null, agentState);
  renderGraph(currentMDP, valuesToShow, trainedPolicy, null, agentState);
  updateInferenceStats();
}

function animateInference() {
  if (!currentMDP || !trainedPolicy || !agentState) {
    log('Error: Agent not initialized', 'error');
    return;
  }
  
  if (inferenceHandle) return;
  
  elements.stopBtn.disabled = false;
  elements.playBtn.disabled = true;
  
  inferenceHandle = setInterval(() => {
    const currentStateObj = currentMDP.states.find(s => s.id === agentState);
    if (!currentStateObj || currentStateObj.isTerminal) {
      stopInference();
      return;
    }
    
    stepInference();
  }, 500);
}

function stopInference() {
  if (inferenceHandle) {
    clearInterval(inferenceHandle);
    inferenceHandle = null;
    elements.stopBtn.disabled = true;
    elements.playBtn.disabled = false;
    log('Inference stopped');
  }
}

// training functions

function runAlgorithm() {
  if (!currentMDP) {
    log('Error: No environment loaded', 'error');
    return;
  }
  
  const mode = elements.modeSelect.value;
  
  if (mode === 'inference') {
    if (!trainedPolicy) {
      log('No trained policy. Training first...', 'warning');
      runTraining();
      
      setTimeout(() => {
        if (trainedPolicy) {
          log('Training complete. Starting inference...', 'success');
          startInference();
        }
      }, 100);
    } else {
      startInference();
    }
  } else {
    runTraining();
  }
}

function runTraining() {
  const gamma = parseFloat(elements.gammaInput.value) || 0.9;
  const theta = parseFloat(elements.thetaInput.value) || 0.01;
  const episodes = parseInt(elements.episodesInput.value) || 10000;
  const epsilon = parseFloat(elements.epsilonInput.value) || 0.1;
  const alpha = parseFloat(elements.alphaInput.value) || 0.1;
  const algorithm = elements.algorithmSelect.value;
  
  currentMDP.gamma = gamma;
  
  try {
    log(`Starting ${algorithm} training with ${episodes} episodes`);
    
    // Reset step index
    stepIndex = 0;
    
    switch(algorithm) {
      case 'value-iteration':
        runResult = window.valueIteration(currentMDP, theta, gamma);
        break;
      case 'policy-iteration':
        runResult = window.policyIteration(currentMDP, theta, gamma);
        break;
      case 'sarsa':
        log(`Parameters: gamma=${gamma}, alpha=${alpha}, epsilon=${epsilon}`);
        runResult = window.sarsa(currentMDP, episodes, gamma, alpha, epsilon);
        break;
      case 'q-learning':
        log(`Parameters: gamma=${gamma}, alpha=${alpha}, epsilon=${epsilon}`);
        runResult = window.q_learning(currentMDP, episodes, gamma, alpha, epsilon);
        break;
      case 'monte-carlo-control-first-visit':
        log(`Parameters: gamma=${gamma}, epsilon=${epsilon}`);
        runResult = window.monte_carlo_control_first_visit(currentMDP, episodes, gamma, epsilon);
        break;
      case 'monte-carlo-control-every-visit':
        log(`Parameters: gamma=${gamma}, epsilon=${epsilon}`);
        runResult = window.monte_carlo_control_every_visit(currentMDP, episodes, gamma, epsilon);
        break;
      default:
        throw new Error(`Unknown algorithm: ${algorithm}`);
    }
    
    // Log steps info for debugging
    console.log('Run result:', {
      steps: runResult.steps.length,
      firstStep: runResult.steps[0],
      lastStep: runResult.steps[runResult.steps.length - 1],
      hasValues: !!runResult.values,
      hasPolicy: !!runResult.policy
    });
    
    // Always use the final result for display
    trainedPolicy = runResult.policy;
    trainedValues = runResult.values;
    
    // Show first step immediately
    if (runResult.steps.length > 0) {
      const firstStep = runResult.steps[0];
      const initialValues = firstStep.values || runResult.values || {};
      const initialPolicy = firstStep.policy || runResult.policy || {};
      
      renderGrid(currentMDP, initialValues, initialPolicy);
      renderGraph(currentMDP, initialValues, initialPolicy);
      renderTextViews(initialValues, initialPolicy);
    }
    
    updateProgressAndStats();
    updateTrainingStatus();
    
    log(`${algorithm} training completed in ${runResult.iterations} iterations`, 'success');
    
    // Show statistics for model-free algorithms
    if (runResult.successfulEpisodes !== undefined) {
      const successRate = runResult.successRate || (runResult.successfulEpisodes / runResult.iterations * 100).toFixed(1);
      log(`Success rate: ${successRate}% (${runResult.successfulEpisodes}/${runResult.iterations} episodes reached goal)`);
      
      if (runResult.finalEpsilon !== undefined) {
        log(`Final exploration rate (epsilon): ${runResult.finalEpsilon.toFixed(4)}`);
      }
      
      if (successRate < 10) {
        log('Note: Low success rate. Try more episodes or adjust parameters.', 'warning');
      }
    }
    
  } catch (error) {
    console.error('Error:', error);
    log(`Error: ${error.message}`, 'error');
  }
}

function stepOnce() {
  const mode = elements.modeSelect.value;
  
  if (mode === 'inference') {
    stepInference();
  } else {
    if (!currentMDP) {
      log('Error: No environment loaded', 'error');
      return;
    }
    
    if (!runResult) {
      log('Please run the algorithm first by clicking "Run"', 'warning');
      return;
    }
    
    // Check algorithm type
    const algorithm = elements.algorithmSelect.value;
    const isModelFree = ['sarsa', 'q-learning', 'monte-carlo-control-first-visit', 'monte-carlo-control-every-visit'].includes(algorithm);
    
    if (stepIndex >= runResult.steps.length - 1) {
      log('Already at final step');
      
      // Get final values/policy based on algorithm type
      let finalValues, finalPolicy;
      if (isModelFree) {
        finalValues = runResult.values || {};
        finalPolicy = runResult.policy || {};
      } else {
        const lastStep = runResult.steps[runResult.steps.length - 1];
        finalValues = lastStep.values || runResult.values || {};
        finalPolicy = lastStep.policy || runResult.policy || {};
      }
      
      renderGrid(currentMDP, finalValues, finalPolicy);
      renderGraph(currentMDP, finalValues, finalPolicy);
      renderTextViews(finalValues, finalPolicy);
      updateProgressAndStats();
      return;
    }
    
    stepIndex++;
    const s = runResult.steps[stepIndex];
    
    // Extract values/policy based on algorithm type
    let valuesToShow, policyToShow;
    
    if (isModelFree) {
      // For model-free algorithms, try to get values/policy from step
      // or fall back to final results
      if (s.policy) {
        // Monte Carlo might have policy in step
        valuesToShow = s.values || runResult.values || {};
        policyToShow = s.policy || runResult.policy || {};
      } else {
        // SARSA/Q-learning only have Q-table in step
        // Extract policy from Q-table if available
        if (s.Q) {
          valuesToShow = extractStateValuesFromQ(s.Q, currentMDP);
          policyToShow = extractPolicyFromQ(s.Q, currentMDP);
        } else {
          valuesToShow = runResult.values || {};
          policyToShow = runResult.policy || {};
        }
      }
    } else {
      // Value/Policy iteration have direct values/policy
      valuesToShow = s.values || runResult.values || {};
      policyToShow = s.policy || runResult.policy || {};
    }
    
    renderGrid(currentMDP, valuesToShow, policyToShow);
    renderGraph(currentMDP, valuesToShow, policyToShow);
    renderTextViews(valuesToShow, policyToShow);
    updateProgressAndStats();
    
    // Log appropriate info based on algorithm
    if (isModelFree) {
      const successRate = s.successfulEpisodes !== undefined ? 
        ((s.successfulEpisodes / s.iteration) * 100).toFixed(1) + '%' : 'N/A';
      log(`Episode ${s.iteration}: Avg reward ${s.avgReward?.toFixed(2) || 0}, Success rate ${successRate}`);
    } else {
      log(`Step ${stepIndex + 1}/${runResult.steps.length} - Iteration ${s.iteration || stepIndex}, Delta: ${s.delta?.toFixed(4) || 0}`);
    }
  }
}

function animate() {
  const mode = elements.modeSelect.value;
  
  if (mode === 'inference') {
    animateInference();
  } else {
    if (!currentMDP) {
      log('Error: No environment loaded', 'error');
      return;
    }
    
    if (!runResult) {
      log('Please run the algorithm first by clicking "Run"', 'warning');
      return;
    }
    
    if (animHandle) {
      clearInterval(animHandle);
      animHandle = null;
    }
    
    elements.stopBtn.disabled = false;
    elements.animateBtn.disabled = true;
    elements.stepBtn.disabled = true;
    stepIndex = 0;
    
    log(`Starting animation through ${runResult.steps.length} steps`);
    
    animHandle = setInterval(() => {
      if (stepIndex >= runResult.steps.length) {
        clearInterval(animHandle);
        animHandle = null;
        elements.stopBtn.disabled = true;
        elements.animateBtn.disabled = false;
        elements.stepBtn.disabled = false;
        log('Animation finished');
        return;
      }
      
      const s = runResult.steps[stepIndex];
      const algorithm = elements.algorithmSelect.value;
      const isModelFree = ['sarsa', 'q-learning', 'monte-carlo-control-first-visit', 'monte-carlo-control-every-visit'].includes(algorithm);
      
      // Extract values/policy based on algorithm type
      let valuesToShow, policyToShow;
      
      if (isModelFree) {
        if (s.policy) {
          // Monte Carlo might have policy in step
          valuesToShow = s.values || runResult.values || {};
          policyToShow = s.policy || runResult.policy || {};
        } else if (s.Q) {
          // SARSA/Q-learning - extract from Q-table
          valuesToShow = extractStateValuesFromQ(s.Q, currentMDP);
          policyToShow = extractPolicyFromQ(s.Q, currentMDP);
        } else {
          valuesToShow = runResult.values || {};
          policyToShow = runResult.policy || {};
        }
      } else {
        valuesToShow = s.values || runResult.values || {};
        policyToShow = s.policy || runResult.policy || {};
      }
      
      renderGrid(currentMDP, valuesToShow, policyToShow);
      renderGraph(currentMDP, valuesToShow, policyToShow);
      renderTextViews(valuesToShow, policyToShow);
      
      // Update stats display
      updateProgressAndStats();
      
      // Log progress
      if (isModelFree) {
        if (stepIndex % 10 === 0) {
          const successRate = s.successfulEpisodes !== undefined ? 
            ((s.successfulEpisodes / s.iteration) * 100).toFixed(1) + '%' : 'N/A';
          log(`Episode ${s.iteration}: ε=${s.epsilon?.toFixed(3) || 0}, Reward=${s.avgReward?.toFixed(2) || 0}`);
        }
      } else {
        if (stepIndex % 10 === 0) {
          log(`Animation: Step ${stepIndex + 1}/${runResult.steps.length} - Iteration ${s.iteration || stepIndex}`);
        }
      }
      
      stepIndex++;
    }, 500);
  }
}

// Helper functions to extract values/policy from Q-table
function extractStateValuesFromQ(Q, mdp) {
  const values = {};
  const stateIds = mdp.states.map(s => s.id);
  
  stateIds.forEach(stateId => {
    const stateObj = mdp.states.find(s => s.id === stateId);
    
    if (stateObj && (stateObj.isTerminal || stateObj.isGoal || stateObj.isHole)) {
      values[stateId] = 0;
      return;
    }
    
    if (Q[stateId]) {
      const actionValues = Object.values(Q[stateId]);
      values[stateId] = Math.max(...actionValues);
    } else {
      values[stateId] = 0;
    }
  });
  
  return values;
}

function extractPolicyFromQ(Q, mdp) {
  const policy = {};
  const stateIds = mdp.states.map(s => s.id);
  const actionIds = mdp.actions.map(a => a.id);
  
  stateIds.forEach(stateId => {
    const stateObj = mdp.states.find(s => s.id === stateId);
    
    if (stateObj && (stateObj.isTerminal || stateObj.isGoal || stateObj.isHole)) {
      policy[stateId] = null;
      return;
    }
    
    let bestAction = null;
    let bestValue = -Infinity;
    
    if (Q[stateId]) {
      for (const actionId of actionIds) {
        if (Q[stateId][actionId] > bestValue) {
          bestValue = Q[stateId][actionId];
          bestAction = actionId;
        }
      }
    }
    
    policy[stateId] = bestAction;
  });
  
  return policy;
}

function stopAnimation() {
  if (animHandle) {
    clearInterval(animHandle);
    animHandle = null;
    elements.stopBtn.disabled = true;
    elements.animateBtn.disabled = false;
    log('Animation stopped');
  }
  stopInference();
}

function reset() {
  runResult = null;
  stepIndex = 0;
  trainedPolicy = null;
  trainedValues = null;
  stopAnimation();
  stopInference();
  
  elements.stopBtn.disabled = true;
  elements.animateBtn.disabled = false;
  elements.playBtn.style.display = 'none';
  elements.inferenceStats.style.display = 'none';
  
  if (currentMDP) {
    renderGrid(currentMDP, {}, {});
    renderGraph(currentMDP, {}, {});
  } else {
    renderGrid({ states: [], actions: [] }, {}, {});
    renderGraph({ states: [], actions: [] }, {}, {});
  }
  
  renderTextViews({}, {});
  updateProgressAndStats();
  updateTrainingStatus();
  elements.log.innerHTML = '';
  log('System reset');
}

// Event Listeners
elements.envSelect.addEventListener('change', loadSelectedEnvironment);

elements.algorithmSelect.addEventListener('change', function() {
  const algorithm = this.value;
  const isModelFree = ['sarsa', 'q-learning', 'monte-carlo-control-first-visit', 'monte-carlo-control-every-visit'].includes(algorithm);
  const isTD = ['sarsa', 'q-learning'].includes(algorithm);
  
  // show/hide parameters based on algorithm type
  elements.mcParams.style.display = isModelFree ? 'inline-block' : 'none';
  
  // show alpha only for TD methods
  elements.alphaLabel.style.display = isTD ? 'inline-block' : 'none';
  
  // update default episodes based on algorithm
  if (algorithm.includes('monte-carlo')) {
    elements.episodesInput.value = '10000';
  } else if (isTD) {
    elements.episodesInput.value = '1000';
  }
  
  updateEnvInfo();
  updateAlgorithmExplanation();
});

elements.modeSelect.addEventListener('change', function() {
  const isInference = this.value === 'inference';
  if (isInference && !trainedPolicy) {
    log('Switch to Inference mode. Note: Train first or click "Run" to train automatically.', 'warning');
  }
  updateEnvInfo();
});

elements.loadBtn.addEventListener('click', loadSelectedEnvironment);
elements.runBtn.addEventListener('click', runAlgorithm);
elements.stepBtn.addEventListener('click', stepOnce);
elements.animateBtn.addEventListener('click', animate);
elements.stopBtn.addEventListener('click', stopAnimation);
elements.playBtn.addEventListener('click', animateInference);
elements.resetBtn.addEventListener('click', reset);

// Initialize modals
document.querySelectorAll('.hint').forEach(hint => {
  hint.addEventListener('click', function() {
    const envType = elements.envSelect.value;
    document.getElementById('envModal').style.display = 'block';
    
    // show environment info
    document.getElementById('gridworldInfo').style.display = envType === 'gridworld' ? 'block' : 'none';
    document.getElementById('frozenlakeInfo').style.display = envType === 'frozenlake' ? 'block' : 'none';
  });
});

document.querySelectorAll('.close-modal').forEach(btn => {
  btn.addEventListener('click', function() {
    document.getElementById('envModal').style.display = 'none';
  });
});

window.addEventListener('click', function(e) {
  if (e.target.id === 'envModal') {
    document.getElementById('envModal').style.display = 'none';
  }
});

loadSelectedEnvironment();