function generateDeterministicTransitions(rows, cols, states) {
  const transitions = [];
  const totalStates = rows * cols;
  
  function isValidMove(row, col, action) {
    switch(action) {
      case 'up': return row > 0;
      case 'down': return row < rows - 1;
      case 'left': return col > 0;
      case 'right': return col < cols - 1;
      default: return false;
    }
  }
  
  function getNextState(currentState, action) {
    const row = Math.floor(currentState / cols);
    const col = currentState % cols;
    
    switch(action) {
      case 'up': return (row - 1) * cols + col;
      case 'down': return (row + 1) * cols + col;
      case 'left': return row * cols + (col - 1);
      case 'right': return row * cols + (col + 1);
      default: return currentState;
    }
  }
  
  for (let state = 0; state < totalStates; state++) {
    const row = Math.floor(state / cols);
    const col = state % cols;
    const stateId = `s${state}`;
    
    const isHole = states.find(s => s.id === stateId)?.isHole;
    const isGoal = states.find(s => s.id === stateId)?.isGoal;
    const isTerminal = isHole || isGoal;
    
    ['up', 'right', 'down', 'left'].forEach(action => {
      let nextState, reward;
      
      if (isTerminal) {
        nextState = state;
        reward = 0;
      } else {
        if (isValidMove(row, col, action)) {
          nextState = getNextState(state, action);
        } else {
          nextState = state;
        }
        
        const nextStateId = `s${nextState}`;
        const nextStateInfo = states.find(s => s.id === nextStateId);
        
        if (nextStateInfo?.isGoal) {
          reward = 1;
        } else if (nextStateInfo?.isHole) {
          reward = -1;
        } else {
          reward = 0;
        }
      }
      
      transitions.push({
        fromState: stateId,
        action: action,
        toState: `s${nextState}`,
        probability: 1.0,
        reward: reward
      });
    });
  }
  
  return transitions;
}

window.EXAMPLE_MDPS = {
  gridworld: {
    name: "GridWorld 3x3",
    gamma: 0.9,
    states: [
      { id: "s1", name: "S1", x: 0, y: 0, isStart: true },
      { id: "s2", name: "S2", x: 1, y: 0 },
      { id: "s3", name: "S3", x: 2, y: 0 },
      { id: "s4", name: "S4", x: 0, y: 1 },
      { id: "s5", name: "S5", x: 1, y: 1 },
      { id: "s6", name: "S6", x: 2, y: 1 },
      { id: "s7", name: "S7", x: 0, y: 2 },
      { id: "s8", name: "S8", x: 1, y: 2 },
      { id: "s9", name: "Goal", x: 2, y: 2, isTerminal: true, isGoal: true }
    ],
    actions: [
      { id: "up", name: "Up" },
      { id: "down", name: "Down" },
      { id: "left", name: "Left" },
      { id: "right", name: "Right" }
    ],
    transitions: [
      { fromState: "s1", action: "right", toState: "s2", probability: 0.8, reward: -1 },
      { fromState: "s1", action: "right", toState: "s1", probability: 0.2, reward: -1 },
      { fromState: "s1", action: "down", toState: "s4", probability: 0.8, reward: -1 },
      { fromState: "s1", action: "down", toState: "s1", probability: 0.2, reward: -1 },
      { fromState: "s2", action: "right", toState: "s3", probability: 0.8, reward: -1 },
      { fromState: "s2", action: "right", toState: "s2", probability: 0.2, reward: -1 },
      { fromState: "s2", action: "left", toState: "s1", probability: 0.8, reward: -1 },
      { fromState: "s2", action: "left", toState: "s2", probability: 0.2, reward: -1 },
      { fromState: "s3", action: "down", toState: "s6", probability: 0.8, reward: -1 },
      { fromState: "s3", action: "down", toState: "s3", probability: 0.2, reward: -1 },
      { fromState: "s4", action: "right", toState: "s5", probability: 0.8, reward: -1 },
      { fromState: "s4", action: "right", toState: "s4", probability: 0.2, reward: -1 },
      { fromState: "s5", action: "right", toState: "s6", probability: 0.8, reward: -1 },
      { fromState: "s5", action: "right", toState: "s5", probability: 0.2, reward: -1 },
      { fromState: "s5", action: "down", toState: "s8", probability: 0.8, reward: -1 },
      { fromState: "s5", action: "down", toState: "s5", probability: 0.2, reward: -1 },
      { fromState: "s6", action: "down", toState: "s9", probability: 0.8, reward: 100 },
      { fromState: "s6", action: "down", toState: "s6", probability: 0.2, reward: -1 },
      { fromState: "s8", action: "right", toState: "s9", probability: 0.8, reward: 100 },
      { fromState: "s8", action: "right", toState: "s8", probability: 0.2, reward: -1 }
    ],
    startState: "s1"
  },
  
  frozenlake_4x4: {
    name: "Frozen Lake 4x4",
    gamma: 0.9,
    states: [
      { id: "s0", name: "🏁", x: 0, y: 0, isStart: true },
      { id: "s1", name: "❄️", x: 1, y: 0 },
      { id: "s2", name: "❄️", x: 2, y: 0 },
      { id: "s3", name: "❄️", x: 3, y: 0 },
      { id: "s4", name: "❄️", x: 0, y: 1 },
      { id: "s5", name: "🌊", x: 1, y: 1, isTerminal: true, isHole: true },
      { id: "s6", name: "❄️", x: 2, y: 1 },
      { id: "s7", name: "🌊", x: 3, y: 1, isTerminal: true, isHole: true },
      { id: "s8", name: "❄️", x: 0, y: 2 },
      { id: "s9", name: "❄️", x: 1, y: 2 },
      { id: "s10", name: "❄️", x: 2, y: 2 },
      { id: "s11", name: "🌊", x: 3, y: 2, isTerminal: true, isHole: true },
      { id: "s12", name: "🌊", x: 0, y: 3, isTerminal: true, isHole: true },
      { id: "s13", name: "❄️", x: 1, y: 3 },
      { id: "s14", name: "❄️", x: 2, y: 3 },
      { id: "s15", name: "🎯", x: 3, y: 3, isTerminal: true, isGoal: true }
    ],
    actions: [
      { id: "up", name: "Up" },
      { id: "right", name: "Right" },
      { id: "down", name: "Down" },
      { id: "left", name: "Left" }
    ],
    transitions: [],
    startState: "s0"
  }
};

window.EXAMPLE_MDPS.frozenlake_4x4.transitions = generateDeterministicTransitions(
  4, 4, window.EXAMPLE_MDPS.frozenlake_4x4.states
);