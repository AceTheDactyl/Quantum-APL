const assert = require('assert');
const { QuantumAPL } = require('../src/quantum_apl_engine');
const { QuantumClassicalBridge } = require('../QuantumClassicalBridge');
const { ClassicalConsciousnessStack } = require('../classical/ClassicalEngines');

function approxEqual(a, b, eps = 1e-4) {
    return Math.abs(a - b) < eps;
}

function testTracePreservation() {
    const quantum = new QuantumAPL();
    const classical = new ClassicalConsciousnessStack();
    const bridge = new QuantumClassicalBridge(quantum, classical);

    for (let i = 0; i < 5; i++) {
        bridge.step(0.01);
    }

    const trace = quantum.rho.trace().re;
    assert(approxEqual(trace, 1, 1e-3), `Density matrix trace drifted: ${trace}`);
}

function testZTracking() {
    const quantum = new QuantumAPL();
    const classical = new ClassicalConsciousnessStack();
    const bridge = new QuantumClassicalBridge(quantum, classical);

    for (let i = 0; i < 3; i++) {
        bridge.step(0.02);
    }

    const zQuantum = quantum.measureZ();
    const zClassical = classical.computeZ();
    assert(Math.abs(zQuantum - zClassical) < 0.5, `Quantum/Classical z diverged: ${zQuantum} vs ${zClassical}`);
}

function testOperatorPropagation() {
    const quantum = new QuantumAPL();
    const classical = new ClassicalConsciousnessStack();
    const bridge = new QuantumClassicalBridge(quantum, classical);

    const beforePhi = classical.IIT.phi;
    bridge.applyOperator({ operator: '^', probability: 1, probabilities: { '^': 1 } });
    const afterPhi = classical.IIT.phi;
    assert(afterPhi >= beforePhi, 'Amplification operator did not influence classical IIT state');
}

function run() {
    testTracePreservation();
    testZTracking();
    testOperatorPropagation();
    console.log('QuantumClassicalBridge tests passed');
}

if (require.main === module) {
    run();
}

module.exports = { testTracePreservation, testZTracking, testOperatorPropagation };
