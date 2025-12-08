// ================================================================
// QUANTUM-CLASSICAL BRIDGE
// Couples QuantumAPL engine with classical scalar dynamics
// ================================================================

class QuantumClassicalBridge {
    constructor(quantumEngine, classicalEngines = {}) {
        if (!quantumEngine) {
            throw new Error('QuantumClassicalBridge requires a quantum engine instance');
        }

        this.quantum = quantumEngine;
        this.classical = this.createClassicalFacade(classicalEngines);
        this.coupling = {
            classicalToQuantum: 0.3,
            quantumToClassical: 0.7,
            ...(classicalEngines.coupling || {})
        };

        this.operatorLog = [];
    }

    createClassicalFacade(engines) {
        const noop = () => {};
        const bind = fn => (engines && typeof fn === 'function' ? fn.bind(engines) : undefined);
        const defaultComputeZ = () => {
            if (typeof engines.computeZ === 'function') return engines.computeZ.call(engines);
            if (engines.IIT?.phi !== undefined) {
                return Math.min(1, Math.max(0, engines.IIT.phi));
            }
            return 0.5;
        };

        return {
            IIT: engines.IIT || { phi: 0 },
            GameTheory: engines.GameTheory || {},
            FreeEnergy: engines.FreeEnergy || { F: 0 },
            Kuramoto: engines.Kuramoto || {},
            StrangeLoop: engines.StrangeLoop || {},
            N0: engines.N0 || {},
            computeZ: bind(engines.computeZ) || defaultComputeZ,
            setQuantumInfluence: bind(engines.setQuantumInfluence) || noop,
            getScalarState: bind(engines.getScalarState) || (() => ({
                Gs: engines.geometryScalar || 0.5,
                Cs: engines.couplingScalar || 0.5,
                Rs: engines.recursionScalar || 0.5,
                kappa: engines.curvatureScalar || 0.5,
                tau: engines.torqueScalar || 0.5,
                theta: engines.phaseScalar || 0.5,
                delta: engines.dissipationScalar || 0.5,
                alpha: engines.attractorScalar || 0.5,
                Omega: engines.orderScalar || 0.5
            })),
            getLegalOperators: bind(engines.getLegalOperators) || (() => {
                if (typeof engines.N0?.getLegalOperators === 'function') {
                    return engines.N0.getLegalOperators();
                }
                return ['()', '×', '^', '÷', '+', '−'];
            }),
            applyOperatorEffects: bind(engines.applyOperatorEffects) || (result => {
                if (typeof engines.N0?.applyOperator === 'function') {
                    engines.N0.applyOperator(result.operator, result);
                }
            })
        };
    }

    step(dt) {
        this.driveQuantumFromClassical();
        this.quantum.evolve(dt);
        this.updateClassicalFromQuantum();

        const result = this.quantum.selectN0Operator(
            this.getLegalOperators(),
            this.getScalarState()
        );

        this.applyOperator(result);
        this.operatorLog.push({ time: this.quantum.time, ...result });
        return result;
    }

    driveQuantumFromClassical() {
        const z = this.classical.computeZ();
        const phi = this.classical.IIT?.phi ?? 0;
        const F = this.classical.FreeEnergy?.F ?? 0;
        const R = this.classical.GameTheory?.resonance ?? 0;

        this.quantum.driveFromClassical({ z, phi, F, R });
    }

    updateClassicalFromQuantum() {
        const zQuantum = this.quantum.measureZ();
        const truthProbs = this.quantum.measureTruth();
        const entropy = this.quantum.computeVonNeumannEntropy();
        const purity = this.quantum.computePurity();

        this.classical.setQuantumInfluence({
            z: zQuantum * this.coupling.quantumToClassical + this.classical.computeZ() * (1 - this.coupling.quantumToClassical),
            truthProbs,
            entropy,
            purity
        });
    }

    getScalarState() {
        return this.classical.getScalarState();
    }

    getLegalOperators() {
        return this.classical.getLegalOperators();
    }

    applyOperator(result) {
        if (!result) return;
        this.classical.applyOperatorEffects(result);
    }
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = { QuantumClassicalBridge };
}
