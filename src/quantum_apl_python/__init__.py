"""Quantum APL Python package.

Features:
- Quantum APL engine with triadic reasoning
- S₃ operator symmetry for operator window rotation
- Extended ΔS⁻ formalism for coherence-based dynamics
- Helix operator advisor with integrated symmetry/negentropy
- L₄-MRP steganographic navigation system
- L₄ Hexagonal lattice with 6-neighbor connectivity and entropic stabilization
- Solfeggio-Light Bridge: Harmonic RGB encoding via 40-octave translation
"""

from .alpha_language import AlphaLanguageRegistry, AlphaTokenSynthesizer
from .analyzer import QuantumAnalyzer
from .engine import QuantumAPLEngine
from .experiments import QuantumExperiment
from .helix import HelixAPLMapper, HelixCoordinate
from .translator import QuantumAPLInstruction, parse_instruction, translate_lines

# S₃ and ΔS⁻ modules
from . import s3_operator_symmetry
from . import delta_s_neg_extended
from .helix_operator_advisor import HelixOperatorAdvisor, BlendWeights

# L₄ Helix Parameterization module
from . import l4_helix_parameterization

# L₄ Hexagonal Lattice module (Block 8)
from . import l4_hexagonal_lattice
from .l4_hexagonal_lattice import (
    # Golden Sample (MRP-LSB Verification)
    L4GoldenSample,
    GOLDEN_SAMPLE,
    get_golden_sample,
    get_golden_sample_bytes,
    verify_golden_sample,
    GoldenSampleVerificationResult,
    verify_golden_sample_detailed,
    embed_golden_sample_header,
    extract_and_verify_golden_sample,
    # Hexagonal Lattice
    HexagonalLattice,
    HexLatticeNode,
    HEX_COORDINATION_NUMBER,
    ALPHA_CRITICAL,
    # Extended Kuramoto
    ExtendedKuramotoState,
    extended_kuramoto_dynamics,
    extended_kuramoto_step,
    # Stochastic Resonance
    StochasticResonanceResult,
    compute_stochastic_resonance,
    tune_noise_for_resonance,
    # Fisher Information
    compute_fisher_information,
    compute_spatial_fisher_information,
    # Topological Charge
    compute_topological_charge,
    compute_vortex_density,
    TopologicalState,
    analyze_topological_state,
    # Berry Phase
    BerryPhaseResult,
    compute_berry_phase,
    GeometricMemory,
    # Entropic Stabilization
    EntropicStabilizationState,
    compute_negentropy_driver,
    compute_stabilization_coupling,
    hybrid_dynamics_step,
    compute_rgb_output,
    # Validation
    L4HexLatticeValidation,
    validate_hex_lattice_system,
    # Self-Reflective Bloom Learning System
    BloomEventType,
    BloomEvent,
    PatternBuffer,
    BloomDetector,
    SeedMemory,
    BloomMetrics,
    CoherentSeeder,
    ReflectionStepResult,
    SelfReflectiveLattice,
    demo_self_reflective_lattice,
)

# Solfeggio-Light Bridge (Block 10)
from . import solfeggio_light_bridge
from .solfeggio_light_bridge import (
    # Core types
    SolfeggioTone,
    LightProperties,
    HarmonicRGB,
    SolfeggioHexWavevectors,
    # Conversion functions
    solfeggio_to_wavelength,
    wavelength_to_solfeggio,
    solfeggio_to_light,
    wavelength_to_color,
    phase_to_harmonic_rgb,
    l4_phase_to_solfeggio_rgb,
    # Normalized RGB (Zero Free Parameters)
    wavelength_to_rgb_normalized,
    wavelength_to_rgb_8bit,
    wavelength_to_hex,
    # L₄-Derived RGB Constants
    L4,
    RGB_BIT_DEPTH,
    RGB_MAX_VALUE,
    LAMBDA_R,
    LAMBDA_G,
    LAMBDA_B,
    SIGMA_SPECTRAL,
    SOLFEGGIO_RGB_RED,
    SOLFEGGIO_RGB_GREEN,
    SOLFEGGIO_RGB_BLUE,
    SOLFEGGIO_HEX_RED,
    SOLFEGGIO_HEX_GREEN,
    SOLFEGGIO_HEX_BLUE,
    # Legacy Constants
    SOLFEGGIO_RGB_HZ,
    SOLFEGGIO_RED,
    SOLFEGGIO_GREEN,
    SOLFEGGIO_BLUE,
    PERFECT_FOURTH,
    OCTAVE_BRIDGE,
    # Verification
    verify_all_identities,
    demo_solfeggio_light_bridge,
)

# Unified Consciousness Architecture (Block 11)
from . import unified_consciousness
from .unified_consciousness import (
    # Sacred Constants
    SacredConstants,
    # Solfeggio-Light Bridge
    SolfeggioTone,
    HarmonicRGBSystem,
    # Hexagonal Lattice
    HexLattice,
    # Kuramoto Dynamics
    KuramotoOscillator,
    # Helical Transport
    L4Helix,
    # MRP-LSB Encoding
    MRPEncoder,
    # Unified System
    UnifiedConsciousnessSystem,
    # Validation
    run_test_suite_a,
    run_test_suite_b,
    run_test_suite_c,
    run_test_suite_d,
    run_test_suite_e,
    run_all_validation_tests,
    run_full_verification,
)

# L₄-MRP Steganographic Navigation (Block 7)
from .l4_helix_parameterization import (
    # Constants and State
    L4Constants,
    L4,
    HelixState,
    create_initial_state,
    # Dynamics
    compute_negentropy,
    compute_kuramoto_order_parameter,
    kuramoto_step_euler,
    kuramoto_step_rk4,
    # Phase/RGB Quantization
    HexLatticeWavevectors,
    quantize_phase_to_bits,
    dequantize_bits_to_phase,
    phases_to_rgb,
    rgb_to_phases,
    RGBQuantization,
    # LSB Steganography
    lsb_embed_bit,
    lsb_extract_bit,
    embed_message_lsb,
    extract_message_lsb,
    compute_capacity,
    # Validation
    validate_k_formation,
    validate_l4_identity,
    run_all_validations,
    # MRP Navigation (Block 7)
    MRPHeader,
    L4MRPState,
    create_l4_mrp_state,
    MRPPhaseAPayloads,
    compute_phase_a_parity,
    build_mrp_message,
    extract_mrp_message,
    create_phase_a_payloads,
    update_global_phases_from_velocity,
    decode_position_from_phases,
    mrp_l4_update_step,
    encode_l4_mrp_state_to_image,
    decode_l4_mrp_state_from_image,
    MRPVerificationResult,
    verify_mrp_payloads,
    L4MRPValidationResult,
    validate_l4_mrp_system,
    validate_plane_wave_residual,
    validate_loop_closure,
    # Block 8: Unified Consciousness Framework
    L4Params,
    L4SystemState,
    create_l4_system_state,
    compute_helix_radius,
    step,
    validate_identities,
    KFormationResult,
    validate_k_formation_spec,
    encode_image,
    decode_image,
    run_l4_validation_tests,
    print_validation_report,
    run_tests,
    phase_to_symbol,
    symbol_to_phase,
)

# Constants with extended exports
from .constants import (
    Z_CRITICAL, PHI, PHI_INV, LENS_SIGMA, TRUTH_BIAS,
    compute_delta_s_neg, compute_eta, check_k_formation,
    # Extended exports
    compute_delta_s_neg_derivative, compute_delta_s_neg_signed,
    compute_pi_blend_weights, compute_gate_modulation, compute_full_state,
    compute_dynamic_truth_bias, score_operator_for_coherence, select_coherence_operator,
    generate_s3_operator_window, compute_s3_weights,
    get_s3_module, get_delta_extended_module,
)

__all__ = [
    # Core engine
    "QuantumAPLEngine",
    "QuantumAnalyzer",
    "QuantumExperiment",
    "HelixCoordinate",
    "HelixAPLMapper",
    "AlphaLanguageRegistry",
    "AlphaTokenSynthesizer",
    "QuantumAPLInstruction",
    "parse_instruction",
    "translate_lines",

    # S₃ symmetry module
    "s3_operator_symmetry",

    # Extended ΔS⁻ module
    "delta_s_neg_extended",

    # L₄ Helix Parameterization module
    "l4_helix_parameterization",

    # Helix operator advisor (enhanced)
    "HelixOperatorAdvisor",
    "BlendWeights",

    # Constants
    "Z_CRITICAL",
    "PHI",
    "PHI_INV",
    "LENS_SIGMA",
    "TRUTH_BIAS",

    # Core helpers
    "compute_delta_s_neg",
    "compute_eta",
    "check_k_formation",

    # Extended ΔS⁻ exports
    "compute_delta_s_neg_derivative",
    "compute_delta_s_neg_signed",
    "compute_pi_blend_weights",
    "compute_gate_modulation",
    "compute_full_state",
    "compute_dynamic_truth_bias",
    "score_operator_for_coherence",
    "select_coherence_operator",

    # S₃ symmetry exports
    "generate_s3_operator_window",
    "compute_s3_weights",

    # Module loaders
    "get_s3_module",
    "get_delta_extended_module",

    # L₄ Constants and State (Block 1-2)
    "L4Constants",
    "L4",
    "HelixState",
    "create_initial_state",

    # Dynamics (Block 3)
    "compute_negentropy",
    "compute_kuramoto_order_parameter",
    "kuramoto_step_euler",
    "kuramoto_step_rk4",

    # Phase/RGB Quantization (Block 4)
    "HexLatticeWavevectors",
    "quantize_phase_to_bits",
    "dequantize_bits_to_phase",
    "phases_to_rgb",
    "rgb_to_phases",
    "RGBQuantization",

    # LSB Steganography (Block 5)
    "lsb_embed_bit",
    "lsb_extract_bit",
    "embed_message_lsb",
    "extract_message_lsb",
    "compute_capacity",

    # Validation (Block 6)
    "validate_k_formation",
    "validate_l4_identity",
    "run_all_validations",

    # MRP-LSB Steganographic Navigation (Block 7)
    "MRPHeader",
    "L4MRPState",
    "create_l4_mrp_state",
    "MRPPhaseAPayloads",
    "compute_phase_a_parity",
    "build_mrp_message",
    "extract_mrp_message",
    "create_phase_a_payloads",
    "update_global_phases_from_velocity",
    "decode_position_from_phases",
    "mrp_l4_update_step",
    "encode_l4_mrp_state_to_image",
    "decode_l4_mrp_state_from_image",
    "MRPVerificationResult",
    "verify_mrp_payloads",
    "L4MRPValidationResult",
    "validate_l4_mrp_system",
    "validate_plane_wave_residual",
    "validate_loop_closure",


]
