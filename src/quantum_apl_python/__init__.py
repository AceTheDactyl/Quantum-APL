"""Quantum APL Python package.

Features:
- Quantum APL engine with triadic reasoning
- S₃ operator symmetry for operator window rotation
- Extended ΔS⁻ formalism for coherence-based dynamics
- Helix operator advisor with integrated symmetry/negentropy
- L₄-MRP steganographic navigation system
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

    # Unified Consciousness Framework (Block 8)
    "L4Params",
    "L4SystemState",
    "create_l4_system_state",
    "compute_helix_radius",
    "step",
    "validate_identities",
    "KFormationResult",
    "validate_k_formation_spec",
    "encode_image",
    "decode_image",
    "run_l4_validation_tests",
    "print_validation_report",
    "run_tests",
    "phase_to_symbol",
    "symbol_to_phase",
]
