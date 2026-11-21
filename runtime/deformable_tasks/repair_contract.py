"""Explicit repair selection; selection never constitutes runtime execution evidence."""
TASK_GROUPS = {
    "robot_volume": {"entry": "scripts/tasks.py", "unified_task": "robot_volume",
                     "unified_lifecycle_validated": False,
                     "required_contact": ["articulated_rigid_volume", "friction", "rotating_frames"]},
    "cloth_grip": {"entry": "scripts/tasks.py", "unified_task": "cloth_grip",
                   "required_contact": ["rigid_surface", "friction", "slip_release"]},
    "cloth_fold": {"entry": "scripts/tasks.py", "unified_task": "cloth_fold",
                   "required_contact": ["surface_self_contact", "friction"]},
    "multi_soft": {"entry": "scripts/tasks.py", "unified_task": "multi_soft",
                   "required_contact": ["volume_volume", "two_sided_updates"]},
}
LATEST_CONTACT_REPAIR = {
    "id": "contact-candidate-activation-v5",
    "unified_backend_registered": False,
    "validated_scope": ["single_volume", "frictionless", "fixed_cartesian_articulation",
                        "nonrotating_world_z_plane"],
    "full_cycle_receipt": "results/native-contact-candidate-full-continuation-v1/receipt.json",
    "all_four_task_groups_validated": False,
}
from .weight_backend import WEIGHT_BACKENDS,SUPPORT_BACKENDS
def selection(backend):
    requested_backend = backend
    from .robot_surface_backend import ROUTES
    if backend in ROUTES:
        base,policy=ROUTES[backend];result=selection(base)
        extra=['finite_table_boundary_node_coverage','within_step_native_multiplier_history','original_row_native_copyback'] if policy=='persistent' else ['same_binary_surface_adapter_disabled']
        result.update(requested_backend=backend,selected_components=result['selected_components']+extra,scope_note='Experimental authored robot/finite-table route. Selection is not execution, force/work, regression or cost acceptance.')
        return result
    components = []
    from .cloth_history_transport_backend import CONTROL as TRANSPORT_CONTROL,POLICIES as TRANSPORT_POLICIES
    from .cloth_reentry_runtime_backend import BACKEND as RUNTIME_BACKEND
    if backend==TRANSPORT_CONTROL:backend='cloth-split-envelope-refresh16-reentry-runtime-control'
    elif backend in TRANSPORT_POLICIES:
        components.append('consecutive_preEE_geometry_snapshots')
        components.append('certified_retained_side_reentry' if TRANSPORT_POLICIES[backend] else 'retained_side_transport_disabled_ablation')
        backend=RUNTIME_BACKEND
    from .cloth_reentry_runtime_backend import CONTROL as RUNTIME_CONTROL,POLICIES as RUNTIME_POLICIES
    if backend==RUNTIME_CONTROL:
        components.append('same_binary_native_control')
        backend='cloth-split-envelope-refresh16-history-finite-control'
    elif backend in RUNTIME_POLICIES:
        components.extend(['explicit_device_initialization_policy','transition_only_surface_release_counters','finite_feature_surface_separation_release'])
        components.append('certified_finite_feature_initialization' if RUNTIME_POLICIES[backend] else 'finite_feature_initialization_disabled_ablation')
        backend='cloth-split-envelope-refresh16-edge-query-history-finite-contact'
    if backend=="cloth-split-envelope-refresh16-edge-query-history-surface-reentry-contact":
        components.append("certified_finite_feature_initialization")
        backend="cloth-split-envelope-refresh16-edge-query-history-surface-release-contact"
    if backend=="cloth-split-envelope-refresh16-edge-query-history-surface-release-contact":
        components.append("finite_feature_surface_separation_release")
        backend="cloth-split-envelope-refresh16-edge-query-history-ring-observe"
    if backend=="cloth-split-envelope-refresh16-edge-query-history-ring-observe":
        components.append("read_only_all_feature_pre_failure_ring")
        backend="cloth-split-envelope-refresh16-edge-query-history-inner-observe"
    if backend=="cloth-split-envelope-refresh16-edge-query-history-inner-observe":
        components.append("read_only_inner_ee_feature_history_trace")
        backend="cloth-split-envelope-refresh16-edge-query-history-finite-contact"
    if backend=="cloth-split-envelope-refresh16-edge-query-history-finite-contact":
        components.append("bounded_finite_segment_history_initialization")
        backend="cloth-split-envelope-refresh16-edge-query-history-birth-contact"
    elif backend=="cloth-split-envelope-refresh16-history-finite-control":
        backend="cloth-split-envelope-refresh16-history-birth-control"
    if backend=="cloth-split-envelope-refresh16-edge-query-history-birth-contact":
        components.append("conservative_endpoint_transition_history_birth")
        backend="cloth-split-envelope-refresh16-edge-query-history-diagnostic-contact"
    elif backend=="cloth-split-envelope-refresh16-history-birth-control":
        backend="cloth-split-envelope-refresh16-history-events-control"
    if backend=="cloth-split-envelope-refresh16-edge-query-history-diagnostic-contact":
        components.append("first_ee_history_rejection_observation")
        backend="cloth-split-envelope-refresh16-edge-query-history-events-contact"
    if backend=="cloth-split-envelope-refresh16-edge-query-history-events-contact":
        components.append("event_ordered_ee_history_publication")
        backend="cloth-split-envelope-refresh16-edge-query-history-contact"
    elif backend=="cloth-split-envelope-refresh16-history-events-control":
        backend="cloth-split-envelope-refresh16-history-control"
    if backend=="cloth-split-envelope-refresh16-edge-query-history-contact":
        components.append("persistent_owned_ee_side_history")
        backend="cloth-split-envelope-refresh16-edge-query-contact"
    elif backend=="cloth-split-envelope-refresh16-history-control":
        components.append("history_binding_control")
        backend="cloth-split-envelope-refresh16-control"
    if backend in ("cloth-split-envelope-refresh16-compact-proximity-control","cloth-split-envelope-refresh16-edge-query-graph-compact-proximity-contact"):
        components.append("preintegration_geometry_triggered_observation")
        backend=backend.replace("compact-proximity-","compact-observe-")
    if backend in ("cloth-split-control","cloth-split-refresh-control","cloth-split-envelope-control","cloth-split-envelope-refresh-control"):components.append("cloth_contact_binding_control")
    if backend in ("cloth-split-contact","cloth-split-refresh-contact","cloth-split-refresh-fused","cloth-split-envelope-contact","cloth-split-envelope-refresh-contact"):components.extend(["cloth_fp64_contact_geometry","cloth_shared_absolute_weight_query","cloth_normal_velocity_friction_split","cloth_two_sided_degree_weighted_publication"])
    if backend in ("cloth-split-refresh-control","cloth-split-refresh-contact","cloth-split-refresh-fused","cloth-split-envelope-refresh-control","cloth-split-envelope-refresh-contact"):components.append("native_surface_pair_refresh_four_per_step")
    if backend in ("cloth-split-envelope-control","cloth-split-envelope-contact","cloth-split-envelope-refresh-control","cloth-split-envelope-refresh-contact"):components.append("native_rigid_candidate_envelope_20mm_trial")
    if backend in ('cloth-split-envelope-refresh16-compact-observe-control','cloth-split-envelope-refresh16-control','cloth-split-envelope-refresh16-contact','cloth-split-envelope-refresh16-edge-query-contact','cloth-split-envelope-refresh16-edge-query-graph-contact','cloth-split-envelope-refresh16-edge-query-graph-reduced-contact','cloth-split-envelope-refresh16-edge-query-graph-compact-contact','cloth-split-envelope-refresh16-edge-query-graph-compact-observe-contact'):
        components.extend(['native_surface_pair_refresh_sixteen_per_step','native_rigid_candidate_envelope_20mm_trial'])
        components.extend(['cloth_contact_binding_control'] if backend.endswith('control') else ['cloth_fp64_contact_geometry','cloth_shared_absolute_weight_query','cloth_normal_velocity_friction_split','cloth_two_sided_degree_weighted_publication'])
    if backend in ('cloth-split-envelope-refresh16-edge-query-contact','cloth-split-envelope-refresh16-edge-query-graph-contact','cloth-split-envelope-refresh16-edge-query-graph-reduced-contact','cloth-split-envelope-refresh16-edge-query-graph-compact-contact','cloth-split-envelope-refresh16-edge-query-graph-compact-observe-contact'):components.append('cloth_ee_unequal_length_validity_correction')
    if backend in ('cloth-split-envelope-refresh16-edge-query-graph-contact','cloth-split-envelope-refresh16-edge-query-graph-reduced-contact','cloth-split-envelope-refresh16-edge-query-graph-compact-contact','cloth-split-envelope-refresh16-edge-query-graph-compact-observe-contact'):components.append('explicit_owned_contact_cuda_graph')
    if backend in ('cloth-split-envelope-refresh16-edge-query-graph-reduced-contact','cloth-split-envelope-refresh16-edge-query-graph-compact-contact','cloth-split-envelope-refresh16-edge-query-graph-compact-observe-contact'):components.append('warp_aggregated_execution_counters')
    if backend in ('cloth-split-envelope-refresh16-edge-query-graph-compact-contact','cloth-split-envelope-refresh16-edge-query-graph-compact-observe-contact'):components.append('bounded_contact_grid_stride_dispatch_128')
    if backend in ('cloth-split-envelope-refresh16-compact-observe-control','cloth-split-envelope-refresh16-edge-query-graph-compact-observe-contact'):components.append('intrusive_compact_cloth_self_row_observation')
    if backend=="cloth-split-refresh-fused":components.append("fused_frozen_descriptor_preparation")
    if backend in ("volume-batched-geometry-control","volume-batched-geometry-mapping"):components.append("ordered_gpu_root_geometry_observation")
    if backend in ("volume-batched-control","volume-batched-geometry-control"):components.extend(["volume_cp_mapping_control","frame_boundary_state_validation","device_cp_execution_counters"])
    if backend in ("volume-batched-mapping","volume-batched-geometry-mapping"):components.extend(["volume_cp_mass_split","volume_copy_velocity_merge","native_contact_kernels","batched_two_volume_cp_helpers","frame_boundary_state_validation","device_cp_execution_counters"])
    if backend=="volume-mapping":components.extend(["volume_cp_mass_split","volume_copy_velocity_merge","native_contact_kernels"])
    if backend=="ss-cp-observe-native":components.extend(["two_volume_cp_mapping_observer","native_soft_soft_contact_binding_control","native_velocity_stage_split","velocity_cone_binding_control"])
    if backend in ("ss-cp-map-native","ss-cp-map-common"):
        components.extend(["two_volume_cp_mass_split","two_volume_copy_velocity_merge","native_velocity_stage_split","velocity_cone_binding_control"])
        components.extend(["two_volume_common_contact_relaxation","matched_cumulative_contact_history_relaxation"] if backend=="ss-cp-map-common" else ["native_soft_soft_contact_binding_control"])
    if backend in ("ss-cone-control","ss-cone-plane"):
        components.extend(["two_volume_common_contact_relaxation","matched_cumulative_contact_history_relaxation","native_velocity_stage_split"])
        components.append("auxiliary_fixed_plane_velocity_cone" if backend=="ss-cone-plane" else "velocity_cone_binding_control")
    if backend in ("ss-common-history","ss-common-nodes"):components.append("two_volume_common_contact_relaxation")
    if backend=="ss-common-history":components.append("matched_cumulative_contact_history_relaxation")
    if backend=="ss-common-control":components.append("two_volume_binding_control")
    if backend=='cp-contact-weights' or backend in SUPPORT_BACKENDS:components.append('normalized_contact_endpoint_weights')
    if backend=='cp-contact-weights-support':components.append('positive_weight_reference_support')
    if backend=='cp-contact-weights-query-control':components.append('native_query_clone_control')
    if backend=='cp-contact-weights-control':components.append('native_contact_clone_control')
    if backend in ("contact-refresh", "contact-adaptive", "contact-local"):
        components.append("native_surface_contact_schedule")
    if backend == "contact-local":
        components.append("explicit_surface_self_contact_filter")
    if backend in ("volume-project", "cp-volume-project"):
        components.append("end_step_positive_volume_projection")
    if backend in ("volume-project-control", "cp-volume-project-control"):
        components.append("volume_read_write_control")
    if backend.startswith("cp-") and backend not in ("cp-control",)+WEIGHT_BACKENDS:
        components.append("single_volume_material_kernel_intervention")
    if backend.startswith("cp-contact") and backend not in WEIGHT_BACKENDS:
        components.append("experimental_single_volume_normal_contact")
    return {
        "requested_backend": requested_backend,
        "selected_components": components,
        "execution_verified": False,
        "runtime_evidence_required": True,
        "latest_candidate_activation_selected": False,
        "scope_note": "Selection only. Candidate-v5 composition is not registered by this backend name. "
                      "Environment-selected extensions require separate runtime evidence.",
    }
def integration_inventory(backends):
    return {
        "schema_version": 1,
        "status": "integration_required",
        "task_groups": TASK_GROUPS,
        "latest_contact_repair": LATEST_CONTACT_REPAIR,
        "backend_selections": {name: selection(name) for name in backends},
        "platform_acceptance": False,
        "next_required": [
            "Integrate the actual Franka grasp/squeeze task into the shared configuration and lifecycle.",
            "Generalize contact and momentum mappings to surface, self-contact, multiple bodies and rotating articulations.",
            "Register each implemented repair and reject unsupported shapes before GPU launch.",
            "Record executed corrections, eligible rounds, both-side updates and source/build identities.",
            "Run frozen native/control/improved comparisons under the preregistered protocol.",
        ],
    }

