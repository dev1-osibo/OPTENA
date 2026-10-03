"""
DCPLOS — Data Center Physical Layer Orchestration System
Core engine: Interdependency graph, cascade propagation, risk scoring.
"""
import networkx as nx
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from enum import Enum


class ComponentType(Enum):
    UTILITY_FEED = "utility_feed"
    TRANSFORMER = "transformer"
    ATS = "ats"  # Automatic Transfer Switch
    UPS = "ups"
    PDU = "pdu"
    RACK_PDU = "rack_pdu"
    CRAC = "crac"
    CHILLER = "chiller"
    COOLING_LOOP = "cooling_loop"
    RACK = "rack"
    SERVER = "server"
    CORE_SWITCH = "core_switch"
    AGG_SWITCH = "agg_switch"
    TOR_SWITCH = "tor_switch"


class EdgeType(Enum):
    POWER_FEEDS = "power_feeds"
    THERMAL_SERVES = "thermal_serves"
    NETWORK_CONNECTS = "network_connects"
    PROTECTS = "protects"


class ComponentStatus(Enum):
    OPERATIONAL = "operational"
    DEGRADED = "degraded"
    FAILED = "failed"
    THROTTLED = "throttled"


# ============================================================
# C-10: Criticality weights for the CRS weighted-sum formula.
# Maps a component's SLA tier to the Criticality(D) coefficient.
# ============================================================
CRITICALITY_WEIGHT = {
    "critical": 1.0,
    "standard": 0.5,
    "low": 0.2,
}

# ============================================================
# C-4: Redundancy failover success probability by equipment tier.
# Physical basis: automatic transfer switches / static transfer switches /
# redundant UPS modules from premium vendors transfer reliably in
# milliseconds; budget and legacy gear has materially higher transfer-failure
# ("failover-to-nothing") rates. These values give equipment quality a real
# causal path into cascade CONTAINMENT — a failover that fails lets the cascade
# propagate past the redundant path that should have stopped it.
#
# Values are illustrative-but-defensible ranges; the experiment reports results
# as a function of these, and the paper discloses them as modeled assumptions.
# ============================================================
TIER_FAILOVER_SUCCESS_PROB = {
    "tier1_premium": 0.98,
    "tier2_midrange": 0.93,
    "tier3_budget": 0.85,
    "tier4_legacy": 0.70,
}


@dataclass
class Component:
    id: str
    name: str
    component_type: ComponentType
    rated_capacity_kw: float
    current_load_kw: float
    age_years: float = 0.0
    mtbf_hours: float = 100000.0  # Mean Time Between Failures
    last_maintenance_days: int = 0
    redundancy_role: str = "primary"  # primary, backup, N+1
    status: ComponentStatus = ComponentStatus.OPERATIONAL
    row: int = 0  # physical row location
    position: int = 0  # rack position in row
    thermal_mass_kwh_per_c: float = 0.5  # thermal inertia
    max_temp_c: float = 35.0  # thermal shutdown threshold
    current_temp_c: float = 22.0  # current operating temp
    # --- C-10: business criticality of workloads served by this component ---
    # Used as the Criticality(D) term in the CRS weighted formula. One of
    # "critical" / "standard" / "low"; mapped to a numeric weight in the engine.
    sla_tier: str = "standard"
    # --- C-4: manufacturing/quality tier, used to derive redundancy failover
    # reliability. Kept as an explicit attribute so equipment quality has a real
    # causal path into cascade PROPAGATION (not only failure_probability). ---
    equipment_tier: str = "tier1_premium"

    @property
    def utilization(self) -> float:
        if self.rated_capacity_kw == 0:
            return 0.0
        return self.current_load_kw / self.rated_capacity_kw

    @property
    def headroom_kw(self) -> float:
        return max(0, self.rated_capacity_kw - self.current_load_kw)

    @property
    def failure_probability(self) -> float:
        """Estimate failure probability based on age, load, maintenance"""
        # Base probability from MTBF
        base_prob = 1 - np.exp(-self.age_years * 8760 / self.mtbf_hours)
        # Stress factor from utilization (exponential above 80%)
        stress = 1.0 + max(0, (self.utilization - 0.8)) * 5.0
        # Maintenance factor (higher risk if overdue)
        maint_factor = 1.0 + max(0, (self.last_maintenance_days - 180)) / 365.0
        prob = min(0.95, base_prob * stress * maint_factor)
        return round(prob, 4)


@dataclass
class CascadeEvent:
    """A single event in a cascade propagation timeline"""
    time_seconds: float
    component_id: str
    component_name: str
    event_type: str  # "power_loss", "thermal_rise", "overload", "throttle", "shutdown"
    description: str
    severity: str  # "warning", "critical", "failure"


class DCPLOSEngine:
    """Core DCPLOS engine for interdependency modeling and cascade analysis"""

    def __init__(self):
        self.graph = nx.DiGraph()
        self.components: Dict[str, Component] = {}

    def add_component(self, component: Component):
        """Add a component to the facility model"""
        self.components[component.id] = component
        self.graph.add_node(component.id, **{
            'name': component.name,
            'type': component.component_type.value,
            'capacity': component.rated_capacity_kw,
            'load': component.current_load_kw,
            'status': component.status.value,
            'row': component.row,
            'position': component.position
        })

    def add_dependency(self, source_id: str, target_id: str, edge_type: EdgeType,
                       capacity_contribution_kw: float = 0.0, failover_time_s: float = 0.0,
                       failover_success_prob: float = 1.0):
        """Add a dependency edge between components"""
        self.graph.add_edge(source_id, target_id, **{
            'edge_type': edge_type.value,
            'capacity_contribution': capacity_contribution_kw,
            'failover_time': failover_time_s,
            'failover_success_prob': failover_success_prob,
        })

    def _attempt_failover(self, comp_id: str) -> bool:
        """
        C-4: Determine whether a redundant path protects comp_id and whether
        that failover SUCCEEDS. A component is protected if it has an incoming
        PROTECTS edge (a backup unit). The failover success probability is taken
        from the protecting edge (set from the backup's equipment tier at
        facility-construction time). Returns True if the cascade is contained at
        this component (failover succeeded), False otherwise.

        This is the mechanism that gives equipment quality a real causal path
        into cascade MAGNITUDE: premium backups contain cascades reliably;
        legacy backups fail to transfer more often, letting the cascade spread.
        """
        best_prob = 0.0
        has_protection = False
        for pred in self.graph.predecessors(comp_id):
            edge = self.graph.edges[pred, comp_id]
            if edge.get('edge_type') == EdgeType.PROTECTS.value:
                # Only a backup that is itself still operational can carry load.
                if self.components[pred].status == ComponentStatus.OPERATIONAL:
                    has_protection = True
                    best_prob = max(best_prob, edge.get('failover_success_prob', 1.0))
        if not has_protection:
            return False
        return np.random.random() < best_prob

    def get_downstream_dependents(self, component_id: str) -> List[str]:
        """Get all components that depend on the given component"""
        return list(nx.descendants(self.graph, component_id))

    def get_upstream_providers(self, component_id: str) -> List[str]:
        """Get all components that the given component depends on"""
        return list(nx.ancestors(self.graph, component_id))

    def simulate_cascade(self, initiating_component_id: str,
                         degradation_pct: float = 100.0) -> Tuple[List[CascadeEvent], Dict]:
        """
        Simulate cascade failure propagation from an initiating component.
        
        Args:
            initiating_component_id: The component that fails/degrades
            degradation_pct: 0-100, how much capacity is lost (100 = total failure)
            
        Returns:
            Tuple of (timeline of events, final state dict)
        """
        timeline: List[CascadeEvent] = []
        affected_set = set()
        component_states = {cid: {'status': 'operational', 'load_change_kw': 0.0, 'temp_c': c.current_temp_c}
                           for cid, c in self.components.items()}
        # --- C-10: per-component impact tracking for the weighted CRS formula.
        # For each affected component D we record load_impact (fraction of its
        # capacity affected, 0-1) and time_affected (seconds into the cascade at
        # which it was first impacted). Time_Exposure(D) is later derived as the
        # interval from time_affected to the end of the cascade. ---
        component_impact: Dict[str, Dict[str, float]] = {}

        def _record_impact(cid: str, load_impact: float, t: float):
            existing = component_impact.get(cid)
            if existing is None:
                component_impact[cid] = {
                    'load_impact': float(max(0.0, min(1.0, load_impact))),
                    'time_affected': float(t),
                }
            else:
                # Keep the earliest impact time and the largest load impact seen.
                existing['load_impact'] = float(max(existing['load_impact'], max(0.0, min(1.0, load_impact))))
                existing['time_affected'] = float(min(existing['time_affected'], t))

        # Initial failure
        init_comp = self.components[initiating_component_id]
        capacity_lost_kw = init_comp.rated_capacity_kw * (degradation_pct / 100.0)
        component_states[initiating_component_id]['status'] = 'failed' if degradation_pct >= 100 else 'degraded'
        affected_set.add(initiating_component_id)
        _record_impact(initiating_component_id, degradation_pct / 100.0, 0.0)

        timeline.append(CascadeEvent(
            time_seconds=0,
            component_id=initiating_component_id,
            component_name=init_comp.name,
            event_type="initial_failure",
            description=f"{init_comp.name} loses {degradation_pct:.0f}% capacity ({capacity_lost_kw:.1f} kW)",
            severity="failure" if degradation_pct >= 100 else "critical"
        ))

        # Propagation queue: (time, component_id, impact_type, impact_kw)
        propagation_queue = []

        # C-4: if the initiating component has a live redundant peer that
        # transfers successfully, its power-fed dependents stay powered — the
        # power cascade is contained at the source. Rolled once for the
        # initiating component; thermal/network paths are unaffected.
        init_power_contained = self._attempt_failover(initiating_component_id)
        if init_power_contained:
            timeline.append(CascadeEvent(
                time_seconds=0.01,
                component_id=initiating_component_id,
                component_name=init_comp.name,
                event_type="failover_success",
                description=f"{init_comp.name} failed but redundant peer carried load — power cascade contained at source",
                severity="warning"
            ))

        # Find direct dependents and their edge types
        for neighbor in self.graph.successors(initiating_component_id):
            edge_data = self.graph.edges[initiating_component_id, neighbor]
            edge_type = edge_data['edge_type']

            if edge_type == EdgeType.POWER_FEEDS.value:
                if not init_power_contained:
                    propagation_queue.append((0.01, neighbor, 'power_loss', capacity_lost_kw))
            elif edge_type == EdgeType.THERMAL_SERVES.value:
                propagation_queue.append((60.0, neighbor, 'thermal_rise', capacity_lost_kw))
            elif edge_type == EdgeType.NETWORK_CONNECTS.value:
                propagation_queue.append((0.5, neighbor, 'network_loss', 0))
            elif edge_type == EdgeType.PROTECTS.value:
                propagation_queue.append((0.01, neighbor, 'redundancy_loss', 0))

        # Process propagation
        propagation_queue.sort(key=lambda x: x[0])
        max_iterations = 500  # safety limit
        iteration = 0

        while propagation_queue and iteration < max_iterations:
            iteration += 1
            time_s, comp_id, impact_type, impact_kw = propagation_queue.pop(0)
            
            if comp_id in affected_set and component_states[comp_id]['status'] == 'failed':
                continue  # already fully failed

            comp = self.components[comp_id]
            affected_set.add(comp_id)

            if impact_type == 'power_loss':
                # Component loses power
                component_states[comp_id]['status'] = 'failed'
                _record_impact(comp_id, 1.0, time_s)
                timeline.append(CascadeEvent(
                    time_seconds=time_s,
                    component_id=comp_id,
                    component_name=comp.name,
                    event_type="power_loss",
                    description=f"{comp.name} loses power supply ({impact_kw:.1f} kW path interrupted)",
                    severity="failure"
                ))
                # C-4: attempt redundant failover. If this component has a live
                # redundant peer that transfers successfully, its downstream
                # dependents stay powered — the cascade is contained here.
                if self._attempt_failover(comp_id):
                    timeline.append(CascadeEvent(
                        time_seconds=time_s + 0.01,
                        component_id=comp_id,
                        component_name=comp.name,
                        event_type="failover_success",
                        description=f"{comp.name} failed but redundant peer carried load — downstream protected",
                        severity="warning"
                    ))
                else:
                    # No protection (or failover failed): propagate to dependents
                    for downstream in self.graph.successors(comp_id):
                        if downstream not in affected_set:
                            edge_data = self.graph.edges[comp_id, downstream]
                            propagation_queue.append((time_s + 0.01, downstream, 'power_loss', comp.current_load_kw))

            elif impact_type == 'thermal_rise':
                # Cooling lost — temperature rises over time
                heat_rate_c_per_min = comp.current_load_kw / (comp.thermal_mass_kwh_per_c * 60) if comp.thermal_mass_kwh_per_c > 0 else 5.0
                time_to_threshold_s = max(60, (comp.max_temp_c - comp.current_temp_c) / (heat_rate_c_per_min / 60))

                component_states[comp_id]['status'] = 'degraded'
                timeline.append(CascadeEvent(
                    time_seconds=time_s,
                    component_id=comp_id,
                    component_name=comp.name,
                    event_type="thermal_rise",
                    description=f"{comp.name} cooling lost — temp rising at {heat_rate_c_per_min:.1f}°C/min, throttle in {time_to_threshold_s/60:.1f} min",
                    severity="warning"
                ))

                # After time_to_threshold, thermal throttling kicks in
                throttle_time = time_s + time_to_threshold_s
                timeline.append(CascadeEvent(
                    time_seconds=throttle_time,
                    component_id=comp_id,
                    component_name=comp.name,
                    event_type="thermal_throttle",
                    description=f"{comp.name} reaches {comp.max_temp_c}°C — thermal throttling/shutdown triggered",
                    severity="critical"
                ))
                component_states[comp_id]['status'] = 'throttled'
                _record_impact(comp_id, 1.0, throttle_time)

                # Workload migration from throttled component overloads neighbors
                for downstream in self.graph.successors(comp_id):
                    if downstream not in affected_set:
                        edge_data = self.graph.edges[comp_id, downstream]
                        if edge_data['edge_type'] in [EdgeType.POWER_FEEDS.value, EdgeType.THERMAL_SERVES.value]:
                            # Load redistribution
                            propagation_queue.append((throttle_time + 5, downstream, 'overload', comp.current_load_kw * 0.5))

            elif impact_type == 'overload':
                # Component receives extra load from failed neighbor
                new_load = comp.current_load_kw + impact_kw
                if new_load > comp.rated_capacity_kw:
                    # Overloaded — this component also degrades
                    component_states[comp_id]['status'] = 'degraded'
                    _record_impact(comp_id, 1.0, time_s)
                    timeline.append(CascadeEvent(
                        time_seconds=time_s,
                        component_id=comp_id,
                        component_name=comp.name,
                        event_type="overload",
                        description=f"{comp.name} overloaded: {new_load:.1f}/{comp.rated_capacity_kw:.1f} kW ({new_load/comp.rated_capacity_kw*100:.0f}% — exceeds capacity)",
                        severity="critical"
                    ))
                    # Cascade continues
                    for downstream in self.graph.successors(comp_id):
                        if downstream not in affected_set:
                            propagation_queue.append((time_s + 2, downstream, 'overload', impact_kw * 0.7))
                else:
                    # Can absorb the load (partial impact, no propagation)
                    _record_impact(comp_id, new_load / comp.rated_capacity_kw if comp.rated_capacity_kw > 0 else 0.0, time_s)
                    timeline.append(CascadeEvent(
                        time_seconds=time_s,
                        component_id=comp_id,
                        component_name=comp.name,
                        event_type="load_increase",
                        description=f"{comp.name} absorbs redirected load: {new_load:.1f}/{comp.rated_capacity_kw:.1f} kW ({new_load/comp.rated_capacity_kw*100:.0f}%)",
                        severity="warning"
                    ))

            elif impact_type == 'network_loss':
                component_states[comp_id]['status'] = 'failed'
                _record_impact(comp_id, 1.0, time_s)
                timeline.append(CascadeEvent(
                    time_seconds=time_s,
                    component_id=comp_id,
                    component_name=comp.name,
                    event_type="network_loss",
                    description=f"{comp.name} loses network connectivity — isolated",
                    severity="failure"
                ))

            elif impact_type == 'redundancy_loss':
                timeline.append(CascadeEvent(
                    time_seconds=time_s,
                    component_id=comp_id,
                    component_name=comp.name,
                    event_type="redundancy_loss",
                    description=f"{comp.name} redundancy reduced — now single point of failure",
                    severity="warning"
                ))

            propagation_queue.sort(key=lambda x: x[0])

        # Compute final state
        final_state = {
            'total_affected': len(affected_set),
            'failed_components': sum(1 for s in component_states.values() if s['status'] == 'failed'),
            'degraded_components': sum(1 for s in component_states.values() if s['status'] == 'degraded'),
            'throttled_components': sum(1 for s in component_states.values() if s['status'] == 'throttled'),
            'affected_racks': sum(1 for cid in affected_set if self.components[cid].component_type == ComponentType.RACK),
            'cascade_duration_minutes': max(e.time_seconds for e in timeline) / 60 if timeline else 0,
            'component_states': component_states,
            'affected_set': affected_set,
            # C-10: per-component impact fractions and first-impact times, plus
            # the cascade end time, so a weighted CRS can be computed downstream.
            'component_impact': component_impact,
            'cascade_end_s': max((e.time_seconds for e in timeline), default=0.0),
        }

        return timeline, final_state

    def compute_cascade_risk_scores(self) -> Dict[str, float]:
        """
        Compute CRS for every component using the weighted formula documented in
        the paper/patent (C-10 fix — previously the code used a flat affected
        count with none of the per-downstream weighting terms):

            CRS(C) = [ Σ_{D in cascade(C)} Criticality(D) x Load_Impact(D)
                                          x Time_Exposure(D) ]
                     x P(failure of C) x (1 - Mitigation_Factor)

        where, for each downstream component D in C's cascade set:
          - Criticality(D)  = CRITICALITY_WEIGHT[D.sla_tier]  (business weight)
          - Load_Impact(D)  = fraction of D's capacity affected (0-1)
          - Time_Exposure(D) = minutes D is impacted during the cascade window
        """
        risk_scores = {}

        for comp_id, comp in self.components.items():
            # Simulate cascade from this component
            timeline, final_state = self.simulate_cascade(comp_id, degradation_pct=100)

            # --- C-10: weighted downstream impact sum ---
            impact_map = final_state.get('component_impact', {})
            cascade_end_s = final_state.get('cascade_end_s', 0.0)
            weighted_impact = 0.0
            for d_id, d_info in impact_map.items():
                d_comp = self.components.get(d_id)
                if d_comp is None:
                    continue
                criticality = CRITICALITY_WEIGHT.get(d_comp.sla_tier, CRITICALITY_WEIGHT["standard"])
                load_impact = d_info.get('load_impact', 0.0)
                # Time_Exposure(D): minutes from first impact to end of cascade.
                # Floor at a small positive value so instantaneous-but-real
                # impacts still contribute a nonzero exposure term.
                exposure_min = max((cascade_end_s - d_info.get('time_affected', 0.0)) / 60.0, 0.1)
                weighted_impact += criticality * load_impact * exposure_min

            probability = comp.failure_probability
            
            # Mitigation: if component has redundancy, risk is lower
            has_backup = any(
                self.graph.edges[pred, comp_id].get('edge_type') == EdgeType.PROTECTS.value
                for pred in self.graph.predecessors(comp_id)
                if self.graph.edges[pred, comp_id].get('edge_type') == EdgeType.PROTECTS.value
            )
            mitigation = 0.7 if has_backup else 0.0

            crs = weighted_impact * probability * (1 - mitigation)
            risk_scores[comp_id] = round(crs, 4)

        return risk_scores

    def get_risk_heatmap_data(self) -> List[Dict]:
        """Generate risk heatmap data for rack positions"""
        risk_scores = self.compute_cascade_risk_scores()
        heatmap_data = []

        for comp_id, comp in self.components.items():
            if comp.component_type == ComponentType.RACK:
                # Rack's exposure = sum of CRS of all upstream providers
                upstream = self.get_upstream_providers(comp_id)
                exposure = sum(risk_scores.get(uid, 0) for uid in upstream)
                
                heatmap_data.append({
                    'rack_id': comp_id,
                    'name': comp.name,
                    'row': comp.row,
                    'position': comp.position,
                    'exposure_score': round(exposure, 3),
                    'load_kw': comp.current_load_kw,
                    'capacity_kw': comp.rated_capacity_kw
                })

        return heatmap_data

    def identify_cspcf(self, threshold_multiplier: float = 2.0) -> List[Dict]:
        """Identify Critical Single Points of Cascading Failure"""
        risk_scores = self.compute_cascade_risk_scores()
        
        if not risk_scores:
            return []
        
        mean_crs = np.mean(list(risk_scores.values()))
        threshold = mean_crs * threshold_multiplier
        
        cspcf = []
        for comp_id, crs in risk_scores.items():
            if crs > threshold:
                comp = self.components[comp_id]
                _, state = self.simulate_cascade(comp_id, 100)
                cspcf.append({
                    'component_id': comp_id,
                    'name': comp.name,
                    'type': comp.component_type.value,
                    'crs': crs,
                    'blast_radius': state['total_affected'],
                    'affected_racks': state['affected_racks'],
                    'cascade_duration_min': state['cascade_duration_minutes'],
                    'failure_probability': comp.failure_probability
                })
        
        cspcf.sort(key=lambda x: x['crs'], reverse=True)
        return cspcf

    def compute_rack_deliverable_capacity(self, rack_id: str) -> Dict:
        """
        Compute the actual deliverable capacity at a rack position
        considering all coupled constraints (power, cooling, network).
        """
        rack = self.components[rack_id]
        
        # Find upstream power path capacity
        power_headroom = float('inf')
        cooling_headroom = float('inf')
        network_available = True
        
        for predecessor in self.graph.predecessors(rack_id):
            edge_data = self.graph.edges[predecessor, rack_id]
            pred_comp = self.components[predecessor]
            
            if edge_data['edge_type'] == EdgeType.POWER_FEEDS.value:
                power_headroom = min(power_headroom, pred_comp.headroom_kw)
            elif edge_data['edge_type'] == EdgeType.THERMAL_SERVES.value:
                cooling_headroom = min(cooling_headroom, pred_comp.headroom_kw)
            elif edge_data['edge_type'] == EdgeType.NETWORK_CONNECTS.value:
                network_available = pred_comp.headroom_kw > 0
        
        # Also check deeper upstream (PDU → UPS → Transformer)
        for predecessor in self.graph.predecessors(rack_id):
            for grandparent in self.graph.predecessors(predecessor):
                gp_comp = self.components[grandparent]
                edge_data = self.graph.edges[grandparent, predecessor]
                if edge_data['edge_type'] == EdgeType.POWER_FEEDS.value:
                    power_headroom = min(power_headroom, gp_comp.headroom_kw)
        
        # Deliverable = min of all constraints
        rack_space_kw = rack.rated_capacity_kw - rack.current_load_kw
        deliverable_kw = min(rack_space_kw, power_headroom, cooling_headroom)
        if not network_available:
            deliverable_kw = 0
        
        # Identify binding constraint
        constraints = {
            'rack_space': rack_space_kw,
            'power_path': power_headroom,
            'cooling': cooling_headroom,
        }
        binding = min(constraints, key=constraints.get)
        
        return {
            'rack_id': rack_id,
            'name': rack.name,
            'row': rack.row,
            'position': rack.position,
            'current_load_kw': rack.current_load_kw,
            'rated_capacity_kw': rack.rated_capacity_kw,
            'deliverable_kw': max(0, round(deliverable_kw, 1)),
            'power_headroom_kw': round(min(power_headroom, 9999), 1),
            'cooling_headroom_kw': round(min(cooling_headroom, 9999), 1),
            'rack_space_kw': round(rack_space_kw, 1),
            'binding_constraint': binding,
            'network_available': network_available
        }

    def recommend_placement(self, workload_kw: float, criticality: str = "standard",
                           count: int = 1) -> List[Dict]:
        """
        Recommend optimal rack positions for a new workload.
        Considers: capacity feasibility, cascade exposure, and load balancing.
        
        Args:
            workload_kw: Power requirement per rack unit
            criticality: "critical", "standard", or "low"
            count: Number of rack positions needed
            
        Returns:
            Ranked list of recommended positions
        """
        # Compute risk scores (cached if available)
        risk_scores = self.compute_cascade_risk_scores()
        
        candidates = []
        for comp_id, comp in self.components.items():
            if comp.component_type != ComponentType.RACK:
                continue
            
            # Get deliverable capacity
            capacity_info = self.compute_rack_deliverable_capacity(comp_id)
            
            if capacity_info['deliverable_kw'] < workload_kw:
                continue  # can't fit
            
            # Compute cascade exposure for this position
            upstream = self.get_upstream_providers(comp_id)
            exposure_score = sum(risk_scores.get(uid, 0) for uid in upstream)
            
            # Compute how much placing here increases systemic risk
            # (adding load reduces headroom of upstream components)
            risk_increase = workload_kw / max(1, capacity_info['power_headroom_kw']) * 10
            
            # Utilization balance score (prefer positions that even out load)
            all_rack_utils = [c.utilization for c in self.components.values() 
                           if c.component_type == ComponentType.RACK]
            mean_util = np.mean(all_rack_utils) if all_rack_utils else 0.5
            new_util = (comp.current_load_kw + workload_kw) / comp.rated_capacity_kw
            balance_penalty = abs(new_util - mean_util) * 5
            
            # Combined score (lower = better placement)
            if criticality == "critical":
                # For critical workloads, heavily weight cascade exposure
                score = exposure_score * 3 + risk_increase * 2 + balance_penalty
            elif criticality == "low":
                # For low-criticality, weight balance and cost more
                score = exposure_score * 0.5 + risk_increase + balance_penalty * 2
            else:
                score = exposure_score * 1.5 + risk_increase * 1.5 + balance_penalty
            
            candidates.append({
                'rack_id': comp_id,
                'name': comp.name,
                'row': comp.row,
                'position': comp.position,
                'deliverable_kw': capacity_info['deliverable_kw'],
                'binding_constraint': capacity_info['binding_constraint'],
                'cascade_exposure': round(exposure_score, 3),
                'risk_increase': round(risk_increase, 3),
                'balance_score': round(balance_penalty, 3),
                'overall_score': round(score, 3),
                'current_load_kw': comp.current_load_kw,
                'headroom_after': round(capacity_info['deliverable_kw'] - workload_kw, 1)
            })
        
        # Sort by overall score (lower = better)
        candidates.sort(key=lambda x: x['overall_score'])
        
        # Return top recommendations
        for i, c in enumerate(candidates):
            c['rank'] = i + 1
            if i < count:
                c['recommendation'] = '✅ RECOMMENDED'
            elif i < count * 2:
                c['recommendation'] = '⚠️ ACCEPTABLE'
            else:
                c['recommendation'] = '— AVAILABLE'
        
        return candidates

    def identify_unlock_moves(self, target_row: int = None, min_capacity_gain_kw: float = 5.0) -> List[Dict]:
        """
        Identify workload relocations that free capacity in constrained zones.
        Like disk defragmentation but for data center physical resources.
        
        Args:
            target_row: If specified, focus on freeing capacity in this row
            min_capacity_gain_kw: Minimum capacity gain to consider a move worthwhile
            
        Returns:
            List of recommended moves ranked by capacity freed vs disruption
        """
        moves = []
        
        # Identify constrained zones (rows where cooling or power is tight)
        row_constraints = {}
        for comp_id, comp in self.components.items():
            if comp.component_type in [ComponentType.CRAC, ComponentType.PDU]:
                utilization = comp.utilization
                if utilization > 0.7:  # >70% loaded = constrained
                    row_constraints[comp.row] = {
                        'component': comp,
                        'utilization': utilization,
                        'headroom_kw': comp.headroom_kw,
                        'type': comp.component_type.value
                    }
        
        if target_row:
            constrained_rows = [target_row]
        else:
            constrained_rows = list(row_constraints.keys())
        
        if not constrained_rows:
            return []
        
        # Find racks in constrained zones that could be moved
        for constrained_row in constrained_rows:
            # Get racks in this zone
            racks_in_zone = [
                (cid, c) for cid, c in self.components.items()
                if c.component_type == ComponentType.RACK 
                and c.row == constrained_row
                and c.current_load_kw > 0
            ]
            
            # Find potential destination racks in less-loaded zones
            unconstrained_racks = [
                (cid, c) for cid, c in self.components.items()
                if c.component_type == ComponentType.RACK
                and c.row not in constrained_rows
                and c.headroom_kw > min_capacity_gain_kw
            ]
            
            for source_id, source_rack in racks_in_zone:
                # Only consider moving low-density workloads (easier to relocate)
                if source_rack.current_load_kw > 20:  # Skip high-density (AI) racks
                    continue
                
                for dest_id, dest_rack in unconstrained_racks:
                    # Can the destination absorb this workload?
                    dest_capacity = self.compute_rack_deliverable_capacity(dest_id)
                    if dest_capacity['deliverable_kw'] >= source_rack.current_load_kw:
                        # Calculate benefit
                        capacity_freed = source_rack.current_load_kw
                        
                        # Disruption cost (based on workload size and distance)
                        row_distance = abs(source_rack.row - dest_rack.row)
                        disruption = source_rack.current_load_kw * 0.1 * (1 + row_distance * 0.2)
                        
                        # Benefit/cost ratio
                        benefit_ratio = capacity_freed / max(0.1, disruption)
                        
                        if capacity_freed >= min_capacity_gain_kw:
                            moves.append({
                                'source_rack': source_rack.name,
                                'source_row': source_rack.row,
                                'source_load_kw': source_rack.current_load_kw,
                                'dest_rack': dest_rack.name,
                                'dest_row': dest_rack.row,
                                'dest_headroom_kw': dest_capacity['deliverable_kw'],
                                'capacity_freed_kw': round(capacity_freed, 1),
                                'constrained_zone': f"Row {constrained_row}",
                                'constraint_type': row_constraints.get(constrained_row, {}).get('type', 'unknown'),
                                'disruption_score': round(disruption, 2),
                                'benefit_ratio': round(benefit_ratio, 2)
                            })
        
        # Sort by benefit ratio (higher = better move)
        moves.sort(key=lambda x: x['benefit_ratio'], reverse=True)
        
        # Deduplicate — only show best destination per source
        seen_sources = set()
        unique_moves = []
        for move in moves:
            if move['source_rack'] not in seen_sources:
                seen_sources.add(move['source_rack'])
                unique_moves.append(move)
        
        return unique_moves[:10]  # Top 10 moves

    def compute_facility_yield(self, target_density_kw: float = 40.0) -> Dict:
        """
        Compute how many racks at target_density can be deployed,
        with and without orchestration (unlock moves).
        """
        # Without orchestration: just count positions with enough deliverable capacity
        naive_count = 0
        positions_detail = []
        
        for comp_id, comp in self.components.items():
            if comp.component_type != ComponentType.RACK:
                continue
            capacity = self.compute_rack_deliverable_capacity(comp_id)
            if capacity['deliverable_kw'] >= target_density_kw:
                naive_count += 1
            positions_detail.append(capacity)
        
        # With orchestration: simulate moving low-density workloads out of constrained zones
        # This frees cooling/power headroom, enabling more deployments
        unlock_moves = self.identify_unlock_moves(min_capacity_gain_kw=5.0)
        total_freed_kw = sum(m['capacity_freed_kw'] for m in unlock_moves)
        
        # Each unlock move frees capacity in a constrained zone
        # Estimate how many additional target-density racks could fit
        # (conservative: only count freed capacity that exceeds target)
        additional_racks = 0
        cumulative_freed = 0
        for move in unlock_moves:
            cumulative_freed += move['capacity_freed_kw']
            if cumulative_freed >= target_density_kw:
                additional_racks += 1
                cumulative_freed -= target_density_kw
        
        orchestrated_count = naive_count + additional_racks
        total_racks = sum(1 for c in self.components.values() if c.component_type == ComponentType.RACK)
        
        return {
            'target_density_kw': target_density_kw,
            'total_rack_positions': total_racks,
            'deployable_without_orchestration': naive_count,
            'deployable_with_orchestration': orchestrated_count,
            'capacity_gain_pct': round((orchestrated_count - naive_count) / max(1, naive_count) * 100, 1) if naive_count > 0 else (100.0 if orchestrated_count > 0 else 0.0),
            'unlock_moves_available': len(unlock_moves),
            'capacity_freed_by_moves_kw': round(total_freed_kw, 1),
            'additional_racks_from_moves': additional_racks
        }


def create_demo_facility() -> DCPLOSEngine:
    """Create a realistic demo facility for testing"""
    engine = DCPLOSEngine()

    # === POWER INFRASTRUCTURE ===
    # Utility feeds (2x for redundancy)
    engine.add_component(Component("util_a", "Utility Feed A", ComponentType.UTILITY_FEED, 3000, 1800, age_years=5))
    engine.add_component(Component("util_b", "Utility Feed B", ComponentType.UTILITY_FEED, 3000, 1800, age_years=5))
    
    # Transformers
    engine.add_component(Component("xfmr_a", "Transformer A", ComponentType.TRANSFORMER, 2500, 1500, age_years=8))
    engine.add_component(Component("xfmr_b", "Transformer B", ComponentType.TRANSFORMER, 2500, 1500, age_years=8))
    
    # UPS Systems
    engine.add_component(Component("ups_a", "UPS System A", ComponentType.UPS, 2000, 1200, age_years=4, mtbf_hours=80000))
    engine.add_component(Component("ups_b", "UPS System B", ComponentType.UPS, 2000, 1200, age_years=6, mtbf_hours=80000))
    
    # PDUs (4 main PDUs serving different rows)
    engine.add_component(Component("pdu_1", "PDU Row 1-2", ComponentType.PDU, 600, 420, age_years=3, row=1))
    engine.add_component(Component("pdu_2", "PDU Row 3-4", ComponentType.PDU, 600, 380, age_years=3, row=3))
    engine.add_component(Component("pdu_3", "PDU Row 5-6", ComponentType.PDU, 600, 510, age_years=5, row=5))
    engine.add_component(Component("pdu_4", "PDU Row 7-8", ComponentType.PDU, 600, 350, age_years=2, row=7))

    # === COOLING INFRASTRUCTURE ===
    engine.add_component(Component("chiller_1", "Chiller Plant 1", ComponentType.CHILLER, 2000, 1400, age_years=6))
    engine.add_component(Component("crac_1", "CRAC Unit 1 (Row 1-2)", ComponentType.CRAC, 500, 380, age_years=4, row=1))
    engine.add_component(Component("crac_2", "CRAC Unit 2 (Row 3-4)", ComponentType.CRAC, 500, 350, age_years=4, row=3))
    engine.add_component(Component("crac_3", "CRAC Unit 3 (Row 5-6)", ComponentType.CRAC, 500, 470, age_years=7, row=5, mtbf_hours=60000))
    engine.add_component(Component("crac_4", "CRAC Unit 4 (Row 7-8)", ComponentType.CRAC, 500, 300, age_years=3, row=7))

    # === NETWORK INFRASTRUCTURE ===
    engine.add_component(Component("core_sw", "Core Switch", ComponentType.CORE_SWITCH, 50, 35, age_years=3))
    engine.add_component(Component("agg_sw_1", "Aggregation Switch 1", ComponentType.AGG_SWITCH, 25, 18, age_years=2, row=1))
    engine.add_component(Component("agg_sw_2", "Aggregation Switch 2", ComponentType.AGG_SWITCH, 25, 20, age_years=2, row=5))

    # === RACKS (8 rows × 5 racks each = 40 racks) ===
    for row in range(1, 9):
        for pos in range(1, 6):
            rack_id = f"rack_r{row}_p{pos}"
            # AI-ready racks in rows 5-6 (higher density, some with capacity)
            if row in [5, 6]:
                if pos <= 3:  # first 3 positions loaded with AI
                    load = np.random.uniform(30, 45)
                else:  # last 2 positions available for new AI deployments
                    load = np.random.uniform(5, 10)  # light workload, ready for upgrade
                capacity = 50.0
            # Rows 7-8: expansion-ready (high capacity racks, moderate load)
            elif row in [7, 8]:
                load = np.random.uniform(5, 12)
                capacity = 40.0  # new infrastructure, AI-ready
            else:
                load = np.random.uniform(6, 15)  # Standard density
                capacity = 20.0
            
            engine.add_component(Component(
                rack_id, f"Rack R{row}-P{pos}", ComponentType.RACK,
                capacity, round(load, 1),
                age_years=np.random.uniform(1, 5),
                row=row, position=pos,
                thermal_mass_kwh_per_c=0.3 if row in [5,6] else 0.5,
                max_temp_c=32.0 if row in [5,6] else 35.0
            ))

    # === WIRE UP DEPENDENCIES ===
    
    # Power: Utility → Transformer → UPS → PDUs → Racks
    engine.add_dependency("util_a", "xfmr_a", EdgeType.POWER_FEEDS, 2500)
    engine.add_dependency("util_b", "xfmr_b", EdgeType.POWER_FEEDS, 2500)
    engine.add_dependency("xfmr_a", "ups_a", EdgeType.POWER_FEEDS, 2000)
    engine.add_dependency("xfmr_b", "ups_b", EdgeType.POWER_FEEDS, 2000)
    
    # UPS A feeds PDU 1,2 | UPS B feeds PDU 3,4
    engine.add_dependency("ups_a", "pdu_1", EdgeType.POWER_FEEDS, 600)
    engine.add_dependency("ups_a", "pdu_2", EdgeType.POWER_FEEDS, 600)
    engine.add_dependency("ups_b", "pdu_3", EdgeType.POWER_FEEDS, 600)
    engine.add_dependency("ups_b", "pdu_4", EdgeType.POWER_FEEDS, 600)
    
    # PDUs feed racks
    for row in range(1, 9):
        if row <= 2:
            pdu = "pdu_1"
        elif row <= 4:
            pdu = "pdu_2"
        elif row <= 6:
            pdu = "pdu_3"
        else:
            pdu = "pdu_4"
        for pos in range(1, 6):
            rack_id = f"rack_r{row}_p{pos}"
            engine.add_dependency(pdu, rack_id, EdgeType.POWER_FEEDS, 20)

    # Cooling: Chiller → CRACs → Racks (by zone)
    engine.add_dependency("chiller_1", "crac_1", EdgeType.POWER_FEEDS, 500)
    engine.add_dependency("chiller_1", "crac_2", EdgeType.POWER_FEEDS, 500)
    engine.add_dependency("chiller_1", "crac_3", EdgeType.POWER_FEEDS, 500)
    engine.add_dependency("chiller_1", "crac_4", EdgeType.POWER_FEEDS, 500)
    
    for row in range(1, 9):
        if row <= 2:
            crac = "crac_1"
        elif row <= 4:
            crac = "crac_2"
        elif row <= 6:
            crac = "crac_3"
        else:
            crac = "crac_4"
        for pos in range(1, 6):
            rack_id = f"rack_r{row}_p{pos}"
            engine.add_dependency(crac, rack_id, EdgeType.THERMAL_SERVES, 50)

    # Network: Core → Agg → Racks
    engine.add_dependency("core_sw", "agg_sw_1", EdgeType.NETWORK_CONNECTS, 25)
    engine.add_dependency("core_sw", "agg_sw_2", EdgeType.NETWORK_CONNECTS, 25)
    
    for row in range(1, 9):
        agg = "agg_sw_1" if row <= 4 else "agg_sw_2"
        for pos in range(1, 6):
            rack_id = f"rack_r{row}_p{pos}"
            engine.add_dependency(agg, rack_id, EdgeType.NETWORK_CONNECTS, 5)

    # Redundancy: UPS B protects UPS A path
    engine.add_dependency("ups_b", "ups_a", EdgeType.PROTECTS, 0)

    return engine
