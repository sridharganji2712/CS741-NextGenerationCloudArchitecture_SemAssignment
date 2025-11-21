# ojoa.py
import numpy as np
import cvxpy as cp

from uav_mec_ojoa_leaf.models.channel import (
    los_probability, spectral_efficiency, communication_rate,
    phi_term, distance_3d
)
from uav_mec_ojoa_leaf.models.uav_energy import propulsion_energy


class OJOAController:
    """
    OJOA controller implementing:
      - Stage-1: resource allocation and offloading game (best-response)
      - Stage-2: SCA trajectory planning (CVXPY when available; greedy fallback)
      - Lyapunov virtual queues (Qc, Qp) updated in apply_and_step

    Public methods:
      - decide(t, Pu, Pm_xy, fUD, D, eta, Tmax) -> decisions dict
      - apply_and_step(decisions, Pu, Pm_xy, fUD, D, eta, Tmax) -> results dict
    """

    def __init__(self, M, H, B, Fmax_u, vmax_u, tau,
                 beta0, mu_tilde, xi1, xi2, kappa,
                 Pm, N0, k, varpi, gamma, Ebar_u, V, seed=0):
        # system sizes & parameters
        self.M = int(M)
        self.H = float(H)
        self.B = float(B)
        self.Fmax_u = float(Fmax_u)
        self.vmax_u = float(vmax_u)
        self.tau = float(tau)

        # channel & radio params
        self.beta0 = float(beta0)
        self.mu_tilde = float(mu_tilde)
        self.mu = float(mu_tilde) / 2.0
        self.xi1 = float(xi1)
        self.xi2 = float(xi2)
        self.kappa = float(kappa)

        # powers / noise
        self.Pm = float(Pm)
        self.N0 = float(N0)

        # local device cpu model constant (used for local energy)
        self.k = float(k)

        # UAV compute energy per-cycle (J/cycle)
        self.varpi = float(varpi)

        # user cost weights
        self.gamma = np.array(gamma, dtype=float)

        # Lyapunov + energy budgets
        self.Ebar_u = float(Ebar_u)
        self.Ebar_c = 0.4 * self.Ebar_u
        self.Ebar_p = 0.6 * self.Ebar_u

        self.V = float(V)
        # virtual queue initial values
        self.Qc = 0.0
        self.Qp = 0.0

        # RNG (if needed later)
        self.rng = np.random.default_rng(seed)

    # -------------------------
    # Stage-1: Resource allocation
    # -------------------------
    def _alloc_resources(self, A, D, eta, Pu, Pm_xy):
        """
        Given offloading set A (binary vector), compute resource fractions S (compute),
        W (bandwidth fraction) and per-user rates R (bits/s) for offloaded users.

        Uses the closed-form ratios (Theorem 2 in paper) heuristic used previously.
        Returns S, W, R arrays of length M.
        """
        idx = np.where(A == 1)[0]
        if len(idx) == 0:
            return np.zeros(self.M), np.zeros(self.M), np.zeros(self.M)

        # distances and pathloss
        d2d, d3d = distance_3d(Pu, Pm_xy[idx], self.H)
        P_los = los_probability(self.H, d3d, self.xi1, self.xi2)
        P_tilde = P_los + (1.0 - P_los) * self.kappa

        phi = phi_term(self.Pm, self.beta0, P_tilde, self.N0)
        r = spectral_efficiency(phi, self.H, d2d, self.mu)  # spectral efficiency

        # Resource allocation (closed-form like Theorem 2)
        # Avoid division by zero
        sqrt1 = np.sqrt(np.maximum(self.gamma[idx] * eta[idx] * D[idx], 1e-15))
        if np.sum(sqrt1) <= 0:
            s = np.ones_like(sqrt1) / len(sqrt1)
        else:
            s = sqrt1 / np.sum(sqrt1)

        sqrt2 = np.sqrt(np.maximum(self.gamma[idx] * D[idx] + (1.0 - self.gamma[idx]) * self.Pm * D[idx] / np.maximum(r, 1e-12), 1e-15))
        if np.sum(sqrt2) <= 0:
            w = np.ones_like(sqrt2) / len(sqrt2)
        else:
            w = sqrt2 / np.sum(sqrt2)

        S = np.zeros(self.M); S[idx] = s
        W = np.zeros(self.M); W[idx] = w
        R = np.zeros(self.M); R[idx] = communication_rate(w, self.B, r)
        return S, W, R

    # -------------------------
    # Local / Edge models
    # -------------------------
    def _local_latency(self, D, eta, f):
        return (eta * D) / np.maximum(f, 1.0)

    def _local_energy(self, f, T):
        # typical switched capacitance model: k * f^2 * (cycles) * (1/f) => k * f^2 * T? earlier used k*f^3*T
        # We keep the earlier local model consistent with what you've been using:
        return self.k * (f ** 3) * T

    def _edge_latency(self, D, eta, R, S):
        Ttx = D / np.maximum(R, 1e-12)
        Texe = (eta * D) / np.maximum(S * self.Fmax_u, 1.0)
        return Ttx + Texe

    def _user_tx_energy(self, D, R):
        return self.Pm * (D / np.maximum(R, 1e-12))

    def _uav_comp_energy(self, eta, D):
        """
        CORRECTION: use varpi (J per CPU-cycle) times cycles executed on UAV.
        Paper (Eq. 11) uses E_{c m,u}(t) = varpi * eta_m(t) * D_m(t).
        Return an array (if vectorized) or scalar when arrays passed.
        """
        # make it vectorized: if eta or D are arrays, returns array
        return self.varpi * (eta * D)

    # -------------------------
    # Offloading game (MU-TOG)
    # -------------------------
    def _offloading_game(self, Pu, Pm_xy, fUD, D, eta, Tmax, max_iters=20):
        """
        Iterative best-response style algorithm using closed-form allocations per profile.
        Starts with all-local (A zeros) and updates users asynchronously until no change.
        """
        A = np.zeros(self.M, dtype=int)
        changed = True
        it = 0
        while changed and it < max_iters:
            it += 1
            changed = False
            # compute allocations for current A
            S, W, R = self._alloc_resources(A, D, eta, Pu, Pm_xy)

            # asynchronous updates - iterate users
            for m in range(self.M):
                # local utility
                Tloc = self._local_latency(D[m], eta[m], fUD[m])
                Eloc = self._local_energy(fUD[m], Tloc)
                U_loc = self.gamma[m] * Tloc + (1.0 - self.gamma[m]) * Eloc

                # if m flips to edge, compute its effective allocations & utility
                A_tmp = A.copy()
                A_tmp[m] = 1
                S_tmp, W_tmp, R_tmp = self._alloc_resources(A_tmp, D, eta, Pu, Pm_xy)
                Tec = self._edge_latency(D[m], eta[m], R_tmp[m], S_tmp[m])

                if Tec > Tmax[m]:
                    U_edge = np.inf
                else:
                    E_ud_tx = self._user_tx_energy(D[m], R_tmp[m])
                    E_uav_comp = self._uav_comp_energy(eta[m], D[m])
                    U_edge = (self.Qc / max(self.V, 1e-12)) * E_uav_comp + self.gamma[m] * Tec + (1.0 - self.gamma[m]) * E_ud_tx

                best = 0 if U_loc <= U_edge else 1
                if best != A[m]:
                    A[m] = best
                    changed = True

        S, W, R = self._alloc_resources(A, D, eta, Pu, Pm_xy)
        return A, S, W, R

    # -------------------------
    # Stage-2: SCA trajectory planner (CVXPY)
    # -------------------------
    def _plan_trajectory_sca(self, Pu, A, W, D, eta, Pm_xy,
                             max_iters=20, eps=1e-3, verbose=False):
        """
        Solve P2' by SCA (Algorithm 2 in paper).
        Uses CVXPY to solve per-iteration convex surrogate problems.
        Falls back to greedy centroid step if solver fails or returns infeasible.
        """
        idx = np.where(A == 1)[0]
        if len(idx) == 0:
            return Pu.copy()

        # constants (paper)
        C1, C2, C3, C4 = 79.86, 88.63, 0.114, 0.0001
        U_tip = 120.0

        # initial local point
        p_l = Pu.copy()
        G_prev = 1e30

        # Precompute per-offloader constant factors for COMM term
        Cm_const = {}
        for m in idx:
            if W[m] <= 0:
                Cm_const[m] = 1e9
            else:
                Cm_const[m] = (self.gamma[m] * D[m] + (1.0 - self.gamma[m]) * self.Pm * D[m]) / (W[m] * self.B)

        for it in range(max_iters):
            # CVXPY variables
            pu = cp.Variable(2)
            y = cp.Variable(nonneg=True)
            z = cp.Variable(len(idx), nonneg=True)

            constraints = []
            # travel constraint
            constraints.append(cp.norm(pu - Pu, 2) <= self.vmax_u * self.tau)
            # small positive y
            constraints.append(y >= 1e-6)

            # Build linearized spectral-efficiency lower bounds g_lm(pu)
            for j, m in enumerate(idx):
                pm = Pm_xy[m]
                # previous squared distance & denom
                dist_prev = np.linalg.norm(p_l - pm) ** 2
                denom_prev = (self.H ** 2 + dist_prev)
                # estimate phi_m including P_tilde using los_probability at prev distance
                P_los_prev = los_probability(self.H, np.sqrt(dist_prev + 1e-12), self.xi1, self.xi2)
                P_tilde_prev = P_los_prev + (1.0 - P_los_prev) * self.kappa
                phi_m = self.Pm * self.beta0 * P_tilde_prev / max(self.N0, 1e-12)

                denom_prev_mu = denom_prev ** self.mu
                gm_prev = np.log2(1.0 + phi_m / (denom_prev_mu + 1e-30))

                # slope alpha per Theorem 7 (safe numeric)
                alpha = (self.mu * phi_m * np.log2(np.e)) / ((phi_m + denom_prev_mu) * denom_prev + 1e-30)

                # quadratic term = ||pu - pm||^2
                quad_term = cp.sum_squares(pu - pm)
                g_lm = gm_prev - alpha * (quad_term - dist_prev)

                # z_j <= g_lm  (convex constraint because g_lm is concave in pu)
                constraints.append(z[j] <= g_lm)
                constraints.append(z[j] >= 1e-6)

            # propulsion surrogate (convex surrogate)
            vu = cp.norm(pu - Pu, 2) / self.tau
            # Using convex-friendly terms: quadratic and cubic approximations
            prop_surrogate = self.Qp * (C1 * (1 + 3 * cp.square(vu) / (U_tip ** 2)) + C2 * y + C4 * cp.power(vu, 3)) * self.tau

            # simple positive constraint on y (already added).
            # Communication cost: Cm_const * inv_pos(z_j) (convex)
            comm_terms = []
            for j, m in enumerate(idx):
                comm_terms.append(self.V * Cm_const[m] * cp.inv_pos(z[j]))

            objective = cp.Minimize(cp.sum(comm_terms) + prop_surrogate)
            prob = cp.Problem(objective, constraints)

            # Solve robustly
            solved = False
            try:
                prob.solve(solver=cp.OSQP, warm_start=True, verbose=False, eps_abs=1e-4)
                solved = (pu.value is not None)
            except Exception:
                solved = False

            if not solved:
                try:
                    prob.solve(solver=cp.SCS, verbose=False)
                    solved = (pu.value is not None)
                except Exception:
                    solved = False

            if not solved or pu.value is None:
                # fallback greedy step toward weighted centroid
                weights = np.array([ (self.gamma[m]*D[m] + (1-self.gamma[m])*self.Pm*D[m]) for m in idx ])
                if weights.sum() <= 0:
                    centroid = p_l
                else:
                    centroid = (weights[:, None] * Pm_xy[idx]).sum(axis=0) / np.maximum(weights.sum(), 1e-12)
                direction = centroid - Pu
                norm = np.linalg.norm(direction) + 1e-12
                dist = min(self.vmax_u * self.tau, norm)
                if norm < 1e-9:
                    return Pu.copy()
                return Pu + (direction / norm) * dist

            Pu_new = np.array(pu.value)
            G_curr = prob.value if prob.value is not None else G_prev

            if verbose:
                print(f"SCA iter {it}: obj={G_curr:.6e}")

            if abs(G_prev - G_curr) < eps:
                return Pu_new

            G_prev = G_curr
            p_l = Pu_new.copy()

        # reached max_iters
        return p_l

    # -------------------------
    # Public control methods
    # -------------------------
    def decide(self, t, Pu, Pm_xy, fUD, D, eta, Tmax):
        """
        For slot t with current UAV location Pu and user info, return decisions:
        {A, S, W, R, Pu_next}
        """
        A, S, W, R = self._offloading_game(Pu, Pm_xy, fUD, D, eta, Tmax)
        Pu_next = self._plan_trajectory_sca(Pu, A, W, D, eta, Pm_xy)
        return {"A": A, "S": S, "W": W, "R": R, "Pu_next": Pu_next}

    def apply_and_step(self, decisions, Pu, Pm_xy, fUD, D, eta, Tmax):
        """
        Apply the decisions, compute per-slot metrics, update virtual queues.
        Returns a dict with keys:
          Cm, Cs, Eu, Ec_u, Ep_u, uav_workload_cycles, Pu_next
        """
        A = decisions["A"]
        S = decisions["S"]
        W = decisions["W"]
        R = decisions["R"]
        Pu_next = decisions["Pu_next"]

        # UDs: local/edge latencies & energies
        Tloc = self._local_latency(D, eta, fUD)
        Eloc = self._local_energy(fUD, Tloc)
        Tec = self._edge_latency(D, eta, R, S)
        Eud_tx = self._user_tx_energy(D, R)

        Tm = (1 - A) * Tloc + A * Tec
        Em = (1 - A) * Eloc + A * Eud_tx
        Cm = self.gamma * Tm + (1.0 - self.gamma) * Em

        # UAV compute energy (sum over offloaded tasks)
        # _uav_comp_energy returns per-task energy; multiply by indicator A
        Ec_per_task = self._uav_comp_energy(eta, D)
        Ec_u = np.sum(A * Ec_per_task)

        # UAV propulsion energy
        vu = np.linalg.norm(Pu_next - Pu) / max(self.tau, 1e-12)
        Ep_u = propulsion_energy(vu, self.tau)

        Eu = Ec_u + Ep_u
        Cs = np.sum(Cm)
        uav_workload_cycles = np.sum(A * eta * D)

        # update virtual queues
        self.Qc = max(self.Qc + Ec_u - self.Ebar_c, 0.0)
        self.Qp = max(self.Qp + Ep_u - self.Ebar_p, 0.0)

        return {
            "Cm": Cm,
            "Cs": Cs,
            "Eu": Eu,
            "Ec_u": Ec_u,
            "Ep_u": Ep_u,
            "uav_workload_cycles": uav_workload_cycles,
            "Pu_next": Pu_next
        }
