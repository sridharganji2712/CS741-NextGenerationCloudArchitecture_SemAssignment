# run_comparisons.py
import os
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy

# import the project modules (assumes package uav_mec_ojoa_leaf is importable)
from uav_mec_ojoa_leaf.env.sim import make_env
from uav_mec_ojoa_leaf.models.mobility import GaussMarkovMobility
from uav_mec_ojoa_leaf.tasks.generator import TaskGenerator
from uav_mec_ojoa_leaf.algo.ojoa import OJOAController

# ---------------------------
# Utilities for saving/plots
# ---------------------------
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def save_csv(df, out_dir, name):
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(out_dir, f"{name}_{ts}.csv")
    df.to_csv(path, index=False)
    return path

def save_plot(fig, out_dir, name):
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(out_dir, f"{name}_{ts}.png")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path

# ---------------------------
# Baseline policies
# ---------------------------
def baseline_elc(controller, t, Pu, Pm_xy, fUD, D, eta, Tmax):
    """Entire local computing: all A=0, UAV stays or tracks nothing"""
    M = controller.M
    A = np.zeros(M, dtype=int)
    S = np.zeros(M)
    W = np.zeros(M)
    R = np.zeros(M)
    # simple Pu_next: stay
    Pu_next = Pu.copy()
    return {"A": A, "S": S, "W": W, "R": R, "Pu_next": Pu_next}

def baseline_era(controller, t, Pu, Pm_xy, fUD, D, eta, Tmax):
    """Equal Resource Allocation for offloaders with greedy offload decision (if edge better than local)"""
    M = controller.M
    # compute utility individually comparing local vs offload assuming equal share if all offload
    # We'll approximate: assume all users offload, compute equal S and W for those who would prefer offload.
    # Start with everyone deciding greedily comparing local and naive edge assuming equal split.
    A = np.zeros(M, dtype=int)
    # equal partition (if everyone offloads)
    # safe fallback: all local first
    for m in range(M):
        Tloc = controller._local_latency(D[m], eta[m], fUD[m])
        Eloc = controller._local_energy(fUD[m], Tloc)
        Uloc = controller.gamma[m] * Tloc + (1.0 - controller.gamma[m]) * Eloc
        # naive estimate of edge rate using current distance and full 1/M share
        # compute r_m using channel functions
    # Simple heuristic: compute closed-form S/W as in OJOA but applied to all
    A[:] = 1  # allow offload all then resource split equally
    S, W, R = controller._alloc_resources(A, D, eta, Pu, Pm_xy)
    # re-evaluate per-user: if actual edge latency > Tmax or not better, revert to local
    for m in range(M):
        Tec = controller._edge_latency(D[m], eta[m], R[m], S[m])
        if Tec > Tmax[m]:
            A[m] = 0
            S[m] = 0
            W[m] = 0
            R[m] = 0
        else:
            Tloc = controller._local_latency(D[m], eta[m], fUD[m])
            Eloc = controller._local_energy(fUD[m], Tloc)
            Uloc = controller.gamma[m] * Tloc + (1.0 - controller.gamma[m]) * Eloc
            Eud_tx = controller._user_tx_energy(D[m], R[m])
            E_uav_comp = controller._uav_comp_energy(eta[m], D[m])
            Uedge = controller.gamma[m] * Tec + (1.0 - controller.gamma[m]) * Eud_tx + (controller.Qc / max(controller.V,1e-12)) * E_uav_comp
            if Uedge >= Uloc:
                A[m] = 0
                S[m] = 0
                W[m] = 0
                R[m] = 0
    # simple Pu_next: move toward centroid of offloaders
    idx = np.where(A==1)[0]
    if len(idx)==0:
        Pu_next = Pu.copy()
    else:
        centroid = np.mean(Pm_xy[idx], axis=0)
        dirvec = centroid - Pu
        dist = np.linalg.norm(dirvec)
        step = min(controller.vmax_u * controller.tau, dist)
        Pu_next = Pu + (dirvec/dist)*step if dist>1e-9 else Pu.copy()
    return {"A": A, "S": S, "W": W, "R": R, "Pu_next": Pu_next}

def baseline_flp(controller, t, Pu, Pm_xy, fUD, D, eta, Tmax):
    """Fixed Location Deployment: UAV hovers over center of area (paper uses center)"""
    # determine center of service area via user positions bounding box
    # simpler: compute mean of user positions as center
    center = np.mean(Pm_xy, axis=0)
    # Keep UAV moving toward center but do not change offloading logic: use OJOA's offloading_game with Pu fixed at center
    Pu_center = center
    A, S, W, R = controller._offloading_game(Pu_center, Pm_xy, fUD, D, eta, Tmax)
    # Pu_next moves toward center (or stays)
    dirvec = Pu_center - Pu
    dist = np.linalg.norm(dirvec)
    step = min(controller.vmax_u * controller.tau, dist)
    Pu_next = Pu + (dirvec/dist)*step if dist>1e-9 else Pu_center.copy()
    return {"A": A, "S": S, "W": W, "R": R, "Pu_next": Pu_next}

def baseline_ocq(controller, t, Pu, Pm_xy, fUD, D, eta, Tmax):
    """OCQ: only consider QoE (ignore UAV energy constraint) -> set V small or ignore Qc/Qp in utility.
       We'll call OJOA's _offloading_game but temporarily set Qc=0 so offloading ignores UAV comp energy term.
    """
    old_Qc, old_Qp = controller.Qc, controller.Qp
    controller.Qc, controller.Qp = 0.0, 0.0
    A, S, W, R = controller._offloading_game(Pu, Pm_xy, fUD, D, eta, Tmax)
    # decide trajectory same as OJOA's centroid step (use S)
    idx = np.where(A==1)[0]
    if len(idx)==0:
        Pu_next = Pu.copy()
    else:
        weights = np.array([ (controller.gamma[m]*D[m] + (1-controller.gamma[m])*controller.Pm*D[m]) for m in idx ])
        centroid = (weights[:,None] * Pm_xy[idx]).sum(axis=0) / max(weights.sum(), 1e-12)
        dirvec = centroid - Pu
        dist = np.linalg.norm(dirvec)
        step = min(controller.vmax_u * controller.tau, dist)
        Pu_next = Pu + (dirvec/dist)*step if dist>1e-9 else Pu.copy()
    controller.Qc, controller.Qp = old_Qc, old_Qp
    return {"A": A, "S": S, "W": W, "R": R, "Pu_next": Pu_next}

# ---------------------------
# Single-policy simulator
# ---------------------------
def run_policy_sim(policy_name, controller, T=80, seed=42, D_range=(0.1e6,1.0e6)):
    rng = np.random.default_rng(seed)
    M = controller.M
    tau = controller.tau
    area = (400.0, 400.0)
    # env (not used heavily, but kept for compatibility)
    env, uav_node, user_nodes = make_env(M=M, H=controller.H, Fmax_u=controller.Fmax_u, B=controller.B, seed=seed)
    # initial positions
    users_xy = rng.uniform([0,0], area, size=(M,2))
    Pm_xy = users_xy.copy()
    v = rng.normal(0,1.0,size=(M,2))
    mobility = GaussMarkovMobility(alpha=0.8, v_mean=(0,0), sigma=1.0, tau=tau, area=area, rng=rng)
    fUD = rng.choice([1e9, 1.5e9, 2e9], size=M)
    taskgen = TaskGenerator(rng=rng, D_range=D_range, eta_range=(500.0,1500.0), Tmax=1.0)
    Pu = np.array([200.0,200.0])

    records = []
    for t in range(T):
        D, eta, Tmax = taskgen.generate(M)
        # choose policy
        if policy_name == "OJOA":
            decisions = controller.decide(t=t, Pu=Pu, Pm_xy=Pm_xy, fUD=fUD, D=D, eta=eta, Tmax=Tmax)
        elif policy_name == "ELC":
            decisions = baseline_elc(controller, t, Pu, Pm_xy, fUD, D, eta, Tmax)
        elif policy_name == "ERA":
            decisions = baseline_era(controller, t, Pu, Pm_xy, fUD, D, eta, Tmax)
        elif policy_name == "FLP":
            decisions = baseline_flp(controller, t, Pu, Pm_xy, fUD, D, eta, Tmax)
        elif policy_name == "OCQ":
            decisions = baseline_ocq(controller, t, Pu, Pm_xy, fUD, D, eta, Tmax)
        else:
            raise ValueError("Unknown policy")

        # apply_and_step: use controller.apply_and_step for metric computation (works for baselines as well)
        res = controller.apply_and_step(decisions, Pu=Pu, Pm_xy=Pm_xy, fUD=fUD, D=D, eta=eta, Tmax=Tmax)

        rec = {
            "slot": t,
            "Cs": float(res.get("Cs", np.sum(res.get("Cm", np.zeros(M))))),
            "Eu": float(res.get("Eu", np.nan)),
            "Ec_u": float(res.get("Ec_u", np.nan)),
            "Ep_u": float(res.get("Ep_u", np.nan)),
            "uav_workload_cycles": float(res.get("uav_workload_cycles", 0.0)),
            "offload_count": int(np.sum(decisions.get("A", np.zeros(M)))),
            "Qc": float(controller.Qc),
            "Qp": float(controller.Qp),
            "uav_x": float(Pu[0]),
            "uav_y": float(Pu[1])
        }
        records.append(rec)

        # step mobility & update Pu
        Pu = decisions["Pu_next"]
        Pm_xy, v = mobility.step(Pm_xy, v)

    df = pd.DataFrame.from_records(records)
    return df

# ---------------------------
# Create comparison figures (paper-style)
# ---------------------------
def plot_figure2_time_series(results_dict, out_dir):
    # results_dict: {policy_name: df}
    # Make three subplots (a)(b)(c) like paper: UD cost, UAV energy, UAV workload
    policies = list(results_dict.keys())
    T = len(next(iter(results_dict.values())))
    t = np.arange(T)

    fig, axs = plt.subplots(1,3, figsize=(15,4))
    # (a) time-average UD cost per slot (Cs)
    for p in policies:
        axs[0].plot(t, results_dict[p]["Cs"].values, label=p)
    axs[0].set_title("Time-slot UD Cost"); axs[0].set_xlabel("Time slot"); axs[0].set_ylabel("Cost"); axs[0].grid(True)
    axs[0].legend()

    # (b) UAV energy per slot
    for p in policies:
        axs[1].plot(t, results_dict[p]["Eu"].values, label=p)
    axs[1].set_title("UAV Energy per Slot"); axs[1].set_xlabel("Time slot"); axs[1].set_ylabel("Energy (J)"); axs[1].grid(True)

    # (c) UAV workload (Gcycles)
    for p in policies:
        axs[2].plot(t, results_dict[p]["uav_workload_cycles"].values/1e9, label=p)
    axs[2].set_title("UAV Workload per Slot"); axs[2].set_xlabel("Time slot"); axs[2].set_ylabel("Workload (Gcycles)"); axs[2].grid(True)

    path = save_plot(fig, out_dir, "Fig2_time_series")
    return path

def plot_figure3_data_size_sweep(results_sweep, data_sizes, out_dir):
    # results_sweep: dict of policy->list of dfs for each data size (same order as data_sizes)
    policies = list(results_sweep.keys())
    # compute metrics per data size: time-averaged Cs, Eu, workload
    avg_cost = {p: [] for p in policies}
    avg_energy = {p: [] for p in policies}
    avg_workload = {p: [] for p in policies}
    for p in policies:
        for df in results_sweep[p]:
            avg_cost[p].append(df["Cs"].mean())
            avg_energy[p].append(df["Eu"].mean())
            avg_workload[p].append(df["uav_workload_cycles"].mean()/1e9)

    fig, axs = plt.subplots(1,3, figsize=(15,4))
    for p in policies:
        axs[0].plot(data_sizes, avg_cost[p], marker='o', label=p)
    axs[0].set_title("Time-average UD cost"); axs[0].set_xlabel("Data size (Mb)"); axs[0].set_ylabel("Cost"); axs[0].grid(True)

    for p in policies:
        axs[1].plot(data_sizes, avg_energy[p], marker='o', label=p)
    axs[1].set_title("Time-average UAV energy (J)"); axs[1].set_xlabel("Data size (Mb)"); axs[1].grid(True)

    for p in policies:
        axs[2].plot(data_sizes, avg_workload[p], marker='o', label=p)
    axs[2].set_title("Time-average UAV workload (Gcycles)"); axs[2].set_xlabel("Data size (Mb)"); axs[2].grid(True)

    axs[0].legend()
    path = save_plot(fig, out_dir, "Fig3_data_sweep")
    return path

# ---------------------------
# Main experiment orchestration
# ---------------------------
def main():
    out_dir = os.path.abspath("results")
    ensure_dir(out_dir)

    # Simulation config (adjust as you like)
    M = 20
    T = 80
    seed = 42
    policies = ["OJOA","ELC","ERA","FLP","OCQ"]

    # Initialize a fresh controller for each policy run (to avoid shared queue states)
    def make_controller():
        # parameters must match your run_experiment.py defaults
        gamma_m = np.full(M, 0.5)
        return OJOAController(M=M,H=100.0,B=4e6,Fmax_u=20e9,vmax_u=30.0,tau=1.0,
                              beta0=1.0,mu_tilde=2.0,xi1=9.61,xi2=0.16,kappa=0.1,
                              Pm=0.1,N0=1e-13,k=1e-27,varpi=1e-11,gamma=gamma_m,
                              Ebar_u=320.0,V=100.0, seed=seed)

    # 1) Figure 2: time series comparison (single data setting)
    results_time = {}
    for p in policies:
        ctrl = make_controller()
        print("Running policy:", p)
        df = run_policy_sim(p, ctrl, T=T, seed=seed, D_range=(0.1e6,1e6))
        results_time[p] = df
        # save per-policy csv
        save_csv(df, out_dir, f"time_{p}")

    fig2_path = plot_figure2_time_series(results_time, out_dir)
    print("Saved Fig2:", fig2_path)

    # 2) Figure 3: data size sweep
    data_sizes_mb = [0.2, 0.4, 0.6, 0.8, 1.0]  # Mb
    data_sizes_bits = [s*1e6 for s in data_sizes_mb]
    results_sweep = {p: [] for p in policies}
    for ds in data_sizes_bits:
        print("Data size (Mb):", ds/1e6)
        for p in policies:
            ctrl = make_controller()
            df = run_policy_sim(p, ctrl, T=T, seed=seed, D_range=(ds, ds))
            results_sweep[p].append(df)
            save_csv(df, out_dir, f"datasweep_{int(ds)}_{p}")

    fig3_path = plot_figure3_data_size_sweep(results_sweep, data_sizes_mb, out_dir)
    print("Saved Fig3:", fig3_path)

    print("All results saved into:", out_dir)

if __name__ == "__main__":
    main()
