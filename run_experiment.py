# run_experiment.py
import os
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import your modules (assumes package uav_mec_ojoa_leaf is importable)
from uav_mec_ojoa_leaf.env.sim import make_env
from uav_mec_ojoa_leaf.models.mobility import GaussMarkovMobility
from uav_mec_ojoa_leaf.tasks.generator import TaskGenerator
from uav_mec_ojoa_leaf.algo.ojoa import OJOAController

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def save_metrics_csv(results_df, out_dir):
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(out_dir, f"metrics_{ts}.csv")
    results_df.to_csv(csv_path, index=False)
    return csv_path

def plot_timeseries(results_df, out_dir, prefix="ojoy"):
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1) Time-average UD cost (per-slot Cs)
    plt.figure(figsize=(8,4))
    plt.plot(results_df['slot'], results_df['Cs'], marker='o', linewidth=1)
    plt.xlabel('Time slot'); plt.ylabel('Total UD Cost (Cs)')
    plt.title('Time-slot UD Cost')
    plt.grid(True)
    p1 = os.path.join(out_dir, f"{prefix}_ud_cost_{ts}.png")
    plt.tight_layout(); plt.savefig(p1); plt.close()

    # 2) UAV energy (per-slot Eu) and components
    plt.figure(figsize=(8,4))
    plt.plot(results_df['slot'], results_df['Eu'], label='Eu (total)', marker='o', linewidth=1)
    if 'Ec_u' in results_df.columns:
        plt.plot(results_df['slot'], results_df['Ec_u'], label='Ec_u (compute)', marker='x', linewidth=1)
    if 'Ep_u' in results_df.columns:
        plt.plot(results_df['slot'], results_df['Ep_u'], label='Ep_u (propulsion)', marker='s', linewidth=1)
    plt.xlabel('Time slot'); plt.ylabel('Energy (J)')
    plt.title('UAV Energy per Slot')
    plt.legend(); plt.grid(True)
    p2 = os.path.join(out_dir, f"{prefix}_uav_energy_{ts}.png")
    plt.tight_layout(); plt.savefig(p2); plt.close()

    # 3) UAV workload (cycles) and Offload count
    plt.figure(figsize=(8,4))
    plt.plot(results_df['slot'], results_df['uav_workload_cycles'] / 1e9, label='UAV workload (Gcycles)', marker='o', linewidth=1)
    plt.xlabel('Time slot'); plt.ylabel('Workload (Gcycles)')
    plt.title('UAV Workload per Slot')
    plt.grid(True)
    p3 = os.path.join(out_dir, f"{prefix}_uav_workload_{ts}.png")
    plt.tight_layout(); plt.savefig(p3); plt.close()

    # 4) Offloading users per slot
    if 'offload_count' in results_df.columns:
        plt.figure(figsize=(8,4))
        plt.plot(results_df['slot'], results_df['offload_count'], marker='o', linewidth=1)
        plt.xlabel('Time slot'); plt.ylabel('Offloading users')
        plt.title('Number of Offloading Users per Slot')
        plt.grid(True)
        p4 = os.path.join(out_dir, f"{prefix}_offload_count_{ts}.png")
        plt.tight_layout(); plt.savefig(p4); plt.close()
    else:
        p4 = None

    # 5) UAV trajectory
    plt.figure(figsize=(6,6))
    plt.plot(results_df['uav_x'], results_df['uav_y'], marker='o', linewidth=1)
    plt.scatter(results_df['uav_x'].iloc[0], results_df['uav_y'].iloc[0], label='start', marker='D')
    plt.scatter(results_df['uav_x'].iloc[-1], results_df['uav_y'].iloc[-1], label='end', marker='X')
    plt.xlabel('x (m)'); plt.ylabel('y (m)')
    plt.title('UAV Trajectory')
    plt.legend(); plt.grid(True)
    p5 = os.path.join(out_dir, f"{prefix}_trajectory_{ts}.png")
    plt.tight_layout(); plt.savefig(p5); plt.close()

    saved = [p1, p2, p3, p4, p5]
    return [p for p in saved if p is not None]

def main():
    rng = np.random.default_rng(42)

    # Simulation parameters (tune as you wish)
    M = 20
    T = 40
    tau = 1.0
    area = (400.0, 400.0)
    H = 100.0
    vmax_u = 30.0
    Fmax_u = 20e9
    B = 4e6
    P_tx_user = 0.1
    beta0 = 1.0
    mu_tilde = 2.0
    k_switch = 1e-27
    varpi_uav = 1e-11
    N0 = 1e-13
    xi1, xi2 = 9.61, 0.16
    kappa = 0.1
    gamma_m = np.full(M, 0.5)

    # create env (your sim.make_env may ignore returned env obj, but keep for compatibility)
    env, uav_node, user_nodes = make_env(M=M, H=H, Fmax_u=Fmax_u, B=B, seed=7)

    # UD cpu (examples); choose from {1e9,1.5e9,2e9}
    fUD = rng.choice([1e9, 1.5e9, 2e9], size=M)

    # initial user positions
    users_xy = rng.uniform([0,0], area, size=(M,2))

    # mobility & task generator
    mobility = GaussMarkovMobility(alpha=0.8, v_mean=(0,0), sigma=1.0, tau=tau, area=area, rng=rng)
    v_init = rng.normal(0,1.0,size=(M,2))
    taskgen = TaskGenerator(rng=rng, D_range=(0.1e6, 1.0e6), eta_range=(500.0, 1500.0), Tmax=1.0)

    # controller
    controller = OJOAController(M=M,H=H,B=B,Fmax_u=Fmax_u,vmax_u=vmax_u,tau=tau,
        beta0=beta0,mu_tilde=mu_tilde,xi1=xi1,xi2=xi2,kappa=kappa,
        Pm=P_tx_user,N0=N0,k=k_switch,varpi=varpi_uav,gamma=gamma_m,
        Ebar_u=320.0,V=100.0, seed=123)

    # initial UAV pos
    Pu = np.array([200.0, 200.0])

    # logging containers
    records = []
    uav_positions = []
    offload_counts = []

    # simulate
    Pm_xy = users_xy.copy()
    v = v_init.copy()

    for t in range(T):
        D, eta, Tmax = taskgen.generate(M)
        decisions = controller.decide(t=t, Pu=Pu, Pm_xy=Pm_xy, fUD=fUD, D=D, eta=eta, Tmax=Tmax)

        res = controller.apply_and_step(decisions, Pu=Pu, Pm_xy=users_xy, fUD=fUD, D=D, eta=eta, Tmax=Tmax)


        # log per-slot
        offload_count = int(np.sum(decisions['A']))
        offload_counts.append(offload_count)
        uav_positions.append(Pu.copy())

        rec = {
            "slot": t,
            "Cs": float(np.sum(res.get("Cm", np.zeros(M)))),
            "Eu": float(res.get("Eu", np.nan)),
            "Ec_u": float(res.get("Ec_u", np.nan)),
            "Ep_u": float(res.get("Ep_u", np.nan)),
            "uav_workload_cycles": float(res.get("uav_workload_cycles", 0.0)),
            "offload_count": offload_count,
            "Qc": float(controller.Qc),
            "Qp": float(controller.Qp),
            "uav_x": float(Pu[0]),
            "uav_y": float(Pu[1])
        }
        records.append(rec)

        # step
        Pu = decisions["Pu_next"]
        Pm_xy, v = mobility.step(Pm_xy, v)

    # assemble dataframe
    df = pd.DataFrame.from_records(records)

    # create results dir
    out_dir = os.path.abspath("results")
    ensure_dir(out_dir)

    # save CSV
    csv_path = save_metrics_csv(df, out_dir)

    # save plots
    saved_plots = plot_timeseries(df, out_dir, prefix="OJOA")

    # final summary (print & save small text)
    summary = {
        "Total timeslots": T,
        "Average_UE_Cost": df["Cs"].mean(),
        "Average_UAV_Energy_per_slot": df["Eu"].mean(),
        "Average_UAV_Workload_cycles": df["uav_workload_cycles"].mean(),
        "Average_Offloading_Users_Per_Slot": df["offload_count"].mean(),
        "Final_Qc": controller.Qc,
        "Final_Qp": controller.Qp
    }

    print("\n==================== Simulation Summary ====================")
    for k,v in summary.items():
        print(f"{k}: {v}")
    print("\nSaved CSV:", csv_path)
    print("Saved plots:")
    for p in saved_plots:
        print(" ", p)
    print("Results folder:", out_dir)
    print("============================================================\n")

if __name__ == '__main__':
    main()
