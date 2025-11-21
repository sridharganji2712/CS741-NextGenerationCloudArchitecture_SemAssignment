OJOA-UAV-MEC: Online Joint Optimization Approach for QoE Maximization in UAV-Enabled MEC

This repository contains a complete Python-based implementation of the paper:
“An Online Joint Optimization Approach for QoE Maximization in UAV-Enabled Mobile Edge Computing.”

The system reproduces:
MU-TOG Stage-1 Offloading Game
SCA-based Stage-2 Trajectory Optimization (Convexified P2’)
Lyapunov-based long-term control for UAV energy and stability
Full simulation with mobility, task generation, communication, local/edge computation models
Evaluation against baseline algorithms
CSV export + PNG plots for all metrics


This implementation simulates the complete framework presented in the original paper and provides reproducible comparative evaluation.

1. Project Overview
   
The OJOA framework jointly optimizes:
Task offloading decisions
Communication and compute resource allocation
UAV trajectory
Energy-aware behavior through virtual queues
This implementation includes:
CVXPY-based SCA solver for UAV trajectory
Best-response MU-TOG game for offloading
Propulsion & computation energy models
Communication rate and probabilistic LoS channel modeling
Gauss-Markov user mobility
Data export and visualization tools


Baseline algorithms:
Local-Only
Edge-Only
Random-Offload
OJOA-NoTrajectory (control without SCA)
OJOA-Full (ours)

2. Experimental System Setup

The following hardware was used to run and verify experiments:
Host Machine
CPU: Intel Core i7-13700HX (16 cores)
RAM: 16 GB DDR5
Storage: 512 GB NVMe SSD
GPU: Nvidia RTX 3050 (6 GB VRAM)
OS: Windows 11

Python: 3.10+

CVXPY Solvers: OSQP, SCS

The implementation does not require the GPU, but the CPU performance significantly impacts CVXPY SCA iterations.

4. Algorithm Components Implemented
4.1 MU-TOG (Stage-1)
Each user solves a best-response offloading subproblem, considering:
Local execution latency & energy
Edge latency
UAV computational energy (varpi * cycles)
Lyapunov queue penalties (Qc, Qp)

4.2 Resource Allocation
Closed-form computation and communication resource ratios implemented from Theorem 2.
4.3 UAV Trajectory Optimization (Stage-2, SCA)
P2’ convexified using:
Linearized spectral-efficiency bounds (g_lm)
Convex propulsion surrogate (Eq. 15)
CVXPY formulation matching Algorithm 2 in the paper

4.4 Lyapunov Control
Virtual queues Qc and Qp updated:
Qc(t+1) = max(Qc + Ec_u - Ēc, 0)
Qp(t+1) = max(Qp + Ep_u - Ēp, 0)

5. Performance Summary vs Baselines
After integrating full offloading, resource allocation, and SCA trajectory optimization, the OJOA algorithm consistently outperforms:

Metric	Local	Edge-Only	Random	OJOA-NoTraj	OJOA-Full (Our Method)
Average UD Cost	Highest	Medium	Unstable	Lower	Lowest
UAV Compute Energy	0	High	Medium	Medium	Minimized via Lyapunov
UAV Propulsion Energy	—	—	—	Higher	Optimized via SCA
Offloading Ratio	Very Low	Very High	Random	Medium	Balanced & Adaptive
QoE Stability	Poor	Unstable	Poor	Medium	Best

The key reason OJOA performs superior is the joint optimization that couples:
Communication
Computation
Trajectory
Long-term energy management
This is not present in baselines.

6. Installation Instructions (Any New Machine)
6.1 Install Python

Install Python 3.10 or 3.11 from python.org.

6.2 Clone Repository
git clone https://github.com/yourusername/uav_mec_ojoa_leaf.git
cd uav_mec_ojoa_leaf

6.3 Create Virtual Environment
python -m venv .venv
.venv\Scripts\activate    # Windows

6.4 Install Dependencies
pip install numpy scipy pandas matplotlib
pip install cvxpy
pip install setuptools wheel


CVXPY optional solvers:
pip install osqp
pip install scs

7. Running Simulations
7.1 Run OJOA (single algorithm)
python -m uav_mec_ojoa_leaf.run_experiment


Outputs:

CSV file containing metrics
PNG plots:
UD cost
UAV energy
UAV workload
Offloading count
UAV trajectory
Saved in /results/.

7.2 Run All Baseline Comparisons
python -m uav_mec_ojoa_leaf.run_comparisons

Generates:
Comparison plots similar to the original paper
CSV files for each baseline
Combined visualization

8. Exported Data
Each run saves:
metrics_timestamp.csv
OJOA_*.png: all plots

For comparisons:
comparison_cs.png
comparison_energy.png
comparison_offload.png
comparison_trajectory.png
And more depending on configuration

9. Troubleshooting
ModuleNotFoundError: uav_mec_ojoa_leaf
You must run the script from project root:
python -m uav_mec_ojoa_leaf.run_experiment

CVXPY solver warnings
These are normal if OSQP fails; code automatically falls back to SCS.

Plots not showing
Plots are saved into /results/ and not displayed interactively.
