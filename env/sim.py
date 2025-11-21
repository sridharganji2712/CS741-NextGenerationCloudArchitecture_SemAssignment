import simpy
from leaf.infrastructure import Node
from leaf.power import PowerModelNode, PowerMeter

def make_env(M, H, Fmax_u, B, seed=0):
    """
    LEAF requires either:
    - static_power only, OR
    - power_per_cu only

    So here we use a simple linear power model:
        power = static_power + power_per_cu * load
    """

    env = simpy.Environment()

    # UAV: static 30W + 1e-9 W per CPU cycle/s (example scalable model)
    uav_power_model = PowerModelNode(
        static_power=30.0,
        power_per_cu=1e-9
    )
    uav_node = Node("UAV", cu=Fmax_u, power_model=uav_power_model)

    # Users: static 1W + 5e-10 W per CPU cycle/s
    user_nodes = []
    for i in range(M):
        ud_power = PowerModelNode(
            static_power=1.0,
            power_per_cu=5e-10
        )
        user_nodes.append(Node(f"UD-{i}", cu=1e9, power_model=ud_power))

    # Attach power meters
    PowerMeter(uav_node)
    for n in user_nodes:
        PowerMeter(n)

    return env, uav_node, user_nodes
