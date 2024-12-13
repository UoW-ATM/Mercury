# Import system modules
import os

# Import local modules
from sim_manager import SimulationThread

# Main
if __name__ == "__main__":

    # Directories
    dynamo_release_dir = "/icarus/code/mercury/Mercury/libs/dynamo3/releases/dynamo3_v3.7_20241210205013"
    dynamo_configs_dir = "/icarus/code/mercury/Mercury/libs/dynamo3/templates"
    dynamo_loggers_dir = "/icarus/code/mercury/Mercury/input/scenario=-1/case_studies/case_study=-1/logs"

    # Check if the dynamo release  exists
    if not os.path.exists(dynamo_release_dir):
        error_message = "Dynamo3 selected release file not found!"

    # Launch script path
    launcher_path = os.path.join(dynamo_release_dir, 'launch.sh')

    # Dynamo config path
    config_path = os.path.join(dynamo_configs_dir, 'config.json')

    # Solver
    solver = "c3po"

    # Assemble launch command
    cmd = "bash " + str(launcher_path) + " dynamo3 " + " --config " + config_path + " --solver " + solver

    # Simulations ID
    simu_id = "test"

    # Start the simulation in a separate thread
    simulation_thread = SimulationThread(cmd, simu_id, dynamo_loggers_dir)
    simulation_thread.start()

    # Efficiently wait for the process_id to be ready
    # Timeout [seconds] added to prevent indefinite wait
    simulation_thread_ready = simulation_thread.ready_event.wait(timeout=60)
    if not simulation_thread_ready:  # Somehow we exceeded waiting time
        print(f"Timeout waiting for simulation thread to set process_id for Simulation ID: {simu_id}")

    # Safely access the process_id
    process_id = simulation_thread.process_id
    print(f"Retrieved process ID: {process_id} for Simulation ID: {simu_id}")

