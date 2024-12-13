"""
Simulation Management Module

This module defines classes and utilities for managing long-running simulation processes.
It handles the creation, monitoring, and termination of simulations executed through external system commands.

Classes:
    SimulationThread (threading.Thread): A class for managing individual simulation processes.
        It inherits from threading.Thread and overrides the run() and stop() methods to handle simulation executions.

    SimulationManager: A singleton class that maintains a registry of all active simulation threads and their
        associated system processes.

The module also includes the instantiation of a global SimulationManager singleton that manages all simulation threads
and processes. This manager is designed to be accessed and manipulated directly by the main Flask application to control
and monitor the state of simulations.
"""

# Import system modules
import os
import signal
import subprocess
import threading
import time


# Manages the execution of a simulation process
class SimulationThread(threading.Thread):
    """
    A thread class for managing the execution of a simulation process.

    This class is designed to handle the lifecycle of a simulation process initiated via an external command.
    It extends the standard threading.Thread class, providing functionalities to start, monitor, and stop the
    process as needed. The thread captures and logs the output, manages process termination gracefully, and
    updates the simulation status in a persistent storage.

    Attributes:
        cmd (str): Command line string to execute the simulation.
        simulation_id (str): Identifier for the simulation, used for logging and management.
        process (subprocess.Popen): The subprocess object once the process is started.
        process_id (str): System process ID for the simulation process.
        stopped (bool): Flag indicating whether the thread has been externally stopped.
        lock (threading.Lock): A lock object to ensure thread-safe operations.
        ready_event (threading.Event): An event to signal the readiness of the process ID.

    Methods:
        run(): Main method to be executed by the thread. Handles the execution of the simulation process,
               monitors its output, and updates the process state.
        stop(): Stops the simulation process, ensuring all resources are released properly and the process is terminated gracefully.

    Example:
        >>> simulation_thread = SimulationThread("path/to/simulation --args", "12345", "/var/logs/simulations/")
        >>> simulation_thread.start()  # Start the simulation in a separate thread
        >>> simulation_thread.stop()   # Stop the simulation, if needed
    """

    def __init__(self, cmd, simulation_id, log_path):
        super().__init__()
        self.cmd = cmd
        self.log_path = log_path
        self.process = None
        self.process_id = None
        self.simulation_id = simulation_id
        self.stopped = False
        self.lock = threading.Lock()
        self.ready_event = threading.Event()

    def run(self):
        print(f"Starting simulation thread for Simulation ID: {self.simulation_id}")
        try:

            # Start timing
            start_time = time.time()

            # Info
            print(f"Executing command: {self.cmd}")

            # Launch the simulation process
            self.process = subprocess.Popen(
                self.cmd,
                shell=True,
                executable='/bin/bash',
                stdin=None,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=self.set_process_attributes
            )

            # Save process into storage class
            self.process_id = str(self.process.pid)
            simulation_manager.add_process(self.process_id, self.process, self)
            self.ready_event.set()  # Signal that the process_id is now set
            print(f"Process ID {self.process_id} added to SimulationManager")

            # Wait for the command to complete
            raw, error = self.process.communicate()
            print(f"Process {self.process_id} has completed")

            # Compute CPU time
            end_time = time.time()
            elapsed_time = end_time - start_time
            cpu_time = "{:.2f}".format(elapsed_time)
            print(f"Simulation {self.simulation_id} ran for {cpu_time} seconds")

            # Decode the output byte streams
            raw_decoded = raw.decode(encoding='utf-8', errors='replace') if raw else ""
            error_decoded = error.decode(encoding='utf-8', errors='replace') if error else ""

            # Write std/err output streams to file
            stdout_path = os.path.join(self.log_path, "output.txt")
            stderr_path = os.path.join(self.log_path, "error.txt")
            with open(stdout_path, 'w', encoding='utf-8', errors='replace') as out_file, \
                 open(stderr_path, 'w', encoding='utf-8', errors='replace') as err_file:
                out_file.write(raw_decoded)
                err_file.write(error_decoded)
            print(f"Simulation {self.simulation_id} outputs written to {stdout_path} and {stderr_path}")

            # Determine simulation status
            if self.stopped:
                print(f"Simulation {self.simulation_id} was canceled")
            elif self.process.returncode == 0:
                print(f"Simulation {self.simulation_id} completed successfully")
            else:
                print(f"Simulation {self.simulation_id} encountered an error with return code {self.process.returncode}")

        except Exception as e:
            print(f"Unhandled exception in simulation thread for Simulation ID: {self.simulation_id}: {str(e)}")

        finally:
            # Ensure process is removed from simulation_manager
            if self.process_id:
                simulation_manager.remove_process(self.process_id)
                print(f"Process ID {self.process_id} removed from SimulationManager")
            print(f"Simulation thread for Simulation ID: {self.simulation_id} has terminated")


    def stop(self):
        print(f"Stopping simulation thread for Simulation ID: {self.simulation_id}")
        with self.lock:
            self.stopped = True
            if self.process and self.process.poll() is None:
                try:
                    # Send SIGTERM to the process group to politely request termination
                    os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                    print(f"Sent SIGTERM to process group of PID: {self.process.pid}")
                    # Wait for the process to terminate
                    self.process.wait(timeout=3)  # Wait up to 3 seconds
                    print(f"Process PID: {self.process.pid} terminated gracefully")
                except subprocess.TimeoutExpired:
                    # Force kill the process
                    os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                    print(f"Process PID: {self.process.pid} did not terminate after SIGTERM, sent SIGKILL")
                except Exception as e:
                    print(f"Exception occurred while stopping process PID: {self.process.pid}: {str(e)}")
            else:
                print(f"No active process to stop for Simulation ID: {self.simulation_id}")

    @staticmethod
    def set_process_attributes():
        # Start a new session
        os.setsid()
        # Set the niceness level to 15 (low-priority)
        os.nice(15)


# Manages the lifecycle and tracking of all simulation threads and processes
class SimulationManager:
    """
    A singleton class that manages the lifecycle and tracking of all simulation threads and processes.

    This class ensures that all simulation processes are managed in a centralized manner, providing methods to add,
    retrieve, and remove processes and their associated threads. The singleton pattern ensures that only one instance
    of this manager exists within the application, preventing conflicts and providing a single point of control for
    all simulations.

    Attributes:
        _instance (SimulationManager): The singleton instance of the SimulationManager.
        _lock (threading.Lock): A class-level lock to ensure thread-safe instantiation of the singleton.
        processes (dict): A dictionary mapping process IDs to subprocess.Popen objects.
        threads (dict): A dictionary mapping process IDs to SimulationThread objects.
        internal_lock (threading.Lock): An object-level lock for managing state changes safely across multiple threads.

    Methods:
        __new__(cls): Overrides the standard __new__ method to ensure that only one instance of the class is created.
        add_process(process_id, process, thread): Registers a new simulation process and its corresponding thread.
        remove_process(process_id): Removes a simulation process and its thread from the management tracking.
        get_process(process_id): Retrieves the process and thread associated with a given process ID.

    Usage Example:
        >>> sim_manager = SimulationManager()
        >>> sim_thread = SimulationThread("simulate --example", "001", "/logs/")
        >>> sim_thread.start()
        >>> sim_manager.add_process("001", sim_thread.process, sim_thread)
        >>> process, thread = sim_manager.get_process("001")
        >>> sim_manager.remove_process("001")
    """

    # Global variables
    _instance = None
    _lock = threading.Lock()  # Class-level lock to ensure singleton access is thread-safe

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(SimulationManager, cls).__new__(cls)
                cls._instance.processes = {}
                cls._instance.threads = {}
                cls._instance.internal_lock = threading.Lock()  # Object-level lock for managing state
            return cls._instance

    def add_process(self, process_id, process, thread):
        with self.internal_lock:
            self.processes[process_id] = process
            self.threads[process_id] = thread
            print(f"Process {process_id} added to SimulationManager.")

    def remove_process(self, process_id):
        with self.internal_lock:
            if process_id in self.processes:
                del self.processes[process_id]
                print(f"Process {process_id} removed from SimulationManager.")
            if process_id in self.threads:
                del self.threads[process_id]

    def get_process(self, process_id):
        with self.internal_lock:
            return self.processes.get(process_id), self.threads.get(process_id)


# Global SimulationManager singleton instance
simulation_manager = SimulationManager()
