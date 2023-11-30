"""
.. include:: README.md
"""

from Mercury.core.simulation_manager import Mercury
from Mercury.core.read_config import read_scenario_config, read_mercury_config
from Mercury.core.parametriser import ParametriserSelector
from Mercury.core.results_aggregator import ResultsAggregatorSelector
from Mercury.libs.uow_tool_belt.connection_tools import mysql_connection, read_data, write_data, file_connection, generic_connection

__all__ = ['core', 'agents', 'dashboard', 'modules', 'mercury', 'libs']
