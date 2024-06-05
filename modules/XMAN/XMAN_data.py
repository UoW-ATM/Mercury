import pandas as pd

from Mercury.core.delivery_system import Letter

# =========== Modification of AMAN agent =========== #
def on_init_agent(self):
    self.optimisation_horizon = self.XMAN__optimisation_horizon
    self.data_horizon = self.XMAN__data_horizon

# =========== Modification of FlightInAMANHandler Role =========== #
def on_init_FlightInAMANHandler(self):
    self.recorded_flights = 0


def wait_for_flight_in_eaman(self, msg):
    update = msg['body']['update_id']
    flight_uid = msg['body']['flight_uid']

    if update == "execution_horizon":
        self.notify_flight_in_execution_horizon(flight_uid)
        print('USING NEW METHOD ON FLIGHTS {} at horizons {} and {}'.format(flight_uid,
                                                                    self.agent.optimisation_horizon,
                                                                     self.agent.data_horizon))
    elif update == "optimisation_horizon":
        self.notify_flight_in_optimisation_horizon(flight_uid)
        self.recorded_flights += 1
    elif update == "optimisation_horizon":
        self.notify_flight_in_data_horizon(flight_uid)


def notify_flight_in_optimisation_horizon(self, flight_uid):
    msg = Letter()
    msg['to'] = self.agent.uid
    msg['type'] = 'flight_at_optimisation_horizon'
    msg['body'] = {'flight_uid': flight_uid}

    self.agent.atp.wait_for_flight_in_optimisation_horizon(msg)


def notify_flight_in_data_horizon(self, flight_uid):
    msg = Letter()
    msg['to'] = self.agent.uid
    msg['type'] = 'flight_at_odata_horizon'
    msg['body'] = {'flight_uid': flight_uid}

    self.agent.atp.wait_for_flight_in_data_horizon(msg)


# =========== Modification of ArrivalTacticalProvider Role =========== #
def on_init_ArrivalTacticalProvider(self):
    self.optimiser = 'basic'


def notify_flight_in_optimisation_horizon(self, flight_uid):
    pass


def notify_flight_in_data_horizon(self, flight_uid):
    pass


# =========== Get metrics =========== #
def get_metric(world_builder):
    # Create a new dataframe attached to the world builder.
    world_builder.df_xman = pd.DataFrame()
    world_builder.df_xman['n_entry_optimisation'] = [aman.recorded_flights for aman in world_builder.amans]
    world_builder.df_xman['airports'] = [aman.airport_uid for aman in world_builder.amans]

    world_builder.df_xman['scenario_id'] = world_builder.sc.paras['scenario']
    world_builder.df_xman['n_iter'] = world_builder.n_iter
    world_builder.df_xman['model_version'] = world_builder.model_version

    # Add df_xman to the list of metrics to get for consolidated dataframes when running several iterations.
    world_builder.metrics_from_module_to_get = list(set(getattr(world_builder, 'metrics_from_module_to_get', [])\
                                                     + [('df_xman', 'global')]))



