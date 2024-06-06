.. _modules:

Modules
=======

Overview
--------

Modules are a way of changing the default behaviour of Mercury via the implementation of new events, roles, agents, etc.

Modules can be defined by adding a folder to the "modules" directory (but default ``modules``, but can be changed in the
config file). The name chosen for this folder (e.g. XMAN) can then we used
for adding it to the list of modules that are loaded at runtime. In general at least three files are required for a module:

- a main description of the new roles, events, and agents, included in a file named as its module
  (e.g. ``XMAN.py``).
- a toml file describing the changes that will be implemented by the simulator based on the previous file. This file
  needs to be called "X.toml" where X is the name of the module (e.g. ``XMAN.toml``).
- a toml file including all the new parameters introduced by the module and their default values. This file should be
  called "paras_X.toml" where X is the name of the module (e.g. ``paras_XMAN.toml``).

The module can then be loaded by adding it to the list of modules used in a simulation, via the scenario parameter
``paras.modules.modules_to_load``, present in the scenario and case study parameter files. Note that choosing the modules
from the CLI does not work at the moment, you need to insert the change via the parameter in the file directly, or use the Mercury
object programmatically and set the parameter for instance like this:

.. code:: python

    paras_sc_fixed = {
                'modules__modules_to_load':['XMAN']
                }

See more information in the :ref:`notebook` section to know how to use the Mercury object.

Note however that all the parameters defined by the modules are accessible via the CLI or the Mercury object. For
instance, a parameter called "optimisation_horizon" defined by the module XMAN can be set like this:

.. code:: bash

    ./mercury.py -id -1 -XMAN__optimisation_horizon 500

or even iterated over by using several values, e.g. ``-XMAN__optimisation_horizon 500 600 700``. The same behaviour is
available with the Mercury object.

Note: the XMAN dummy module can be found in ``modules/XMAN``.
Note: for now, adding new agents or new role is not supported by the module workflow.


Philosophy of module definition in Mercury
------------------------------------------

Mercury uses a very specific pythonic feature to allow for very flexible modifications of its base code. Indeed, instead
of class inheritance, modules use dynamic method assignment. For instance, imagine that you have the following basic
python class describing a role in the AMAN:

.. code:: python

    class FlightInAMANHandler(Role):
        def notify_flight_in_execution_horizon(self, flight_uid):
            msg = Letter()
            msg['to'] = self.agent.uid
            msg['type'] = 'flight_at_execution_horizon'
            msg['body'] = {'flight_uid': flight_uid}

            self.agent.atp.wait_for_flight_in_execution_horizon(msg)

        def wait_for_flight_in_eaman(self, msg):
            flight_uid = msg['body']['flight_uid']

            self.notify_flight_in_execution_horizon(flight_uid)


This role simply waits for a flight to cross the boundary of the Arrival Manager and notifies another role (``self.atp``)
when this happens. Now imagine that you want to modify this rule to add another horizon where you'll perform some other type of
optimisation. The role could now look like this:

.. code:: python

    class FlightInAMANHandlerNEW(Role):
        def notify_flight_in_execution_horizon(self, flight_uid):
            msg = Letter()
            msg['to'] = self.agent.uid
            msg['type'] = 'flight_at_execution_horizon'
            msg['body'] = {'flight_uid': flight_uid}

            self.agent.atp.wait_for_flight_in_execution_horizon(msg)

        def notify_flight_in_optimisation_horizon(self, flight_uid):
            msg = Letter()
            msg['to'] = self.agent.uid
            msg['type'] = 'flight_at_optimisation_horizon'
            msg['body'] = {'flight_uid': flight_uid}

            self.agent.atp.wait_for_flight_in_optimisation_horizon(msg)

        def wait_for_flight_in_eaman(self, msg):
            update = msg['body']['update_id']
            flight_uid = msg['body']['flight_uid']

            if update == "execution_horizon":
                self.notify_flight_in_execution_horizon(flight_uid)
            elif update == "optimisation_horizon":
                self.notify_flight_in_optimisation_horizon(flight_uid)

How to replace the old class by the old one? One possibility would be to make a child class for the new one, inheriting
from the old one. However, this way of modifying the old classes is potentially problematic when several modules try
to modify the same classes. Multiple inheritance is complex by nature, and in this case is painful to do, because authors
of modules should be aware of the modifications introduced by others. Moreover, modules should in general be used
independently from each other, and thus some classes could potentially inherit from modules that are not active in the
specific run.

We thus use a different strategy, by assigning methods dynamically at runtime to roles. For instance, in this example,
we would first define the new or modified methods outside of any class:


.. code:: python

    def wait_for_flight_in_eaman(self, msg):
        update = msg['body']['update_id']
        flight_uid = msg['body']['flight_uid']

        if update == "execution_horizon":
            self.notify_flight_in_execution_horizon(flight_uid)
        elif update == "optimisation_horizon":
            self.notify_flight_in_optimisation_horizon(flight_uid)

    def notify_flight_in_optimisation_horizon(self, flight_uid):
        msg = Letter()
        msg['to'] = self.agent.uid
        msg['type'] = 'flight_at_optimisation_horizon'
        msg['body'] = {'flight_uid': flight_uid}

        self.agent.atp.wait_for_flight_in_optimisation_horizon(msg)

Note that these are `functions`, not methods, but that we use the name ``self`` for the first argument to make it look
like they are methods, which they will be at runtime. Note also that there is no need to rewrite a function that has not been
modified (here the ``notify_flight_in_execution_horizon``), exactly like we would do for an inheritance.

Now at runtime we do something like this:

.. code:: python

    role = FlightInAMANHandler()
    role.wait_for_flight_in_eaman = wait_for_flight_in_eaman
    role.notify_flight_in_optimisation_horizon = notify_flight_in_optimisation_horizon

i.e., we add the method to the instance of the role. After that, the role has all the required methods as they were
defined from scratch. This has several added benefits compared to inheritance:

- it's easier to check for incompatibilities among modules, i.e. we can check beforehand if several modules modify the
  same methods.
- modules that modify the same classes but not the same methods do not need to care about each other, i.e. they don't 
  inherit one from another.
- it's marginally easier to modify specific agents, since modifications are done after instantiation. For instance, if
  we want to add our new role only to the agent corresponding to Rome Fiumiccino, it is easier to do that after the AMAN for
  Rome has been instantiated.


This method however requires that the module creator tells to Mercury which method should be attached to which classes.


How to define the module
------------------------

The first step in defining a module is to write the new methods in a file (the ``XMAN.py`` for instance), as explained
above. The second step is to create the toml file that allows assignment of methods to classes. In our example above, this
file could look this:

.. cod.e:: toml

    [info]
    name = "XMAN" # This is the name of the module
    description = "Dummy module" # Short description, just for info
    incompatibilities = [] # known incompatibilities with other modules
    requirements = []  # required modules to run this one
    get_metric = 'None' # method to gather information during simulation.

    [agent_modif] # Information on modifications of existing agents and role
        [agent_modif.EAMAN] # Top level is the agent
        on_init = 'None' # allows to run a method when agent is initiated.
        apply_to = [] # allows to modify only some instances of the agents, for instance only an airport
        new_parameters = [
            "optimisation_horizon",
        ] # list all the new parameters introduced by the module.

            [agent_modif.EAMAN.FlightInAMANHandler] # we want to attach the methods to this agent
            on_init = 'None' # you can also run something when the role is created.
            wait_for_flight_in_eaman = "wait_for_flight_in_eaman" # on the left is the name that the method will have,
                                                                  # on the right is the name as defined in the python file
            new_methods = [
                "notify_flight_in_optimisation_horizon",
            ] # lists all new methods to be attached to this class.

            # in this case we would need also to add another method to the "atp" (short for "ArrivalTacticalProvider")
            # role that receives the new notification, like this
            [agent_modif.EAMAN.ArrivalTacticalProvider] # we want to attach the methods to this agent
            on_init = 'None' # you can also run something when the role is created.
            new_methods = [
                "notify_flight_in_optimisation_horizon",
            ] # lists all new methods to be attached to this class.

In this case, we have modified the existing ``wait_for_flight_in_eaman`` function with the new one, and added the new method
``notify_flight_in_optimisation_horizon`` for the ``FlightInAMANHandler`` of the ``AMAN`` agent. We would also need
to add a new function to the ``ArrivalTacticalProvider`` that is supposed to do to something with the new optimisation
horizon. One can also add something to run during the initialisation of the agent.
For instance, in our example we'll probably need to add the value of the new horizon to the agent, for instance adding
this to our ``XMAN.py`` file:

.. code:: python

    def on_init_agent(self):
        self.optimisation_horizon = self.XMAN__optimisation_horizon

and modifying the corresponding line in the toml file to:

.. code:: toml

    [agent_modif.EAMAN] # Top level is the agent
    on_init = "on_init_agent" # allows to run a method when agent is initiated.

We could also add some new attributes to the role during initialisation of the ArrivalTacticalProvider role, for instance
writing this down in the XMAN.py file:

.. code:: python

    def on_init_ArrivalTacticalProvider(self):
        self.optimiser = 'basic'

and modifying the toml file accordingly:

.. code:: toml

    [agent_modif.EAMAN.ArrivalTacticalProvider]
    on_init = "on_init_ArrivalTacticalProvider" # allows to run a method when agent is initiated.

The information given at the top of the toml file have two additional important bits:

- ``incompatibilities``: list of modules that are known to be incompatible. Mercury will raise an
  error if this module is loaded.
- ``requirements``: list of modules that are required for the new module to run.

(Warning: as of v3.1, incompatibilities and requirements are not properly checked).

Finally, the module creator can use a custom method (``get_mnetric``) to gather important metrics for final analysis.
This method should always have one argument, which is the world builder. Metrics can be gathered from the world builder
at the end of the simulation. In our example, we could for instance record the number of flights that cross our
new optimisation horizon. We could modify the ``wait_for_flight_in_eaman`` method like this:

.. code:: python

    def wait_for_flight_in_eaman(self, msg):
        update = msg['body']['update_id']
        flight_uid = msg['body']['flight_uid']

        if update == "execution_horizon":
            self.notify_flight_in_execution_horizon(flight_uid)

        elif update == "optimisation_horizon":
            self.notify_flight_in_optimisation_horizon(flight_uid)
            self.recorded_flights += 1

and we need to add an on_init method to initialise the ``recorded_flights`` attribute (and modify the toml file
accordingly):

.. code:: python

    def on_init_FlightInAMANHandler(self):
        self.recorded_flights = 0


We can finally write our ``get_metric`` function in the XMAN.py file to gather the metrics at the end:

.. code:: python

    def get_metric(world_builder):
        # Create a new dataframe attached to the world builder.
        world_builder.df_xman = pd.DataFrame()
        world_builder.df_xman['n_entry_optimisation'] = [aman.recorded_flights for aman in world_builder.amans]
        world_builder.df_xman['airports'] = [aman.airport_uid for aman in world_builder.amans]

        world_builder.df_xman['scenario_id'] = world_builder.sc.paras['scenario']
		world_builder.df_xman['n_iter'] = world_builder.n_iter
		world_builder.df_xman['model_version'] = model_version

        # Add df_xman to the list of metrics to get for consolidated dataframes when running several iterations.
        world_builder.metrics_from_module_to_get = list(set(getattr(world_builder, 'metrics_from_module_to_get', [])\
                                                         + [('df_xman', 'global')]))

This method HAS to be called ``get_metric``, and there can be only one per module.
The corresponding line of the toml file should then be changed to:

.. code:: toml

    get_metric = "get_metric_XMAN"


The final step for our module definition is to setup a parameter file ``paras_XMAN.toml``. This parameter file is similar
to the ones defined for mercury and the scenario. In our case, we could have the following parameter file:

.. code:: toml

    [paras]
    optimisation_horizon = 800 # In NM.
    optimiser = 'basic'



Module flavour
--------------

Sometime one may write several versions of a module that share a lot in common. Because it's not
practical to write several independent modules in this case, one can also use "flavours" within the same module.
Flavours are detected by the module manager by using an underscore ``_`` in their file name. For instance,
we could write another flavour of our previous module by creating a new file in the ``XMAN`` folder called
``XMAN_data``. In this case the flavour would be "data". The content of this file could be similar to the initial module, with for instance an another additional
horizon, called the data horizon. The modifications to the methods could be as follows (only new of modifieed methods
are shown):

.. code:: python

    def wait_for_flight_in_eaman(self, msg):
        update = msg['body']['update_id']
        flight_uid = msg['body']['flight_uid']

        if update == "execution_horizon":
            self.notify_flight_in_execution_horizon(flight_uid)
        elif update == "optimisation_horizon":
            self.notify_flight_in_optimisation_horizon(flight_uid)
            self.recorded_flights += 1
        elif update == "optimisation_horizon":
            self.notify_flight_in_data_horizon(flight_uid)

    def on_init_agent(self):
        self.optimisation_horizon = self.XMAN__optimisation_horizon
        self.data_horizon = self.XMAN__data_horizon

    def notify_flight_in_data_horizon(self, flight_uid):
        msg = Letter()
        msg['to'] = self.agent.uid
        msg['type'] = 'flight_at_odata_horizon'
        msg['body'] = {'flight_uid': flight_uid}

        self.agent.atp.wait_for_flight_in_data_horizon(msg)

A new toml file called ``XMAN_data.toml`` should then be created and it should look like:

.. code:: toml

    [info]
    name = "XMAN" # This is the name of the module
    description = "Dummy module" # Short description, just for info
    incompatibilities = [] # known incompatibilities with other modules
    requirements = []  # required modules to run this one
    get_metric = "get_metric" # method to gather information during simulation.

    [agent_modif] # Information on modifications of existing agents and role
        [agent_modif.AMAN] # Top level is the agent
        on_init = "on_init_agent" # allows to run a method when agent is initiated.
        apply_to = [] # allows to modify only some instances of the agents, for instance only an airport
        new_parameters = [
            "optimisation_horizon",
            "data_horizon",
        ] # list all the new parameters introduced by the module.

            [agent_modif.AMAN.FlightInAMANHandler] # we want to attach the methods to this agent
            on_init = "on_init_FlightInAMANHandler" # you can also run something when the role is created.
            wait_for_flight_in_eaman = "wait_for_flight_in_eaman" # on the left is the name that the method will have,
                                                                  # on the right is the name as defined in the python file
            new_methods = [
                "notify_flight_in_optimisation_horizon",
                "notify_flight_in_data_horizon",
            ] # lists all new methods to be attached to this class.

            # in this case we would need also to add another method to the "atp" (short for "ArrivalTacticalProvider")
            # role that receives the new notification, like this
            [agent_modif.AMAN.ArrivalTacticalProvider] # we want to attach the methods to this agent
            on_init = "on_init_ArrivalTacticalProvider" # you can also run something when the role is created.
            new_methods = [
                "notify_flight_in_optimisation_horizon",
                "notify_flight_in_data_horizon",
            ] # lists all new methods to be attached to this class.

And we need a new file paras_XMAN_data.py that includes the new parameter:

.. code:: toml

    [paras]
    optimisation_horizon = 800 # In NM.
    data_horizon = 1200 # In NM.
    optimiser = 'basic'

We can then use the flavour by adding the new module flavour to the scenario config file, with the following syntax:

.. code:: toml

    [paras.modules]
    modules_to_load = ["XMAN|data"]

Flavours are indicated with a ``|``, i.e. if a module "XXX|YYY" is to be loaded, Mercury will look for ``XXX_YYY.py``,
``XXX_YYY.toml``, and ``paras_XXX_YYY.toml`` files inside a ``XXX`` folder.

Finally, note that it is not required to add the flavour when calling some parameters from the CLI or elsewhere. For
instance:

.. code:: bash

    ./mercury.py -id -1 -cs -1 --XMAN__data_horizon 500 600

will work even if the ``XMAN|data`` flavour is used



