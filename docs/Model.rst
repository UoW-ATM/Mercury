.. _model:

Model
=====

This is the presentation of the underlying model for Mercury (under construction).


Overview of the model
---------------------

Mercury has a strong agent implementation, with different autonomous instances interacting with each other through
messages. It is event-driven, with a central event engine driving the simulation. Agents can create, trigger, or destroy
events dynamically. The flow of the program is the following:

- an agent triggers an event, for instance, a flight departure,
- all agents listening to this event trigger internal actions,
- these actions may lead them to send messages to other agents, which might lead to another cascade of actions,
- once all messages and actions have been resolved for this event, the engine triggers the next event.

The actions of the agents are concurrent, i.e. in general they act independently. If they need some common
resources to perform the actions, a system of queues is used to solve which resources are currently used by which agent.
Note that if notionally these actions are independent, and happen in-simulation at the same time, the implementation of
Mercury is not multi-core or multi-threads. The Simpy library is used as the event engine, see
`here <https://simpy.readthedocs.io/en/latest/>`_ for more details.

The agents are instances of several agent types. The most important agent types are listed below:

- Airline Operating Center, tasked with managing the flights,
- Flight, tasked with operating the flight trajectory,
- Network Manager, managing Air Traffic Flow Management delays,
- Arrival Manager, taking care of inbound flights at each airport,
- Departure Manger, taking care of outbound traffic,
- Ground Airport, tasked with "moving" passengers inside the airport, and turnaround processes.

TBC
