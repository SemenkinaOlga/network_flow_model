import argparse
import os
import json
import pandas as pd
import numpy as np
from ortools.graph.python import min_cost_flow


def parse_parameters():
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--durations_file_path', help='a durations matrix, which encodes the driving duration '
                                                            'in seconds between each pair of locations')
    parser.add_argument('-r', '--requests_file_path', help='a .CSV file containing ride requests for a specific day, '
                                                           'ordered by the requested pickup timestamp, along with their'
                                                           ' requested pickup and dropoff locations')
    parser.add_argument('-va', '--vehicles_amount', help='available vehicles amount')

    return parser.parse_args()


def prepare_data(durations_file_path, requests_file_path):
    """
    Read CSV and extract data
    Arguments:
        durations_file_path: durations CSV file path
        requests_file_path: requests CSV file path
    Returns:
        durations, requests extracted data in dictionary format
    """
    durations_df = pd.read_csv(durations_file_path)

    # Prepare durations dataframe for transformation: cleaning data
    durations_df = durations_df.rename(columns={'Unnamed: 0': 'index'})
    durations_df['index'] = durations_df['index'].replace({'from_': ''}, regex=True)
    durations_df.columns = [x.replace("to_", "") for x in durations_df.columns]
    durations_df['index'] = durations_df['index'].astype(int)
    durations_df = durations_df.set_index('index')
    durations_df.columns = durations_df.columns.astype(int)

    # Transform durations dataframe to a dictionary
    durations_df = durations_df.T
    durations = durations_df.to_dict()

    requests_df = pd.read_csv(requests_file_path)

    # Prepare requests dataframe to transformation
    requests_df.rename(columns={'Unnamed: 0': 'id'}, inplace=True)
    requests_df = requests_df.set_index('id')

    # Transform requests dataframe to a dictionary
    requests_df_T = requests_df.T
    requests = requests_df_T.to_dict()

    return durations, requests


class GlobalNodeCounter:

    def __init__(self, first_value):
        self.value = first_value

    def get_next_node(self):
        self.value = self.value + 1
        return self.value


def setup_depot_to_request(requests, durations, model, globalNodeCounter):
    """
    Setup nodes and edges from depot to all request pickup points in the model
    Arguments:
        requests: requests data dictionary
        durations: duration between nodes dictionary
        model: model for processing (see the FlowNetworkModel class)
        globalNodeCounter: object for creating unique node numbers while generating a graph
        (see the GlobalNodeCounter class)
    """
    for request_id in requests:
        model.start_nodes.append(model.first_node)
        pickup = requests[request_id]['pickup']
        # Create a new node for unique identification
        pickup_node = globalNodeCounter.get_next_node()
        # Make it the demand node
        model.supplies.append(-1)
        model.end_nodes.append(pickup_node)
        # For future processing: make a link between pickup & dropoff node (used in the setup_dropoff_nodes function)
        requests[request_id]['pickup_node'] = pickup_node
        model.unit_costs.append(durations[0][pickup])


def setup_dropoff_nodes(requests, globalNodeCounter, model):
    """
    Create unique node for each request dropoff location in the model
    Arguments:
        requests: requests data dictionary
        globalNodeCounter: object for creating unique node numbers while generating a graph
        (see the GlobalNodeCounter class)
        model: model for processing (see the FlowNetworkModel class)
    Returns:
        arc_to_request: dictionary to identify request id by graph edge
    """
    arc_to_request = {}

    for request_id in requests:
        pickup_node = requests[request_id]['pickup_node']

        dropoff_node = globalNodeCounter.get_next_node()
        # Make it the supply node
        model.supplies.append(1)
        requests[request_id]['dropoff_node'] = dropoff_node

        # create edge to request link (used in the decode_result function)
        if pickup_node not in arc_to_request:
            arc_to_request[pickup_node] = {}

        arc_to_request[pickup_node]['dropoff_node'] = dropoff_node
        arc_to_request[pickup_node]['id'] = request_id

    return arc_to_request


def setup_dropoffs_to_depot(requests, durations, model):
    """
    Create edges between request dropoff modes to depot node in the model
    Arguments:
        requests: requests data dictionary
        durations: duration between nodes dictionary
        model: model for processing (see the FlowNetworkModel class)
    """
    for request_id in requests:
        dropoff = requests[request_id]['dropoff']
        dropoff_node = requests[request_id]['dropoff_node']
        model.start_nodes.append(dropoff_node)
        model.end_nodes.append(model.last_node)
        # durations[dropoff][0] - ride duration from dropoff to depot
        model.unit_costs.append(durations[dropoff][0])


def setup_request_to_request(requests, durations, model):
    """
    Create edges between different requests if exist
    Arguments:
        requests: requests data dictionary
        durations: duration between nodes dictionary
        model: model for processing (see the FlowNetworkModel class)
    """
    for first_request_id in requests:
        for second_request_id in requests:
            if first_request_id != second_request_id:
                first_pickup = requests[first_request_id]['pickup']
                first_dropoff = requests[first_request_id]['dropoff']
                # dropoff time == pickup time + duration
                first_dropoff_timestamp = requests[first_request_id]['ts'] + durations[first_pickup][first_dropoff]

                second_pickup = requests[second_request_id]['pickup']
                # check if we have enough time to drive to the second request
                if first_dropoff_timestamp + durations[first_dropoff][second_pickup] \
                        <= requests[second_request_id]['ts']:
                    model.start_nodes.append(requests[first_request_id]['dropoff_node'])
                    model.end_nodes.append(requests[second_request_id]['pickup_node'])
                    # Cost between 2 requests is ride duration + waiting time. It's equal to time difference between
                    # dropoff time of first request and pickup time of second request
                    model.unit_costs.append(requests[second_request_id]['ts'] - first_dropoff_timestamp)


class FlowNetworkModel:
    """
        first_node: unique node number for depot in the start of graph (always equals 0)
        start_nodes: all start nodes of all edges in graph
        end_nodes: all end nodes of all edges in graph
        unit_costs: edge using cost in graph for all edges
        supplies: the supply at each node (a positive amount for supply node, a negative amount for demand node)
        capacities: capacities of all graph edges
        last_node: unique node number for depot in the end of graph
    """

    def __init__(self, vehicles_amount):
        self.first_node = 0
        self.start_nodes = []
        self.end_nodes = []
        self.unit_costs = []
        self.supplies = [vehicles_amount]
        self.capacities = []
        self.last_node = -1

    def setup_capacities(self):
        # Each edge has a capacity equals 1 because one and only one vehicle must be assigned per request
        self.capacities = [1] * len(self.unit_costs)

    def setup_last_node(self, globalNodeCounter):
        self.last_node = globalNodeCounter.get_next_node()


def prepare_model(durations, requests, vehicles_amount):
    """
    Creation of data in the suitable format for the ortools algorithm SimpleMinCostFlow
    Arguments:
        durations: duration between nodes dictionary
        requests: requests data dictionary
        vehicles_amount: amount of available vehicles
    Returns:
        model: Flow Network Model (see the FlowNetworkModel class)
        arc_to_request: dictionary to identify request id by graph edge
    """
    model = FlowNetworkModel(vehicles_amount)

    globalNodeCounter = GlobalNodeCounter(model.first_node)
    setup_depot_to_request(requests, durations, model, globalNodeCounter)
    arc_to_request = setup_dropoff_nodes(requests, globalNodeCounter, model)
    model.setup_last_node(globalNodeCounter)
    setup_dropoffs_to_depot(requests, durations, model)
    setup_request_to_request(requests, durations, model)
    model.setup_capacities()

    return model, arc_to_request


def run_solver(model, vehicles_amount):
    """
    Run SimpleMinCostFlow algorithm to resolve the problem
    Arguments:
        model: model with data (see the FlowNetworkModel class)
        vehicles_amount: amount of available vehicles
    Returns:
        simple_min_cost_flow: instance of SimpleMinCostFlow, contains the solution
        all_arcs: all arcs (or edges) of graph (used in the decode_result function)
    """
    start_nodes = np.array(model.start_nodes)
    end_nodes = np.array(model.end_nodes)
    capacities = np.array(model.capacities)
    unit_costs = np.array(model.unit_costs)
    supplies = model.supplies + [-vehicles_amount]

    # Instantiate a SimpleMinCostFlow solver
    simple_min_cost_flow = min_cost_flow.SimpleMinCostFlow()

    # Add arcs, capacities and costs
    all_arcs = simple_min_cost_flow.add_arcs_with_capacity_and_unit_cost(start_nodes, end_nodes, capacities, unit_costs)

    # Add supply for each node
    simple_min_cost_flow.set_nodes_supplies(np.arange(0, len(supplies)), supplies)

    # Find the min cost flow
    status = simple_min_cost_flow.solve()

    if status != simple_min_cost_flow.OPTIMAL:
        print('There was an issue with the min cost flow input.')
        print(f'Status: {status}')
        exit(1)

    return simple_min_cost_flow, all_arcs


def calculate_total_cost(requests, durations, simple_min_cost_flow):
    """
    Calculate total cost based on the algorithm solution and request durations
    Arguments:
        durations: duration between nodes dictionary
        requests: requests data dictionary
        simple_min_cost_flow: instance of SimpleMinCostFlow, contains the solution
    Returns:
        total_costs: sum of time of all ride that is needed to serve requests
    """
    total_costs = simple_min_cost_flow.optimal_cost()

    # Add ride costs to total cost
    for request_id in requests:
        pickup = requests[request_id]['pickup']
        dropoff = requests[request_id]['dropoff']
        total_costs = total_costs + durations[pickup][dropoff]

        return total_costs


def move_to_next_node(node, arc_to_request, vehicle_rides, name, trips):
    """
    Move forward through the graph to find the next ride
    Arguments:
        node: current pickup node
        arc_to_request: dictionary to identify request id by graph edge
        vehicle_rides: dictionary of all request rides that served by each vehicle
        name: current vehicle name
        trips: all chosen edges in graph
    Returns:
        next_node: next pickup node or depot
    """
    current_pickup = node
    current_dropoff = arc_to_request[current_pickup]['dropoff_node']
    vehicle_rides[name] = vehicle_rides[name] + ['ride_' + str(arc_to_request[current_pickup]['id'])]
    next_node = trips[current_dropoff]
    return next_node


def decode_result(simple_min_cost_flow, all_arcs, model, arc_to_request):
    """
    Identify rides for all cars
    Arguments:
        simple_min_cost_flow: instance of SimpleMinCostFlow, contains the solution
        all_arcs: all graph arcs (or edges)
        model: model with data (see the FlowNetworkModel class)
        arc_to_request: dictionary to identify request id by graph edge
    Returns:
        vehicle_rides: result dictionary with rides for all vehicles
    """
    # Create dictionary of edges used by any car
    trips = {}
    # All start edges for cars to identify their rides
    new_starts = []
    solution_flows = simple_min_cost_flow.flows(all_arcs)
    for arc, flow in zip(all_arcs, solution_flows):
        if flow:
            head = simple_min_cost_flow.head(arc)
            tail = simple_min_cost_flow.tail(arc)
            # check if it is a start depot
            if tail == model.first_node:
                new_starts.append(head)
            # check if it is an end depot
            elif head == model.last_node:
                trips[tail] = 0
            else:
                trips[tail] = head

    count = 0
    vehicle_rides = {}
    for current_node in new_starts:
        # For all cars
        name = 'vehicle_' + str(count)
        vehicle_rides[name] = []
        count = count + 1

        next_node = current_node
        # Build current ride until meet end depot
        while next_node != 0:
            next_node = move_to_next_node(next_node, arc_to_request, vehicle_rides, name, trips)

    return vehicle_rides


def read_parameters(path):
    """
    Read command line arguments
    Returns:
        durations_file_path, requests_file_path, vehicles_amount extracted from arguments or default values
    """
    args = parse_parameters()
    durations_file_path = args.durations_file_path
    requests_file_path = args.requests_file_path
    vehicles_amount = args.vehicles_amount

    if args.durations_file_path is None:
        durations_file_path = path + '\\' + 'durations.csv'
        print('durations_file_path is not defined, set default value: ' + durations_file_path)

    if args.requests_file_path is None:
        requests_file_path = path + '\\' + 'requests.csv'
        print('requests_file_path is not defined, set default value: ' + requests_file_path)

    if args.vehicles_amount is None:
        vehicles_amount = 30
        print('vehicles_amount is not defined, set default value: ' + str(vehicles_amount))

    return durations_file_path, requests_file_path, vehicles_amount


def main():
    path = os.getcwd()
    durations_file_path, requests_file_path, vehicles_amount = read_parameters(path)

    durations, requests = prepare_data(durations_file_path, requests_file_path)

    model, arc_to_request = prepare_model(durations, requests, vehicles_amount)

    simple_min_cost_flow, all_arcs = run_solver(model, vehicles_amount)

    total_costs = calculate_total_cost(requests, durations, simple_min_cost_flow)
    result = decode_result(simple_min_cost_flow, all_arcs, model, arc_to_request)

    print("Total cost: " + str(total_costs))
    print("Rides: ")
    print(result)

    # Write to json file
    data = {'total_costs': total_costs, 'rides': result}
    result_path = path + '\\' + 'result.json'
    json_string = json.dumps(data)
    with open(result_path, 'w') as outfile:
        outfile.write(json_string)

    return [total_costs, result]


if __name__ == '__main__':
    main()
