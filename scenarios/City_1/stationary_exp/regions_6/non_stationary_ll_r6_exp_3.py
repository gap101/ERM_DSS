import copy
import pickle
from datetime import datetime
import time
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np

from Environment.DataStructures.State import State
from Environment.EnvironmentModel import EnvironmentModel
from Environment.Simulator import Simulator
from decision_making.HighLevel.MMCHighLevelPolicy import MMCHighLevelPolicy
from decision_making.LowLevel.CentralizedMCTS.DataStructures.LLEnums import MCTStypes
from decision_making.LowLevel.CentralizedMCTS.DecisionEnvironmentDynamics import DecisionEnvironmentDynamics
from decision_making.LowLevel.CentralizedMCTS.LowLevelCentMCTSPolicy import LowLevelCentMCTSPolicy
from decision_making.LowLevel.CentralizedMCTS.Rollout import DoNothingRollout
from decision_making.coordinator.DispatchOnlyCoord import DispatchOnlyCoord
from decision_making.coordinator.LowLevelCoordTest import LowLevelCoord
from decision_making.dispatch.SendNearestDispatchPolicy import SendNearestDispatchPolicy
from scenarios.City_1.definition.generate_start_state import StateGenerator
from Prediction.ChainPredictor.ChainPredictor import ChainPredictor
from Prediction.ChainPredictor.ChainPredictorReturnOne import ChainPredictorReturnOne
from scenarios.City_1.CellToCellTravelModel import CellToCellTravelModel
from scenarios.City_1.BetweenRegionTravelModel import BetweenRegionTravelModel
from scenarios.City_1.WithinRegionTravelModel import WithinRegionTravelModel

from Prediction.RealIncidentPredictor.RealIncidentPredictor import RealIncidentPredictor
from Prediction.ChainPredictor.ChainPredictor_Subset import ChainPredictor_Subset

from Environment.Spatial.spatialStructure import  get_cell_id_and_rates

def print_resp_info(state):
    for key, value in state.responders.items():
        print(key, value.__dict__)

def plot_resp_time_results(resp_time_dict):

    values = list(resp_time_dict.values())

    sns.boxplot(x=values)
    plt.show()


def get_resp_init_assignments_to_regions_and_depots(hl_policy,
                                                    region_ids,
                                                    curr_time,
                                                    region_to_cell_dict,
                                                    total_responders,
                                                    depots,
                                                    predictor):


    region_incident_rates = dict()
    for r in region_ids:
        region_incident_rates[r] = hl_policy.get_region_total_rate(None, region_to_cell_dict[r])

    region_depot_counts = dict()
    for region_id in region_ids:
        region_depot_counts[region_id] = len([_[0] for _ in depots.items() if
                                              _[1].cell_loc in region_to_cell_dict[region_id]])

    initial_resp_to_region_allocation = hl_policy.greedy_responder_allocation_algorithm(
        region_ids=region_ids,
        region_incident_rates=region_incident_rates,
        total_responders=total_responders,
        curr_time=curr_time,
        region_to_depot_count=region_depot_counts)

    region_to_depots = dict()
    for region_id in region_ids:
        region_to_depots[region_id] = {_[0]: _[1] for _ in depots.items() if
                                       _[1].cell_loc in region_to_cell_dict[region_id]}

    depot_cell_rates = dict()
    for region_id in region_ids:
        region_depot_rates = list()
        for region_depot in region_to_depots[region_id].values():
            depot_cell_rate = predictor.get_cell_rate(None, region_depot.cell_loc)
            region_depot_rates.append((region_depot.my_id, depot_cell_rate))

        region_depot_rates.sort(key=lambda _: _[1], reverse=True)
        depot_cell_rates[region_id] = region_depot_rates

    resp_assigned = 0
    resp_assignments = dict()
    resp_ids = list()
    for region_id, num_resp in initial_resp_to_region_allocation.items():
        sorted_depot_rates = depot_cell_rates[region_id]
        pos_in_depots = 0
        num_depots_in_region = len(sorted_depot_rates)
        for _i_resp in range(num_resp):
            if pos_in_depots % num_depots_in_region == 0:
                pos_in_depots = 0

            resp_id = str(resp_assigned)
            resp_assignments[resp_id] = {'region_id': region_id,
                                         'depot_id': sorted_depot_rates[pos_in_depots][0]}
            resp_ids.append(resp_id)

            resp_assigned += 1
            pos_in_depots += 1


    return resp_assignments


def create_high_level_policy(_wait_time_threshold,
                             _predictor,
                             region_centers,
                             _cell_travel_model,
                             region_mean_distance_dict,
                             travel_rate_cells_per_second,
                             mean_service_time,
                             region_ids):


    _between_region_travel_model = BetweenRegionTravelModel(region_centers=region_centers,
                                                           cell_travel_model=_cell_travel_model)

    _within_region_travel_model = WithinRegionTravelModel(mean_dist_for_region_dict=region_mean_distance_dict,
                                                          travel_cells_per_second=travel_rate_cells_per_second)

    # TODO
    class test_incident_service_time_model:
        def __init__(self, region_ids, mean_service_time):
            self.model = dict()
            for id in region_ids:
                self.model[id] = mean_service_time  # mean service time (NOT RATE) in seconds - 20 minutes?

        def get_expected_service_time(self, region_id, curr_time):

            # ignore time for now
            return self.model[region_id]

    service_time_model = test_incident_service_time_model(region_ids=region_ids,
                                                          mean_service_time=mean_service_time)

    hl_policy = MMCHighLevelPolicy(prediction_model=_predictor,
                                   within_region_travel_model=_within_region_travel_model,
                                   trigger_wait_time_threshold=_wait_time_threshold,
                                   incident_service_time_model=service_time_model,
                                   between_region_travel_model=_between_region_travel_model)

    return hl_policy


if __name__ == "__main__":

    ###############################################
    ## constants across runs
    # config info
    num_rows = 30
    num_columns = 30
    num_resp = 26
    travel_rate_cells_per_second = 30.0 / 3600.0  # cells are 1x1 miles; 30 mph
    wait_time_threshold = 10 * 60  # 10 minutes
    mean_service_time = 20 * 60

    min_time_between_allocation = 60 * 60 * 1.0
    lookahead_horizon_delta_t = 60 * 60 * 1.5
    pool_thread_count = 6
    mcts_type = MCTStypes.CENT_MCTS
    mcts_discount_factor = 0.99995
    uct_tradeoff = 1.44
    iter_limit = 1500  # None
    allowed_computation_time = None  #10

    #########################################################
    #### run specific
    data_directory = '../data/'
    ##########################################################
    incident_df_path = data_directory + 'cityMVA_ETrims_Jan2018_April2019.json'
    depot_info_path = data_directory + 'testing/depots_with_cells.json'
    structural_valid_cells = pickle.load(open(data_directory + 'valid_cells_out.pkl', 'rb'))
    preprocessed_chains_file_path = data_directory + 'chain_data/stationary_chains_60_v1.pickle'

    start_time = datetime(year=2019, month=1, day=1).timestamp()
    end_time = datetime(year=2019, month=1, day=11).timestamp()

    structural_valid_cells = [str(_) for _ in structural_valid_cells]

    config_info = dict()
    config_info['iter_lim'] = iter_limit
    config_info['discount_factor'] = mcts_discount_factor
    config_info['uct_tradeoff'] = uct_tradeoff
    config_info['lookahead_horizon'] = lookahead_horizon_delta_t
    config_info['start_time'] = start_time
    config_info['end_time'] = end_time

    state_generator = StateGenerator(num_rows=num_rows, num_columns=num_columns,
                                     structuraly_valid_cells=structural_valid_cells)

    cell_label_dict = state_generator.get_cell_label_to_x_y_coord_dict()

    incident_df = state_generator.get_incident_df(incident_df_path)

    valid_cell_ids, raw_rate_dict = get_cell_id_and_rates(incident_df, structural_valid_cells)

    ############################################
    ### run specific
    pool_thread_count = 6
    region_cluster_file = data_directory + 'cluster_output_6_r.pk'
    num_regions = 6
    ############################################

    region_to_cell_dict, cell_to_region_dict, region_centers_dict, region_mean_distance_dict = state_generator.get_regions(
        region_cluster_file, cell_label_dict, number_of_regions=num_regions)

    cell_ids = list(cell_to_region_dict.keys())
    region_ids = list(region_centers_dict.keys())

    processed_rate_dict = dict()
    for cell in cell_ids:
        if cell in raw_rate_dict.keys():
            processed_rate_dict[cell] = raw_rate_dict[cell]
        else:
            processed_rate_dict[cell] = 0.0

    ############################################
    ### run specific

    # exp_1:
    # index_of_test_chain = 50
    # exp_2:
    # index_of_test_chain = 51
    # exp_3:
    index_of_test_chain = 52
    # exp_4:
    # index_of_test_chain = 53
    # exp_5:
    # index_of_test_chain = 54

    indexes_of_sample_chains = list(range(50))
    predictor = ChainPredictor_Subset(preprocessed_chain_file_path=preprocessed_chains_file_path,
                                      cell_rate_dictionary=processed_rate_dict,
                                      start_time=start_time,
                                      end_time=end_time + (2 * lookahead_horizon_delta_t),
                                      experimental_lookahead_horizon=lookahead_horizon_delta_t,
                                      chain_lookahead_horizon=4.0 * 60 * 60,
                                      check_if_time_diff_constraint_violoated=False,
                                      index_of_exp_chain=index_of_test_chain,
                                      chain_indexes_to_use_for_expierements=indexes_of_sample_chains)

    exp_incident_events = copy.deepcopy(predictor.exp_chain)  # TODO | remove overwrite

    exp_incident_events = [_ for _ in exp_incident_events if _.time < end_time]
    exp_incident_events = [_ for _ in exp_incident_events if _.time >= start_time]
    config_info['predictor'] = 'stationary_predicted_chains - also use as exp incident chain'
    config_info['exp_id'] = 'stationary_r{}_ch{}'.format(num_regions, index_of_test_chain)
    ############################################

    # get depots
    depots = state_generator.get_depots(depot_info_path)
    # TODO | remove depots in region 0
    del depots[4]
    del depots[11]

    cell_travel_model = CellToCellTravelModel(cell_to_xy_dict=cell_label_dict,
                                              travel_rate_cells_per_second=travel_rate_cells_per_second)

    hl_policy = create_high_level_policy(_wait_time_threshold=wait_time_threshold,
                                         _predictor=predictor,
                                         region_centers=region_centers_dict,
                                         _cell_travel_model=cell_travel_model,
                                         region_mean_distance_dict=region_mean_distance_dict,
                                         travel_rate_cells_per_second=travel_rate_cells_per_second,
                                         mean_service_time=mean_service_time,
                                         region_ids=region_ids)

    resp_assignments = get_resp_init_assignments_to_regions_and_depots(hl_policy=hl_policy,
                                                                       region_ids=region_ids,
                                                                       curr_time=None,
                                                                       region_to_cell_dict=region_to_cell_dict,
                                                                       total_responders=num_resp,
                                                                       depots=depots,
                                                                       predictor=predictor)

    resp_ids = list(resp_assignments.keys())

    resp_objs = state_generator.get_responders(num_responders=num_resp,
                                               start_time=start_time,
                                               resp_to_region_and_depot_assignments=resp_assignments,
                                               resp_ids=resp_ids,
                                               depots=depots)

    start_state = State(responders=resp_objs,
                        depots=depots,
                        active_incidents=[],
                        time=start_time,
                        cells=cell_to_region_dict,
                        regions=region_to_cell_dict)

    ts2 = copy.deepcopy(start_state)

    dispatch_policy = SendNearestDispatchPolicy(cell_travel_model)

    env_model = EnvironmentModel(cell_travel_model)

    mdp_environment_model = DecisionEnvironmentDynamics(cell_travel_model, SendNearestDispatchPolicy(cell_travel_model))
    rollout_policy = DoNothingRollout()

    llpolicy = LowLevelCentMCTSPolicy(region_ids=region_ids,
                                      incident_prediction_model=predictor,
                                      min_allocation_period=min_time_between_allocation,  # Every half hour
                                      lookahead_horizon_delta_t=lookahead_horizon_delta_t,  # look ahead 2 hours
                                      pool_thread_count=pool_thread_count,
                                      mcts_type=mcts_type,
                                      mcts_discount_factor=mcts_discount_factor,
                                      mdp_environment_model=mdp_environment_model,
                                      rollout_policy=rollout_policy,
                                      uct_tradeoff=uct_tradeoff,
                                      iter_limit=iter_limit,
                                      allowed_computation_time=allowed_computation_time  # 5 seconds per thread
                                      )

    decision_coordinator = LowLevelCoord(environment_model=env_model,
                                         travel_model=cell_travel_model,
                                         dispatch_policy=dispatch_policy,
                                         min_time_between_allocation=min_time_between_allocation,
                                         low_level_policy=llpolicy)

    simulator = Simulator(starting_event_queue=copy.deepcopy(exp_incident_events),
                          starting_state=ts2,
                          environment_model=env_model,
                          event_processing_callback=decision_coordinator.event_processing_callback_funct)

    simulator.run_simulation()

    print(decision_coordinator.metrics)

    print('avg resp_time: ', np.mean(list(decision_coordinator.metrics['resp_times'].values())))

    print_resp_info(ts2)

    # plot_resp_time_results(decision_coordinator.metrics['resp_times'])

    print(config_info)

    pickle.dump((config_info, decision_coordinator.metrics),
                open(data_directory + 'stationary_result-' + config_info['exp_id'] + '.pkl', 'wb'))  # will overwrite

    print('testing pickle file')

    print(pickle.load(open(data_directory + 'result.pk', 'rb')))

    print('done')

