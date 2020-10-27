import pickle
from datetime import datetime

from Environment.DataStructures.State import State
from Environment.Spatial.spatialStructure import get_cell_id_and_rates
from Prediction.ChainPredictor.ChainPredictor import ChainPredictor
from decision_making.HighLevel.MMCHighLevelPolicy import MMCHighLevelPolicy
from scenarios.City_1.BetweenRegionTravelModel import BetweenRegionTravelModel
from scenarios.City_1.CellToCellTravelModel import CellToCellTravelModel
from scenarios.City_1.WithinRegionTravelModel import WithinRegionTravelModel
from scenarios.City_1.definition.generate_start_state import StateGenerator


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

    print(initial_resp_to_region_allocation)

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


def get_resp_init_assignments_to_regions_and_depots_SPIKED_____(hl_policy,
                                                    region_ids,
                                                    curr_time,
                                                    region_to_cell_dict,
                                                    total_responders,
                                                    depots,
                                                    predictor,
                                                                multipl,
                                                                spiked_region):


    region_incident_rates = dict()
    for r in region_ids:
        if r == spiked_region:
            region_incident_rates[r] = multipl * hl_policy.get_region_total_rate(None, region_to_cell_dict[r])
        else:
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

    print(initial_resp_to_region_allocation)


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
    # config info
    num_rows = 30
    num_columns = 30
    num_resp = 26
    travel_rate_cells_per_second = 30.0 / 3600.0  # cells are 1x1 miles; 60 mph
    wait_time_threshold = 10 * 60  # 10 minutes
    mean_service_time = 20 * 60

    start_time = datetime(year=2019, month=1, day=1).timestamp()
    # end_time = datetime(year=2019, month=1, day=31).timestamp()
    end_time = datetime(year=2019, month=1, day=6).timestamp()

    data_directory = '../../../data/'
    incident_df_path = data_directory + 'cityMVA_ETrims_Jan2018_April2019.json'
    depot_info_path = data_directory + 'testing/depots_with_cells.json'
    region_cluster_file = data_directory + 'cluster_output_7_r.pk'
    num_regions = 7
    structural_valid_cells = pickle.load(open(data_directory + 'valid_cells_out.pkl', 'rb'))
    # preprocessed_chains_file_path = data_directory + 'chain_data/oct_6_2020_macros_processed_chains_v3.pickle'
    # preprocessed_chains_file_path = data_directory + 'chain_data/poisson_chains_v1.pickle'
    # preprocessed_chains_file_path = data_directory + 'chain_data/poisson_chains_15_v1.pickle'
    preprocessed_chains_file_path = data_directory + 'chain_data/poisson_chains_60_v1.pickle'


    structural_valid_cells = [str(_) for _ in structural_valid_cells]

    ###########################################################
    # generate state and other components
    state_generator = StateGenerator(num_rows=num_rows, num_columns=num_columns,
                                     structuraly_valid_cells=structural_valid_cells)

    incident_df = state_generator.get_incident_df(incident_df_path)

    cell_label_dict = state_generator.get_cell_label_to_x_y_coord_dict()

    region_to_cell_dict, cell_to_region_dict, region_centers_dict, region_mean_distance_dict = state_generator.get_regions(
        region_cluster_file, cell_label_dict, number_of_regions=num_regions)

    cell_ids = list(cell_to_region_dict.keys())
    region_ids = list(region_centers_dict.keys())

    # get depots
    depots = state_generator.get_depots(depot_info_path)
    # TODO | remove depots in region 0
    del depots[4]
    del depots[11]

    valid_cell_ids, raw_rate_dict = get_cell_id_and_rates(incident_df, structural_valid_cells)
    # processed_rate_dict = dict()
    # for cell in cell_ids:
    #     if cell in raw_rate_dict.keys():
    #         processed_rate_dict[cell] = raw_rate_dict[cell]
    #     else:
    #         processed_rate_dict[cell] = 0.0

    exp_incident_events = state_generator.get_expieremental_incident_events(incident_df,
                                                                            start_time,
                                                                            end_time,
                                                                            mean_clearance_time=20 * 60,
                                                                            valid_cells=valid_cell_ids)

    cell_travel_model = CellToCellTravelModel(cell_to_xy_dict=cell_label_dict,
                                              travel_rate_cells_per_second=travel_rate_cells_per_second)

    ##############################################
    cell_rates_normal = pickle.load(open(data_directory+ 'chain_data/rates.pickle', 'rb'))
    cell_rates_normal = {str(_[0]): float(_[1]) / 60.0 for _ in cell_rates_normal.items() if str(_[0]) in valid_cell_ids}

    for cell in cell_ids:
        if cell not in cell_rates_normal.keys():
            cell_rates_normal[cell] = 0.0


    predictor_normal = ChainPredictor(preprocessed_chain_file_path=preprocessed_chains_file_path,
                                                          cell_rate_dictionary=cell_rates_normal,
                                                          max_time_diff_on_lookup=31 * 60,
                                                          check_if_time_diff_constraint_violoated=True)

    hl_policy_normal = create_high_level_policy(_wait_time_threshold=wait_time_threshold,
                                         _predictor=predictor_normal,
                                         region_centers=region_centers_dict,
                                         _cell_travel_model=cell_travel_model,
                                         region_mean_distance_dict=region_mean_distance_dict,
                                         travel_rate_cells_per_second=travel_rate_cells_per_second,
                                         mean_service_time=mean_service_time,
                                         region_ids=region_ids)

    normal_assignments = get_resp_init_assignments_to_regions_and_depots(hl_policy=hl_policy_normal,
                                                                       region_ids=region_ids,
                                                                       curr_time=None,
                                                                       region_to_cell_dict=region_to_cell_dict,
                                                                       total_responders=num_resp,
                                                                       depots=depots,
                                                                       predictor=predictor_normal)

    #########

    cell_rates_spiked_rush = pickle.load(open(data_directory+ 'chain_data/spiked_rates_rush.pickle', 'rb'))
    cell_rates_spiked_rush = {str(_[0]): float(_[1]) / 60.0 for _ in cell_rates_spiked_rush.items() if str(_[0]) in valid_cell_ids}
    for cell in cell_ids:
        if cell not in cell_rates_spiked_rush.keys():
            cell_rates_spiked_rush[cell] = 0.0


    predictor_spiked_rush = ChainPredictor(preprocessed_chain_file_path=preprocessed_chains_file_path,
                                      cell_rate_dictionary=cell_rates_spiked_rush,
                                      max_time_diff_on_lookup=31 * 60,
                                      check_if_time_diff_constraint_violoated=True)

    hl_policy_spiked_rush = create_high_level_policy(_wait_time_threshold=wait_time_threshold,
                                                _predictor=predictor_spiked_rush,
                                                region_centers=region_centers_dict,
                                                _cell_travel_model=cell_travel_model,
                                                region_mean_distance_dict=region_mean_distance_dict,
                                                travel_rate_cells_per_second=travel_rate_cells_per_second,
                                                mean_service_time=mean_service_time,
                                                region_ids=region_ids)

    spiked_assignment = get_resp_init_assignments_to_regions_and_depots(hl_policy=hl_policy_spiked_rush,
                                                                         region_ids=region_ids,
                                                                         curr_time=None,
                                                                         region_to_cell_dict=region_to_cell_dict,
                                                                         total_responders=num_resp,
                                                                         depots=depots,
                                                                         predictor=predictor_spiked_rush)

    ##########

    cell_rates_spiked_game = pickle.load(open(data_directory + 'chain_data/spiked_rates_game.pickle', 'rb'))
    cell_rates_spiked_game = {str(_[0]): float(_[1]) / 60.0 for _ in cell_rates_spiked_game.items() if
                              str(_[0]) in valid_cell_ids}
    for cell in cell_ids:
        if cell not in cell_rates_spiked_game.keys():
            cell_rates_spiked_game[cell] = 0.0

    predictor_spiked_game = ChainPredictor(preprocessed_chain_file_path=preprocessed_chains_file_path,
                                           cell_rate_dictionary=cell_rates_spiked_game,
                                           max_time_diff_on_lookup=31 * 60,
                                           check_if_time_diff_constraint_violoated=True)

    hl_policy_spiked_game = create_high_level_policy(_wait_time_threshold=wait_time_threshold,
                                                     _predictor=predictor_spiked_game,
                                                     region_centers=region_centers_dict,
                                                     _cell_travel_model=cell_travel_model,
                                                     region_mean_distance_dict=region_mean_distance_dict,
                                                     travel_rate_cells_per_second=travel_rate_cells_per_second,
                                                     mean_service_time=mean_service_time,
                                                     region_ids=region_ids)

    spiked_assignment = get_resp_init_assignments_to_regions_and_depots(hl_policy=hl_policy_spiked_game,
                                                                        region_ids=region_ids,
                                                                        curr_time=None,
                                                                        region_to_cell_dict=region_to_cell_dict,
                                                                        total_responders=num_resp,
                                                                        depots=depots,
                                                                        predictor=predictor_spiked_game)

    # hl_policy_spiked_rush.check_if_triggered(s)













    print('done')

