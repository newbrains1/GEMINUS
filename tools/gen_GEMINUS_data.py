import os
import json
import numpy as np
import gzip
import time
import argparse
import multiprocessing as mp
from tqdm import trange
import queue

# --- CONSTANTS (Model/Data Structure Specific) ---
INPUT_FRAMES = 1
FUTURE_FRAMES = 4 * 5  # 10hz --> 2hz


# --- UTILITY CLASSES AND FUNCTIONS ---

class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    RESET = '\033[0m'


def get_action(index):
    # This function remains unchanged
    Discrete_Actions_DICT = {
        0: (0, 0, 1, False), 1: (0.7, -0.5, 0, False), 2: (0.7, -0.3, 0, False),
        3: (0.7, -0.2, 0, False), 4: (0.7, -0.1, 0, False), 5: (0.7, 0, 0, False),
        6: (0.7, 0.1, 0, False), 7: (0.7, 0.2, 0, False), 8: (0.7, 0.3, 0, False),
        9: (0.7, 0.5, 0, False), 10: (0.3, -0.7, 0, False), 11: (0.3, -0.5, 0, False),
        12: (0.3, -0.3, 0, False), 13: (0.3, -0.2, 0, False), 14: (0.3, -0.1, 0, False),
        15: (0.3, 0, 0, False), 16: (0.3, 0.1, 0, False), 17: (0.3, 0.2, 0, False),
        18: (0.3, 0.3, 0, False), 19: (0.3, 0.5, 0, False), 20: (0.3, 0.7, 0, False),
        21: (0, -1, 0, False), 22: (0, -0.6, 0, False), 23: (0, -0.3, 0, False),
        24: (0, -0.1, 0, False), 25: (1, 0, 0, False), 26: (0, 0.1, 0, False),
        27: (0, 0.3, 0, False), 28: (0, 0.6, 0, False), 29: (0, 1.0, 0, False),
        30: (0.5, -0.5, 0, True), 31: (0.5, -0.3, 0, True), 32: (0.5, -0.2, 0, True),
        33: (0.5, -0.1, 0, True), 34: (0.5, 0, 0, True), 35: (0.5, 0.1, 0, True),
        36: (0.5, 0.2, 0, True), 37: (0.5, 0.3, 0, True), 38: (0.5, 0.5, 0, True),
    }
    throttle, steer, brake, _ = Discrete_Actions_DICT[index]
    return throttle, steer, brake


def parse_scenario_file(file_path):
    # This function remains unchanged
    scenarios = {}
    current_category = None
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if '_Town' not in line:
                    current_category = line
                    scenarios[current_category] = []
                    print(f"{Colors.CYAN}Found category: {current_category}{Colors.RESET}")
                elif current_category:
                    scenarios[current_category].append(line)
    except FileNotFoundError:
        print(f"{Colors.RED}Error: Scenario file not found at {file_path}{Colors.RESET}")
        print(
            f"{Colors.YELLOW}Please ensure 'all_scenarios_train.txt' and 'all_scenarios_val.txt' are in the same directory as the script.{Colors.RESET}")
        exit(1)
    return scenarios


# --- DATA GENERATION CORE LOGIC ---

def gen_single_route(route_folder, count):
    # This function's internal logic remains the same
    folder_path = os.path.join(route_folder, 'anno')
    if not os.path.exists(folder_path):
        print(f"{Colors.YELLOW}Warning: Annotation folder not found for {route_folder}, skipping.{Colors.RESET}")
        return None
    length = len([name for name in os.listdir(folder_path)]) - 1
    if length < INPUT_FRAMES + FUTURE_FRAMES:
        return None

    # ... (rest of the function is unchanged)
    seq_future_x, seq_future_y, seq_future_theta, seq_future_feature, seq_future_action, seq_future_action_index, seq_future_only_ap_brake = [], [], [], [], [], [], []
    seq_input_x, seq_input_y, seq_input_theta = [], [], []
    seq_front_img, seq_feature, seq_value, seq_speed = [], [], [], []
    seq_action, seq_action_index = [], []
    seq_x_target, seq_y_target, seq_target_command, seq_only_ap_brake = [], [], [], []
    full_seq_x, full_seq_y, full_seq_theta = [], [], []
    full_seq_feature, full_seq_action, full_seq_action_index, full_seq_only_ap_brake = [], [], [], []

    for i in range(length):
        with gzip.open(os.path.join(route_folder, f'anno/{i:05}.json.gz'), 'rt', encoding='utf-8') as gz_file:
            anno = json.load(gz_file)
        expert_feature = np.load(os.path.join(route_folder, f'expert_assessment/{i:05}.npz'), allow_pickle=True)[
            'arr_0']

        full_seq_x.append(anno['x'])
        full_seq_y.append(anno['y'])
        full_seq_theta.append(anno['theta'])
        full_seq_feature.append(expert_feature[:-2])
        throttle, steer, brake = get_action(int(expert_feature[-1]))
        full_seq_action.append(np.array([throttle, steer, brake], dtype=np.float32))
        full_seq_action_index.append(int(expert_feature[-1]))
        full_seq_only_ap_brake.append(anno['only_ap_brake'])

    for i in range(INPUT_FRAMES - 1, length - FUTURE_FRAMES - 5):
        with gzip.open(os.path.join(route_folder, f'anno/{i:05}.json.gz'), 'rt', encoding='utf-8') as gz_file:
            anno = json.load(gz_file)
        expert_feature = np.load(os.path.join(route_folder, f'expert_assessment/{i:05}.npz'), allow_pickle=True)[
            'arr_0']

        seq_input_x.append(full_seq_x[i - (INPUT_FRAMES - 1):i + 5:5])
        seq_input_y.append(full_seq_y[i - (INPUT_FRAMES - 1):i + 5:5])
        seq_input_theta.append(full_seq_theta[i - (INPUT_FRAMES - 1):i + 5:5])
        seq_future_x.append(full_seq_x[i + 5:i + FUTURE_FRAMES + 5:5])
        seq_future_y.append(full_seq_y[i + 5:i + FUTURE_FRAMES + 5:5])
        seq_future_theta.append(full_seq_theta[i + 5:i + FUTURE_FRAMES + 5:5])
        seq_future_feature.append(full_seq_feature[i + 5:i + FUTURE_FRAMES + 5:5])
        seq_future_action.append(full_seq_action[i + 5:i + FUTURE_FRAMES + 5:5])
        seq_future_action_index.append(full_seq_action_index[i + 5:i + FUTURE_FRAMES + 5:5])
        seq_future_only_ap_brake.append(full_seq_only_ap_brake[i + 5:i + FUTURE_FRAMES + 5:5])
        seq_feature.append(expert_feature[:-2])
        seq_value.append(expert_feature[-2])
        front_img_list = [os.path.join(route_folder, f'camera/rgb_front/{i:05}.jpg') for _ in
                          range(INPUT_FRAMES - 1, -1, -1)]
        seq_front_img.append(front_img_list)
        seq_speed.append(anno["speed"])
        throttle, steer, brake = get_action(int(expert_feature[-1]))
        seq_action.append(np.array([throttle, steer, brake], dtype=np.float32))
        seq_action_index.append(int(expert_feature[-1]))
        seq_x_target.append(anno["x_target"])
        seq_y_target.append(anno["y_target"])
        seq_target_command.append(anno["next_command"])
        seq_only_ap_brake.append(anno["only_ap_brake"])

    with count.get_lock():
        count.value += 1
    return seq_future_x, seq_future_y, seq_future_theta, seq_future_feature, seq_future_action, seq_future_action_index, seq_future_only_ap_brake, seq_input_x, seq_input_y, seq_input_theta, seq_front_img, seq_feature, seq_value, seq_speed, seq_action, seq_action_index, seq_x_target, seq_y_target, seq_target_command, seq_only_ap_brake


def gen_sub_folder(seq_data_list, category_name, category_idx, is_train_mode, output_dir):
    # This function remains unchanged
    print('Aggregating and saving data...', flush=True)
    all_data = [[] for _ in range(20)]

    for seq_data in seq_data_list:
        if not seq_data: continue
        for i, data_list in enumerate(seq_data):
            all_data[i].extend(data_list)

    if not all_data[0]:
        print(f"{Colors.YELLOW}No data collected for category {category_name}. Skipping save.{Colors.RESET}")
        return

    data_dict = {
        'future_x': all_data[0], 'future_y': all_data[1], 'future_theta': all_data[2],
        'future_feature': all_data[3], 'future_action': all_data[4], 'future_action_index': all_data[5],
        'future_only_ap_brake': all_data[6], 'input_x': all_data[7], 'input_y': all_data[8],
        'input_theta': all_data[9], 'front_img': all_data[10], 'feature': all_data[11],
        'value': all_data[12], 'speed': all_data[13], 'action': all_data[14],
        'action_index': all_data[15], 'x_target': all_data[16], 'y_target': all_data[17],
        'target_command': all_data[18], 'only_ap_brake': all_data[19]
    }
    data_dict['scenario'] = [category_idx] * len(data_dict['value'])

    # Python's truthiness handles is_train_mode (1=True, 0=False) correctly
    mode = "train" if is_train_mode else "val"
    file_name = f"GEMINUS_{category_name}-{mode}.npy"
    output_path = os.path.join(output_dir, file_name)

    np.save(output_path, data_dict)
    print(f"{Colors.GREEN}Successfully saved {len(data_dict['value'])} data points to {output_path}{Colors.RESET}",
          flush=True)


# --- MULTIPROCESSING WORKERS ---

def get_folder_path(folder_paths, total, route_list_for_category, base_data_path):
    # This function remains unchanged
    for d0 in os.listdir(base_data_path):
        if d0 in route_list_for_category:
            folder_paths.put(os.path.join(base_data_path, d0))
            with total.get_lock():
                total.value += 1


def worker(folder_paths, count, seq_data_list, stop_event, worker_num, completed_workers):
    while not stop_event.is_set():
        try:
            folder_path = folder_paths.get(timeout=1)
            seq_data = gen_single_route(folder_path, count)
            if seq_data:
                seq_data_list.append(seq_data)
        except queue.Empty:
            break

    with completed_workers.get_lock():
        completed_workers.value += 1
        if completed_workers.value == worker_num:
            stop_event.set()

def display(count, total, stop_event):
    # This function remains unchanged
    t1 = time.time()
    with trange(int(total.value)) as pbar:
        while not stop_event.is_set():
            processed_count = count.value
            pbar.n = processed_count
            pbar.set_description(f"Processing routes")
            pbar.refresh()
            if processed_count > 0:
                pbar.set_postfix(its=f"{processed_count / (time.time() - t1):.2f}it/s")

            if processed_count == total.value:
                break
            time.sleep(0.5)


# --- MAIN EXECUTION LOGIC ---

def process_category(category_name, category_idx, route_list, args, output_dir):
    # This function remains unchanged
    print(
        f"\n{Colors.BLUE}--- Starting processing for category: {category_name} ({len(route_list)} routes) ---{Colors.RESET}")

    manager = mp.Manager()
    folder_paths = manager.Queue()
    seq_data_list = manager.list()
    count = mp.Value('d', 0)
    total = mp.Value('d', 0)
    stop_event = mp.Event()
    completed_workers = mp.Value('i', 0)

    get_folder_path(folder_paths, total, route_list, args.data_path)

    if total.value == 0:
        print(
            f"{Colors.YELLOW}No matching route folders found in {args.data_path} for this category. Skipping.{Colors.RESET}")
        return

    processes = []
    for _ in range(args.workers):
        p = mp.Process(target=worker,
                       args=(folder_paths, count, seq_data_list, stop_event, args.workers, completed_workers))
        p.daemon = True
        p.start()
        processes.append(p)

    p_disp = mp.Process(target=display, args=(count, total, stop_event))
    p_disp.daemon = True
    p_disp.start()
    processes.append(p_disp)

    for p in processes:
        p.join()

    print(f"\n{Colors.GREEN}Finished processing all routes for {category_name}.{Colors.RESET}")

    gen_sub_folder(list(seq_data_list), category_name, category_idx, args.train, output_dir)


if __name__ == '__main__':
    # 1. Setup Argument Parser
    parser = argparse.ArgumentParser(description="Preprocess Bench2Drive dataset based on scenario categories.")

    parser.add_argument(
        '--train',
        type=int,
        default=1,
        choices=[0, 1],
        help='Set mode: 1 for training, 0 for validation. (default: 0)'
    )

    parser.add_argument('--data_path', type=str,
                        default='/path/to/your/Bench2Drive-base',
                        help='Path to the base Bench2Drive dataset directory.')

    # Corrected line below
    parser.add_argument('--workers', type=int, default=64,
                        help='Number of worker processes for data generation.')

    args = parser.parse_args()

    # 2. Dynamic Path Calculation
    script_dir = os.path.dirname(os.path.abspath(__file__))

    scenario_filename = "all_scenarios_train.txt" if args.train == 1 else "all_scenarios_val.txt"
    scenario_file_path = os.path.join(script_dir, scenario_filename)

    output_dir = os.path.join(script_dir, 'processed_data')
    os.makedirs(output_dir, exist_ok=True)
    print(f"{Colors.MAGENTA}Output files will be saved to: {output_dir}{Colors.RESET}")

    mode_str = 'Training' if args.train == 1 else 'Validation'
    print(f"{Colors.MAGENTA}Mode selected: {mode_str}{Colors.RESET}")

    # 3. Parse the scenario file
    all_scenarios = parse_scenario_file(scenario_file_path)

    category_to_idx = {
        "Merging": 0, "Overtaking": 1, "EmergencyBrake": 2,
        "GiveWay": 3, "TrafficSign": 4
    }

    # 4. Loop through each category and process it
    for category_name, route_list in all_scenarios.items():
        if category_name in category_to_idx:
            category_idx = category_to_idx[category_name]
            process_category(category_name, category_idx, route_list, args, output_dir)
        else:
            print(
                f"{Colors.RED}Warning: Category '{category_name}' from file is not in category_to_idx mapping. Skipping.{Colors.RESET}")

    print(f"\n{Colors.MAGENTA}--- All categories have been processed successfully. ---{Colors.RESET}")