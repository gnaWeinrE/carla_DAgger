
import random
import numpy as np
import tensorflow as tf

from agent import create_model
from reward_functions import reward_functions

USE_ROUTE_ENVIRONMENT = False

if USE_ROUTE_ENVIRONMENT:
    from CarlaEnv.carla_route_env import CarlaRouteEnv as CarlaEnv
else:
    from CarlaEnv.carla_lap_env import CarlaLapEnv as CarlaEnv


class DaggerBuffer:
    def __init__(self):
        self.limit = 20000
        self.state_buffer = []
        self.speed_buffer = []
        self.action_buffer = []
        self.state_cache = []
        self.speed_cache = []
        self.action_cache = []

    def add(self, state, action):
        self.state_cache.append(state[0])
        self.speed_cache.append(state[1])
        self.action_cache.append(action)

    def concatenate(self):

        total_length = len(self.state_buffer) + len(self.state_cache)
        # print(self.action_cache)

        if total_length <= self.limit:
            self.state_buffer = self.state_cache + self.state_buffer
            self.speed_buffer = self.speed_cache + self.speed_buffer
            self.action_buffer = self.action_cache + self.action_buffer
        else:
            bias = total_length - self.limit
            self.state_buffer = self.state_cache + self.state_buffer[bias:]
            self.speed_buffer = self.speed_cache + self.speed_buffer[bias:]
            self.action_buffer = self.action_cache + self.action_buffer[bias:]

        self.state_cache = []
        self.speed_cache = []
        self.action_cache = []

    def get_buffer(self):
        buffer = [np.array(self.state_buffer), np.array(self.speed_buffer)]
        # print(buffer)
        return buffer



def train(params, start_carla=True, restart=False):
    # Read parameters
    learning_rate = params["learning_rate"]
    lr_decay = params["lr_decay"]
    discount_factor = params["discount_factor"]
    gae_lambda = params["gae_lambda"]
    ppo_epsilon = params["ppo_epsilon"]
    initial_std = params["initial_std"]
    value_scale = params["value_scale"]
    entropy_scale = params["entropy_scale"]
    horizon = params["horizon"]
    num_epochs = params["num_epochs"]
    num_episodes = params["num_episodes"]
    batch_size = params["batch_size"]
    synchronous = params["synchronous"]
    fps = params["fps"]
    action_smoothing = params["action_smoothing"]
    model_name = params["model_name"]
    reward_fn = params["reward_fn"]
    seed = params["seed"]
    eval_interval = params["eval_interval"]
    record_eval = params["record_eval"]

    # Set seeds
    if isinstance(seed, int):
        tf.random.set_random_seed(seed)
        np.random.seed(seed)
        random.seed(0)

    print("")
    print("Training parameters:")
    for k, v, in params.items(): print(f"  {k}: {v}")
    print("")

    # Create env
    print("Creating environment")
    env = CarlaEnv(obs_res=(80, 160),
                   action_smoothing=action_smoothing,
                   reward_fn=reward_functions[reward_fn],
                   synchronous=synchronous,
                   fps=fps,
                   start_carla=start_carla)
    if isinstance(seed, int):
        env.seed(seed)

    # Create model
    print("Creating model")

    model_cnn = create_model()
    model_cnn.summary()

    # Prompt to load existing model if any
    # if not restart:
    #     if os.path.isdir(model.log_dir) and len(os.listdir(model.log_dir)) > 0:
    #         answer = input(
    #             "Model \"{}\" already exists. Do you wish to continue (C) or restart training (R)? ".format(model_name))
    #         if answer.upper() == "C":
    #             pass
    #         elif answer.upper() == "R":
    #             restart = True
    #         else:
    #             raise Exception(
    #                 "There are already log files for model \"{}\". Please delete it or change model_name and try again".format(
    #                     model_name))

    # if restart:
    #     shutil.rmtree(model.model_dir)
    #     for d in model.dirs:
    #         os.makedirs(d)
    # model.init_session()
    # if not restart:
    #     model.load_latest_checkpoint()
    # model.write_dict_to_summary("hyperparameters", params, 0)

    # For every episode
    beta = 1
    episode_idx = 0
    buffer = DaggerBuffer()

    while num_episodes <= 0 or episode_idx < num_episodes:

        # Run evaluation periodically
        # if episode_idx % eval_interval == 0:
        #     video_filename = os.path.join(model.video_dir, "episode{}.avi".format(episode_idx))
        #     eval_reward = run_eval(env, model, video_filename=video_filename)
        #     model.write_value_to_summary("eval/reward", eval_reward, episode_idx)
        #     model.write_value_to_summary("eval/distance_traveled", env.distance_traveled, episode_idx)
        #     model.write_value_to_summary("eval/average_speed", 3.6 * env.speed_accum / env.step_count, episode_idx)
        #     model.write_value_to_summary("eval/center_lane_deviation", env.center_lane_deviation, episode_idx)
        #     model.write_value_to_summary("eval/average_center_lane_deviation",
        #                                  env.center_lane_deviation / env.step_count, episode_idx)
        #     model.write_value_to_summary("eval/distance_over_deviation",
        #                                  env.distance_traveled / env.center_lane_deviation, episode_idx)
        #     if eval_reward > best_eval_reward:
        #         model.save()
        #         best_eval_reward = eval_reward

        # Reset environment
        if episode_idx % 10 == 0:
            beta *= 0.99

        state, terminal_state, total_reward = env.reset(), False, 0

        # While episode not done
        while not terminal_state:
            for i in range(horizon):
                # print(i)
                # action, value = model.predict(state, write_to_summary=True)
                action = model_cnn.predict([state[0][np.newaxis], state[1][np.newaxis]])
                # print("------------------")
                # print(action)
                # Perform action

                expert_flag = False
                if random.random() < beta:
                    expert_flag = True

                new_state, expert_action, reward, terminal_state, info = env.step(action[0], expert_flag)

                buffer.add(new_state, expert_action)

                if info["closed"] == True:
                    exit(0)

                env.render()
                total_reward += reward

                # Store state, action and reward
                # states.append(state)  # [T, *input_shape]
                # taken_actions.append(action)  # [T,  num_actions]
                # values.append(value)  # [T]
                # rewards.append(reward)  # [T]
                # dones.append(terminal_state)  # [T]
                state = new_state

                if terminal_state:
                    break

            buffer.concatenate()
            # action_predict = model_cnn.predict(buffer.get_buffer())

            model_cnn.fit(buffer.get_buffer(), np.array(buffer.action_buffer))

            terminal_state = True

        # Write episodic values
        # model.write_value_to_summary("train/reward", total_reward, episode_idx)
        # model.write_value_to_summary("train/distance_traveled", env.distance_traveled, episode_idx)
        # model.write_value_to_summary("train/average_speed", 3.6 * env.speed_accum / env.step_count, episode_idx)
        # model.write_value_to_summary("train/center_lane_deviation", env.center_lane_deviation, episode_idx)
        # model.write_value_to_summary("train/average_center_lane_deviation", env.center_lane_deviation / env.step_count,
        #                              episode_idx)
        # model.write_value_to_summary("train/distance_over_deviation", env.distance_traveled / env.center_lane_deviation,
        #                              episode_idx)
        # model.write_episodic_summaries()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Trains a CARLA agent with PPO")

    # PPO hyper parameters
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Initial learning rate")
    parser.add_argument("--lr_decay", type=float, default=1.0, help="Per-episode exponential learning rate decay")
    parser.add_argument("--discount_factor", type=float, default=0.99, help="GAE discount factor")
    parser.add_argument("--gae_lambda", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--ppo_epsilon", type=float, default=0.2, help="PPO epsilon")
    parser.add_argument("--initial_std", type=float, default=1.0,
                        help="Initial value of the std used in the gaussian policy")
    parser.add_argument("--value_scale", type=float, default=1.0, help="Value loss scale factor")
    parser.add_argument("--entropy_scale", type=float, default=0.01, help="Entropy loss scale factor")
    parser.add_argument("--horizon", type=int, default=600, help="Number of steps to simulate per training step")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of PPO training epochs per traning step")
    parser.add_argument("--batch_size", type=int, default=32, help="Epoch batch size")
    parser.add_argument("--num_episodes", type=int, default=0,
                        help="Number of episodes to train for (0 or less trains forever)")

    # Environment settings
    parser.add_argument("--synchronous", type=int, default=False,
                        help="Set this to True when running in a synchronous environment")
    parser.add_argument("--fps", type=int, default=30, help="Set this to the FPS of the environment")
    parser.add_argument("--action_smoothing", type=float, default=0.0, help="Action smoothing factor")
    parser.add_argument("-start_carla", action="store_true",
                        help="Automatically start CALRA with the given environment settings")

    # Training parameters
    # parser.add_argument("--model_name", type=str, required=True, help="Name of the model to train. Output written to models/model_name")
    parser.add_argument("--model_name", default="win_test",
                        help="Name of the model to train. Output written to models/model_name")
    parser.add_argument("--reward_fn", type=str,
                        default="reward_speed_centering_angle_multiply",
                        help="Reward function to use. See reward_functions.py for more info.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed to use. (Note that determinism unfortunately appears to not be garuanteed " +
                             "with this option in our experience)")
    parser.add_argument("--eval_interval", type=int, default=5, help="Number of episodes between evaluation runs")
    parser.add_argument("--record_eval", type=bool, default=True,
                        help="If True, save videos of evaluation episodes " +
                             "to models/model_name/videos/")

    # parser.add_argument("-restart", action="store_true",
    parser.add_argument("-restart", default=True,
                        help="If True, delete existing model in models/model_name before starting training")

    params = vars(parser.parse_args())

    # Remove a couple of parameters that we dont want to log
    start_carla = params["start_carla"];
    del params["start_carla"]
    restart = params["restart"];
    del params["restart"]

    # Reset tf graph
    tf.reset_default_graph()

    # Start training
    train(params, start_carla, restart)
