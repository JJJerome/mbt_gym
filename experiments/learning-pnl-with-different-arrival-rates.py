import sys

sys.path.append("../")

from experiments.helpers import (
    get_cj_env,
    get_ppo_learner_and_callback,
    get_experiment_string,
    create_time_plot,
    create_inventory_plot,
)

num_trajectories = 1000
terminal_time = 1.0
phi = 0
alpha = 0
sigma = 0.0
initial_inventory = (-5, 6)
random_start = None

final_model_path = "./final_models"

arrival_rates = [1.0, 10.0, 100.0]
fill_exponents = [0.1, 1, 10]

for arrival_rate in arrival_rates:
    for fill_exponent in fill_exponents:
        n_steps = int(10 * terminal_time * arrival_rate)
        env = get_cj_env(
            num_trajectories=num_trajectories,
            terminal_time=terminal_time,
            arrival_rate=arrival_rate,
            fill_exponent=fill_exponent,
            phi=phi,
            alpha=alpha,
            sigma=sigma,
            initial_inventory=initial_inventory,
        )
        model, callback = get_ppo_learner_and_callback(env)
        model.learn(total_timesteps=300_000_000, callback=callback)
        model.save(final_model_path + "/" + get_experiment_string(env))
        create_inventory_plot(model=model, env=env, save_figure=True)
        create_time_plot(model=model, env=env, save_figure=True)
