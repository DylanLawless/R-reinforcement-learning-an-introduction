library(ggplot2); theme_set(theme_bw())
library(dplyr)
library(tidyr)
library(gganimate)
library(patchwork)
set.seed(666)

### 1. Environment (Reactor) Setup ###
grid_size <- 3
init_temp <- 20       # initial temperature (e.g. 20)
goal_temp <- 30       # desired operating temperature (set point)
limit_min <- 28
limit_max <- 34
min_temp <- 10
max_temp <- 50
rand_heat_min <- 1.0
rand_heat_max <- 1.4

# RL and simulation parameters
SOFTMAX_TEMP <- 1     # temperature parameter for softmax (6 get stuck at bad reward after 8 episodes)
EPISODE_MAX_FAIL <- 50   # maximum failures (episodes) before termination (100 is enough to see some improvement in 2 minutes)
MAX_STEPS <- 1000      # maximum steps per episode
STABILISE_STEPS <- 50  # steps to allow reactor to stabilise before checking avg_temp

# RL algorithm parameters
ALPHA <- 1000         # actor learning rate
BETA <- 0.5           # critic learning rate
GAMMA <- 0.95         # discount factor
LAMBDAw <- 0.9        # actor eligibility trace decay
LAMBDAv <- 0.8        # critic eligibility trace decay

# Remove possible_target and use goal_temp as fixed target
num_stir <- 6
possible_stir <- seq(0, 1, length.out = num_stir)
num_actions <- num_stir

### State Discretisation ###
# check if out conditions can be learned ----

### State Discretisation ###
N_avg <- 5   # number of bins for average temperature
N_std <- 4   # number of bins for standard deviation
N_STATES <- N_avg * N_std

# Define discretisation bounds
avg_lower_bound <- 10
avg_upper_bound <- 50
std_lower_bound <- 0
std_upper_bound <- 10

# Create a grid of average temperatures and standard deviations
avg_range <- seq(avg_lower_bound, avg_upper_bound, length.out = 100)
std_range <- seq(std_lower_bound, std_upper_bound, length.out = 50)
test_states <- expand.grid(avg = avg_range, std = std_range)

# Function to compute state bin index based on discretisation settings
discretise <- function(avg, std) {
  if(avg < min_temp || avg > max_temp) return(NA)
  bin_avg <- floor((avg - avg_lower_bound) / ((avg_upper_bound - avg_lower_bound) / N_avg)) + 1
  if(bin_avg > N_avg) bin_avg <- N_avg
  bin_std <- floor((std - std_lower_bound) / ((std_upper_bound - std_lower_bound) / N_std)) + 1
  if(bin_std > N_std) bin_std <- N_std
  (bin_avg - 1) * N_std + (bin_std - 1)
}

test_states$bin <- mapply(discretise, test_states$avg, test_states$std)

# Plot the bins for visual inspection
ggplot(test_states, aes(x = avg, y = std, colour = factor(bin))) +
  geom_point() +
  labs(colour = "State Bin", x = "Average Temperature", y = "Standard Deviation",
       title = "Discretisation of Reactor State")

# Figure Caption:
# "Discretisation of the Reactor State Space.
# The state space is divided into N_avg = 5 bins for the average temperature (ranging from 5 to 60)
# and N_std = 4 bins for the standard deviation (ranging from 0 to 10), resulting in N_STATES = 20 distinct bins.
# Here, 'avg' represents the mean temperature across the reactor grid, while 'std' quantifies the variation
# in temperatures. The plot colours each (avg, std) point according to its assigned state bin, with vertical
# bands reflecting differences in average temperature and horizontal bands indicating differences in standard
# deviation. This visualisation helps confirm that the discretisation adequately covers the reactor state space,
# supporting the agent's learning process."









# then continue ----
get_state_index <- function(state) {
  avg <- state[1]
  std <- state[2]
  if(avg < min_temp || avg > max_temp) return(-1)
  lower_bound <- 5
  upper_bound <- 60
  bin_avg <- floor((avg - lower_bound) / ((upper_bound - lower_bound) / N_avg)) + 1
  if(bin_avg > N_avg) bin_avg <- N_avg
  bin_std <- floor(std / (10 / N_std)) + 1
  if(bin_std > N_std) bin_std <- N_std
  # Return a list containing the index and the discretisation bins for logging.
  list(index = (bin_avg - 1) * N_std + (bin_std - 1),
       bin_avg = bin_avg, bin_std = bin_std,
       avg = avg, std = std)
}

### 2. Reactor Simulation Functions ###
init_grid <- function() {
  matrix(init_temp, nrow = grid_size, ncol = grid_size)
}

get_neighbours <- function(grid, x, y) {
  indices <- expand.grid(dx = -1:1, dy = -1:1) %>%
    filter(!(dx == 0 & dy == 0)) %>%
    mutate(nx = x + dx, ny = y + dy) %>%
    filter(nx >= 1, nx <= grid_size, ny >= 1, ny <= grid_size)
  mean(sapply(1:nrow(indices), function(i) grid[indices$ny[i], indices$nx[i]]))
}

reactor_step <- function(grid, target_temp, stirring) {
  noise <- matrix(rnorm(length(grid), mean = 0, sd = 0.5), nrow = nrow(grid))
  new_grid <- grid
  for (y in 1:grid_size) {
    for (x in 1:grid_size) {
      # random_heat <- runif(1, min = 0, max = 2)
      random_heat <- runif(1, min = rand_heat_min, max = rand_heat_max)
      nb_avg <- get_neighbours(grid, x, y)
      diffusion <- stirring * (nb_avg - grid[y, x])
      cooling <- stirring * 0.05 * grid[y, x]
      delta <- (1 - stirring) * random_heat + diffusion - cooling
      new_grid[y, x] <- grid[y, x] + delta + noise[y, x]
      new_grid[y, x] <- max(min_temp, min(new_grid[y, x], max_temp))
    }
  }
  new_grid
}

get_state <- function(grid) {
  avg <- mean(grid)
  std <- sd(as.vector(grid))
  c(avg, std)
}

### 4. RL Simulation Setup ###
compute_reward <- function(avg_temp, target_temp, stirring) {
  r <- - (abs(avg_temp - goal_temp) + abs(target_temp - goal_temp) + 2 * abs(stirring - 0.5))
  if(abs(avg_temp - goal_temp) < 1 && abs(target_temp - goal_temp) < 1 && abs(stirring - 0.5) < 0.1)
    r <- r + 1
  r
}

# Always use goal_temp as fixed target.
get_action <- function(action_index) {
  stirring <- possible_stir[action_index + 1]
  c(goal_temp, stirring)
}

softmax_fn <- function(x, temperature) {
  exp_x <- exp((x - max(x)) / temperature)
  exp_x / sum(exp_x)
}

simulate_bioreactor_control <- function() {
  set.seed(123)
  w <- matrix(runif(N_STATES * num_actions, -1, 1), nrow = N_STATES, ncol = num_actions)
  e <- matrix(0, nrow = N_STATES, ncol = num_actions)
  v <- runif(N_STATES, -1, 1)
  xbar <- rep(0, N_STATES)
  
  grid <- init_grid()
  state_vec <- get_state(grid)
  state_info <- get_state_index(state_vec)
  state_index <- state_info$index
  if(state_index < 0) stop("Initial state failure.")
  
  history <- data.frame(
    GlobalStep = integer(),
    Episode = integer(),
    StepInEpisode = integer(),
    StateIndex = integer(),
    rhat = numeric(),
    chosen_action = integer(),
    target = numeric(),
    stirring = numeric(),
    reward = numeric(),
    avg_temp = numeric(),
    std_temp = numeric(),
    w_val = numeric(),
    v_val = numeric(),
    p_action = numeric()
  )
  
  episode_lengths <- c()
  global_steps <- 0
  last_failure <- 0
  failures <- 0
  episode <- 1
  
  while(failures < EPISODE_MAX_FAIL && (global_steps - last_failure) < MAX_STEPS) {
    global_steps <- global_steps + 1
    step_in_episode <- global_steps - last_failure
    
    state_info <- get_state_index(state_vec)
    state_index <- state_info$index
    if(state_index < 0) {
      failures <- failures + 1
      episode_lengths <- c(episode_lengths, step_in_episode)
      grid <- init_grid()
      state_vec <- get_state(grid)
      last_failure <- global_steps
      episode <- episode + 1
      next
    }
    
    p_actions <- softmax_fn(w[state_index + 1, ], SOFTMAX_TEMP)
    action_index <- sample(0:(num_actions - 1), size = 1, prob = p_actions)
    action <- get_action(action_index)
    target_temp <- action[1]  # always goal_temp
    stirring <- action[2]
    
    old_value <- v[state_index + 1]
    history <- rbind(history, data.frame(
      GlobalStep = global_steps,
      Episode = episode,
      StepInEpisode = step_in_episode,
      StateIndex = state_index,
      rhat = NA,
      chosen_action = action_index,
      target = target_temp,
      stirring = stirring,
      reward = NA,
      avg_temp = state_vec[1],
      std_temp = state_vec[2],
      w_val = w[state_index + 1, action_index + 1],
      v_val = v[state_index + 1],
      p_action = p_actions[action_index + 1]
    ))
    
    grid <- reactor_step(grid, target_temp, stirring)
    new_state_vec <- get_state(grid)
    new_state_info <- get_state_index(new_state_vec)
    new_state_index <- new_state_info$index
    
    reward <- compute_reward(new_state_vec[1], target_temp, stirring)
    
    if(any(grid < min_temp) || any(grid > max_temp) ||
       (step_in_episode > STABILISE_STEPS && (new_state_vec[1] < limit_min || new_state_vec[1] > limit_max))) {
      
      failed <- TRUE
      p_new <- 0
      failures <- failures + 1
      episode_lengths <- c(episode_lengths, step_in_episode)
      last_failure <- global_steps
      
      grid <- init_grid()
      new_state_vec <- get_state(grid)
      new_state_info <- get_state_index(new_state_vec)
      new_state_index <- new_state_info$index
      
      cat(sprintf("Failure at Episode: %d | Episode Length: %d | Global Step: %d | Temp: %.2f -> %.2f\n",
                  episode, step_in_episode, global_steps, state_vec[1], new_state_vec[1]), "\n")
      cat(sprintf("Discretisation (Before -> After): Avg: %.2f (bin %d), Std: %.2f (bin %d) -> Avg: %.2f (bin %d), Std: %.2f (bin %d)\n",
                  state_info$avg, state_info$bin_avg, state_info$std, state_info$bin_std,
                  new_state_info$avg, new_state_info$bin_avg, new_state_info$std, new_state_info$bin_std), "\n")
      
      episode <- episode + 1
      next
    } else {
      failed <- FALSE
      p_new <- v[new_state_index + 1]
    }
    
    rhat <- reward + GAMMA * p_new - old_value
    rhat <- as.numeric(rhat)[1]
    history$rhat[nrow(history)] <- rhat
    history$reward[nrow(history)] <- reward
    
    for(i in 1:N_STATES) {
      noise <- runif(num_actions, -0.01, 0.01)
      w[i, ] <- w[i, ] + ALPHA * rhat * e[i, ] + noise
      v[i] <- v[i] + BETA * rhat * xbar[i]
      if(failed) {
        e[i, ] <- 0.0
        xbar[i] <- 0.0
      } else {
        e[i, ] <- e[i, ] * LAMBDAw
        xbar[i] <- xbar[i] * LAMBDAv
      }
    }
    
    grad <- rep(0, num_actions)
    grad[action_index + 1] <- 1
    grad <- grad - p_actions
    e[state_index + 1, ] <- e[state_index + 1, ] + (1 - LAMBDAw) * grad
    xbar[state_index + 1] <- xbar[state_index + 1] + (1 - LAMBDAv)
    
    state_vec <- new_state_vec
    
    if(!failed && (global_steps %% 50 == 0)) {
      cat(sprintf("Reward %.2f | Stirring %.2f | Step %d | Episode: %d | Episode Length: %d | Temp: %.2f | Discretisation: Avg=%.2f (bin %d), Std=%.2f (bin %d)\n",
                  reward , stirring, global_steps, episode, step_in_episode, new_state_vec[1],
                  new_state_info$avg, new_state_info$bin_avg, new_state_info$std, new_state_info$bin_std), "\n")
      
      # cat(sprintf("Discretisation (Before -> After): Avg: %.2f (bin %d), Std: %.2f (bin %d) -> Avg: %.2f (bin %d), Std: %.2f (bin %d)\n",
      #             state_info$avg, state_info$bin_avg, state_info$std, state_info$bin_std,
      #             new_state_info$avg, new_state_info$bin_avg, new_state_info$std, new_state_info$bin_std), "\n")
      
    }
  }
  
  if((global_steps - last_failure) >= MAX_STEPS) {
    final_state_info <- get_state_index(state_vec)
    cat(sprintf("Success at Episode: %d | Episode Length: %d | Global Step: %d | Final Temp: %.2f | Discretisation: Avg=%.2f (bin %d), Std=%.2f (bin %d)\n",
                episode, global_steps - last_failure, global_steps, state_vec[1],
                final_state_info$avg, final_state_info$bin_avg, final_state_info$std, final_state_info$bin_std), "\n")
  }
  
  list(history = history, episode_lengths = episode_lengths, global_steps = global_steps)
}














### 3. Testing the Reactor Model (No RL) ###
# Simulate reactor dynamics with fixed stirring rates: 0, 0.3, 0.6.
fixed_stir_rates <- possible_stir # same as agent's options

timesteps <- STABILISE_STEPS
results_list <- list()

for (stir in fixed_stir_rates) {
  grid <- init_grid()
  sim_results <- list()
  for (t in 1:timesteps) {
    df <- expand.grid(x = 1:grid_size, y = 1:grid_size)
    df$value <- as.vector(t(grid))
    df$timestep <- t
    df$cell <- paste0(df$x, "-", df$y)
    sim_results[[t]] <- df
    grid <- reactor_step(grid, target_temp = goal_temp, stirring = stir)
  }
  sim_data <- bind_rows(sim_results)
  sim_data$stir_rate <- stir
  results_list[[as.character(stir)]] <- sim_data
}
all_data <- bind_rows(results_list)
all_data <- all_data %>%
  mutate(temp_state = case_when(
    value > goal_temp + 1 ~ "hot",
    value < goal_temp - 1 ~ "cold",
    TRUE ~ "good"
  ))

p_metric <- ggplot(all_data, aes(x = timestep, y = value, group = cell, color = value)) +
  geom_line() +
  facet_wrap(~ stir_rate, ncol = 1, labeller = label_both) +
  geom_hline(yintercept = goal_temp, linetype = "dotted", color = "darkgreen") +
  geom_vline(xintercept = STABILISE_STEPS, linetype = "dotted", color = "darkgreen") +
  scale_color_gradientn(
    colours = c("white", "blue", "green", "yellow", "red"),
    values = scales::rescale(c(min_temp, 10, goal_temp, 20, max_temp)),
    limits = c(min_temp, max_temp)
  ) +
  labs(title = "Bioreactor Cell Temperatures Over Time\n(Fixed Stirring Rates)",
       x = "Timestep", y = "Temperature") +
  theme(legend.position = "none")
print(p_metric)
ggsave(filename = "../figures/bioreactor_metric_faceted.png", plot = p_metric, width = 8, height = 10)





# Run simulation  episodes ----

start.time <- Sys.time()
result <- simulate_bioreactor_control()
history <- result$history
episode_lengths <- result$episode_lengths
global_steps_final <- result$global_steps
end.time <- Sys.time()
cat("Time taken: ", end.time - start.time, "\n")












### 4. Plotting ###
# ---- RL Outcome Plots ----
if(length(episode_lengths) == 0) {
  df_episodes <- data.frame(Episode = 1, Steps = NA)
} else {
  df_episodes <- data.frame(
    Episode = seq_along(episode_lengths),
    Steps = episode_lengths
  )
}
p1_rl <- ggplot(df_episodes, aes(x = Episode, y = Steps)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  labs(subtitle = "Episode Lengths (Steps per Episode)",
       x = "Episode Number", y = "Steps") +
  theme_minimal()
print(p1_rl)

p2_rl <- ggplot(history, aes(x = GlobalStep, y = rhat)) +
  geom_line(color = "darkred") +
  labs(subtitle = "Evolution of Reinforcement Signal (rhat)",
       x = "Global Step", y = "rhat") +
  theme_minimal()
print(p2_rl)

p3_rl <- ggplot(history, aes(x = GlobalStep)) +
  geom_point(aes(y = avg_temp, color = "Average Temperature"), size = 0.5) +
  labs(subtitle = "Evolution of Average Temperature in Current State",
       x = "Global Step", y = "Temperature") +
  scale_color_manual(name = "Metric",
                     values = c("Average Temperature" = "darkblue")) +
  theme_minimal()
print(p3_rl)

p4_rl <- ggplot(history, aes(x = GlobalStep, y = p_action)) +
  geom_point(color = "purple", size = 0.5) +
  labs(subtitle = "Evolution of Action Probability (Chosen Stirring)",
       x = "Global Step", y = "Probability") +
  theme_minimal()
print(p4_rl)

rl_metrics <- (p1_rl / p2_rl) | (p3_rl / p4_rl)
rl_metrics <- rl_metrics + 
  plot_annotation(title = "Figure Example 3.1: Bioreactor Control Learning Dynamics")
combined_all <- p_metric / rl_metrics
print(combined_all)
filename <- file.path("../figures/fig_ex_3_1.png")
ggsave(filename = filename, combined_all, width = 8, height = 10)
# 
# # ---- Dynamic Animation of RL Learning Outcomes ----
# # Add dynamic subtitle text to history.
# history <- history %>%
#   mutate(subtitle_text = paste("StateIndex:", StateIndex,
#                                "\nrhat:", round(rhat, 2),
#                                "\nAction:", chosen_action,
#                                "\nTarget:", target,
#                                "\nStir:", stirring,
#                                "\nReward:", round(reward, 2),
#                                "\nAvg:", round(avg_temp, 2),
#                                "\nStd:", round(std_temp, 2),
#                                "\nw:", round(w_val, 2),
#                                "\nv:", round(v_val, 2),
#                                "\np:", round(p_action, 2)))
# 
# p_anim <- ggplot(history, aes(x = GlobalStep, y = avg_temp)) +
#   geom_line(color = "darkgreen") +
#   geom_point(aes(color = avg_temp), size = 1) +
#   scale_color_gradientn(
#     colours = c("white", "blue", "green", "yellow", "red"),
#     values = scales::rescale(c(min_temp, 10, goal_temp, 20, max_temp)),
#     limits = c(min_temp, max_temp)
#   ) +
#   # labs(title = "Evolution of Reactor Average Temperature",
#   #      subtitle = "{closest_state(subtitle_text)}",
#   #      x = "Global Step", y = "Average Temperature") +
#   
#   labs(title = "Evolution of Reactor Average Temperature\n",
#        subtitle = paste("Goal Temp =", goal_temp,
#                         "| Fine Actions: Target =", paste(round(possible_target, 2), collapse = ", "),
#                         " & Stir =", paste(round(possible_stir, 2), collapse = ", "),
#                         "\nTotal Trials:", length(trial_lengths),
#                         "| Global Step: {round(frame_time,0)}"),
#        x = "Global Step", y = "Average Temperature") +
#   
#   transition_time(GlobalStep) +
#   shadow_mark(exclude_layer = 1, past = TRUE, alpha = 0.5, size = 1)

# main animation ----

# First, compute a dynamic subtitle column in history using the new metrics.
history <- history %>%
  mutate(dynamic_subtitle = paste("Goal Temp =", goal_temp,
                                  "| Fine Action: Stir =", paste(round(possible_stir, 2), collapse = ", "),
                                  "\nEpisode:", Episode,
                                  "\nStepInEpisode:", StepInEpisode,
                                  "\nStateIndex:", StateIndex,
                                  "\nrhat:", round(rhat, 2),
                                  "\nAction:", chosen_action,
                                  "\nReward:", round(reward, 2),
                                  "\nAvg Temp:", round(avg_temp, 2),
                                  "\nStd Temp:", round(std_temp, 2),
                                  "\nw:", round(w_val, 2),
                                  "\nv:", round(v_val, 2),
                                  "\np:", round(p_action, 2),
                                  "\nGlobal Step:", GlobalStep,
                                  "\nStirring:",history$stirring))

# Then, include this dynamic subtitle as an on-plot text layer (rather than in labs)
p_anim <- ggplot(history, aes(x = GlobalStep, y = avg_temp)) +
  geom_line(color = "darkgreen") +
  geom_point(aes(color = avg_temp), size = 1) +
  # Add dynamic text showing the current state's details in the top-right corner:
  geom_text(aes(label = dynamic_subtitle), 
            x = -Inf, y = Inf, hjust = 0, vjust = 1.1, size = 3, color = "black") +
  
  scale_color_gradientn(
    colours = c("white", "blue", "green", "yellow", "red"),
    values = scales::rescale(c(min_temp, limit_min, goal_temp, limit_max, max_temp)),
    limits = c(min_temp, max_temp)
  ) +
  labs(title = "Evolution of Reactor Average Temperature",
       x = "Global Step", y = "Average Temperature") +
  transition_time(GlobalStep) +
  shadow_mark(exclude_layer = 3, past = TRUE, alpha = 0.5, size = 1)

# anim2 <- animate(p_anim, nframes = 100, fps = 10, width = 700, height = 500, res = 120)
anim2 <- animate(p_anim, nframes = 200, fps = 20, width = 800, height = 500, res = 120)
gif_filename <- file.path("../figures/fig_ex_3_1_anim.gif")
anim_save(gif_filename, anim2)

# plot reward ----

p_anim_reward <- ggplot(history, aes(x = GlobalStep, y = reward)) +
  geom_line(color = "darkgreen") +
  geom_point(aes(color = avg_temp), size = 1) +
  # Add dynamic text showing the current state's details in the top-left corner:
  geom_text(aes(label = dynamic_subtitle), 
            x = -Inf, y = Inf, hjust = 0, vjust = 1.1, size = 3, color = "black") +
  scale_color_gradientn(
    colours = c("white", "blue", "green", "yellow", "red"),
    values = scales::rescale(c(min_temp, limit_min, goal_temp, limit_max, max_temp)),
    limits = c(min_temp, max_temp)
  ) +
  labs(title = "Evolution of Reactor Reward",
       x = "Global Step", y = "Reward") +
  transition_time(GlobalStep) +
  shadow_mark(exclude_layer = 3, past = TRUE, alpha = 0.5, size = 1)

anim_reward <- animate(p_anim_reward, nframes = 200, fps = 20, width = 800, height = 500, res = 120)
gif_filename_reward <- file.path("../figures/fig_ex_1_anim_reward.gif")
anim_save(gif_filename_reward, anim_reward)


p_reward_static <- ggplot(history, aes(x = GlobalStep, y = reward)) +
  geom_line(color = "darkgreen") +
  geom_point(aes(color = avg_temp), size = 1) +
  scale_color_gradientn(
    colours = c("white", "blue", "green", "yellow", "red"),
    values = scales::rescale(c(min_temp, limit_min, goal_temp, limit_max, max_temp)),
    limits = c(min_temp, max_temp)
  ) +
  labs(title = "Evolution of Reactor Reward",
       x = "Global Step", y = "Reward")

p_reward_static
