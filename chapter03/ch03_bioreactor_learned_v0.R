library(ggplot2); theme_set(theme_bw())
library(dplyr)
library(tidyr)
library(gganimate)
library(patchwork)

# ---- RL and Environment Parameters ----
# State discretisation: average temperature and standard deviation.
N_avg <- 5   # bins for average temperature (range: 5 to 30)
N_std <- 4   # bins for standard deviation (range: 0 to 10)
N_STATES <- N_avg * N_std

# Action space: actions are (target_temp, stirring) vectors.
num_target <- 3
num_stir <- 3
possible_target <- seq(12, 18, length.out = num_target)  # possible target temperatures
possible_stir <- seq(0.2, 0.8, length.out = num_stir)      # possible stirring rates
num_actions <- num_target * num_stir

# RL parameters (actor-critic with eligibility traces)
ALPHA <- 1000        # actor learning rate
BETA <- 0.5          # critic learning rate
GAMMA <- 0.95        # discount factor
LAMBDAw <- 0.9       # actor eligibility trace decay
LAMBDAv <- 0.8       # critic eligibility trace decay
SOFTMAX_TEMP <- 0.1  # temperature parameter for softmax
MAX_FAILURES <- 1000 # maximum failures before termination
MAX_STEPS <- 200    # maximum steps in a trial
allowed_steps_in_bad_state <- 50  # steps allowed in a persistently bad state

# Reactor environment parameters
grid_size <- 3
init_temp <- 5
goal_temp <- 15      # desired temperature
min_temp <- 0
max_temp <- 60

# ---- Environment Functions ----
init_grid <- function() {
  # matrix(goal_temp, nrow = grid_size, ncol = grid_size)
  matrix(init_temp, nrow = grid_size, ncol = grid_size)
}

get_neighbours <- function(grid, x, y) {
  indices <- expand.grid(dx = -1:1, dy = -1:1) %>%
    filter(!(dx == 0 & dy == 0)) %>%
    mutate(nx = x + dx, ny = y + dy) %>%
    filter(nx >= 1, nx <= grid_size, ny >= 1, ny <= grid_size)
  mean(sapply(1:nrow(indices), function(i) grid[indices$ny[i], indices$nx[i]]))
}

# Reactor step: update each cell using target temperature and stirring.
# reactor_step <- function(grid, target_temp, stirring) {
#   noise <- matrix(rnorm(length(grid), mean = 0, sd = 0.2), nrow = nrow(grid))
#   new_grid <- grid
#   for (y in 1:grid_size) {
#     for (x in 1:grid_size) {
#       nb_avg <- get_neighbours(grid, x, y)
#       delta <- 0.1 * (target_temp - grid[y, x]) + stirring * (nb_avg - grid[y, x])
#       new_grid[y, x] <- grid[y, x] + delta + noise[y, x]
#       new_grid[y, x] <- max(min_temp, min(new_grid[y, x], max_temp))
#     }
#   }
#   new_grid
# }


reactor_step <- function(grid, target_temp, stirring) {
  noise <- matrix(rnorm(length(grid), mean = 0, sd = 0.2), nrow = nrow(grid))
  new_grid <- grid
  for (y in 1:grid_size) {
    for (x in 1:grid_size) {
      # Compute a heat balance:
      #  - "Heat input" increases temperature toward target
      #  - If there is no stirring, an extra constant heating is applied.
      #  - "Cooling effect" increases with stirring (dissipative effect).
      heat_input <- 0.1 * (target_temp - grid[y, x]) + (1 - stirring) * 0.05 * target_temp
      cooling_effect <- stirring * 0.05 * grid[y, x]
      delta <- heat_input - cooling_effect
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

get_state_index <- function(state) {
  avg <- state[1]
  std <- state[2]
  if(avg < min_temp || avg > max_temp) return(-1)
  bin_avg <- floor((avg - min_temp) / ((max_temp - min_temp) / N_avg)) + 1
  if(bin_avg > N_avg) bin_avg <- N_avg
  bin_std <- floor(std / (10 / N_std)) + 1
  if(bin_std > N_std) bin_std <- N_std
  state_index <- (bin_avg - 1) * N_std + (bin_std - 1)
  return(state_index)
}

compute_reward <- function(avg_temp, target_temp, stirring) {
  r <- - (abs(avg_temp - goal_temp) + abs(target_temp - goal_temp) + 2 * abs(stirring - 0.5))
  if(abs(avg_temp - goal_temp) < 1 && abs(target_temp - goal_temp) < 1 && abs(stirring - 0.5) < 0.1)
    r <- r + 1
  return(r)
}

get_action <- function(action_index) {
  target_idx <- floor(action_index / num_stir) + 1
  stir_idx <- (action_index %% num_stir) + 1
  c(possible_target[target_idx], possible_stir[stir_idx])
}

softmax_fn <- function(x, temperature) {
  exp_x <- exp((x - max(x)) / temperature)
  exp_x / sum(exp_x)
}

# ---- RL Simulation for Bioreactor Control ----
simulate_bioreactor_control <- function() {
  set.seed(123)
  w <- matrix(runif(N_STATES * num_actions, -1, 1), nrow = N_STATES, ncol = num_actions)
  e <- matrix(0, nrow = N_STATES, ncol = num_actions)
  v <- runif(N_STATES, -1, 1)
  xbar <- rep(0, N_STATES)
  
  grid <- init_grid()
  state_vec <- get_state(grid)
  state_index <- get_state_index(state_vec)
  if(state_index < 0) stop("Initial state failure.")
  
  # Add columns for actor and critic weights in the history logging.
  history <- data.frame(
    GlobalStep = integer(),
    StepInTrial = integer(),
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
  
  trial_lengths <- c()
  global_steps <- 0
  last_failure <- 0
  failures <- 0
  bad_start <- NA
  
  while(failures < MAX_FAILURES && (global_steps - last_failure) < MAX_STEPS) {
    global_steps <- global_steps + 1
    step_in_trial <- global_steps - last_failure
    
    state_index <- get_state_index(state_vec)
    if(state_index < 0) {
      r <- -5
      rhat <- r
      failures <- failures + 1
      trial_lengths <- c(trial_lengths, step_in_trial)
      grid <- init_grid()
      state_vec <- get_state(grid)
      last_failure <- global_steps
      next
    }
    
    p_actions <- softmax_fn(w[state_index + 1, ], SOFTMAX_TEMP)
    action_index <- sample(0:(num_actions - 1), size = 1, prob = p_actions)
    action <- get_action(action_index)
    target_temp <- action[1]
    stirring <- action[2]
    
    old_value <- v[state_index + 1]
    # Log current step including actor and critic weights.
    history <- rbind(history, data.frame(
      GlobalStep = global_steps,
      StepInTrial = step_in_trial,
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
    new_state_index <- get_state_index(new_state_vec)
    
    reward <- compute_reward(new_state_vec[1], target_temp, stirring)
    
    if(new_state_index < 0) {
      if(is.na(bad_start)) bad_start <- step_in_trial
      if((step_in_trial - bad_start) >= allowed_steps_in_bad_state) {
        failures <- failures + 1
        trial_lengths <- c(trial_lengths, step_in_trial)
        p_new <- 0
        failed <- TRUE
        last_failure <- global_steps
        grid <- init_grid()
        new_state_vec <- get_state(grid)
        new_state_index <- get_state_index(new_state_vec)
        bad_start <- NA
      } else {
        p_new <- v[state_index + 1]
        failed <- FALSE
        new_state_index <- state_index
      }
    } else {
      bad_start <- NA
      p_new <- v[new_state_index + 1]
      failed <- FALSE
    }
    
    rhat <- reward + GAMMA * p_new - old_value
    history$rhat[nrow(history)] <- rhat
    history$reward[nrow(history)] <- reward
    
    for(i in 1:N_STATES) {
      noise <- runif(num_actions, -0.01, 0.01)
      w[i, ] <- w[i, ] + ALPHA * rhat * e[i, ] + noise
      v[i] <- v[i] + BETA * rhat * xbar[i]
      if(failed) {
        e[i, ] <- 0.0
        xbar[i] <- 0.0
        
        cat(sprintf("Step: %d | Trial: %d | State: %d -> %d | Temp: %.2f -> %.2f | Action: %d | (Target: %.2f, Stir: %.2f) | p: %.2f | Reward: %.2f | rhat: %.2f | w_val: %.2f | v_val: %.2f\n",
                    global_steps, step_in_trial, state_index, new_state_index,
                    state_vec[1], new_state_vec[1], action_index, target_temp, stirring,
                    p_actions[action_index + 1], reward, rhat,
                    w[state_index + 1, action_index + 1], v[state_index + 1]), "\n")
        
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
    
    cat(sprintf("Step: %d | Trial: %d | State: %d -> %d | Temp: %.2f -> %.2f | Action: %d | (Target: %.2f, Stir: %.2f) | p: %.2f | Reward: %.2f | rhat: %.2f | w_val: %.2f | v_val: %.2f\n",
                global_steps, step_in_trial, state_index, new_state_index,
                state_vec[1], new_state_vec[1], action_index, target_temp, stirring,
                p_actions[action_index + 1], reward, rhat,
                w[state_index + 1, action_index + 1], v[state_index + 1]), "\n")
    
    state_vec <- new_state_vec
  }
  
  list(history = history, trial_lengths = trial_lengths, global_steps = global_steps)
}

# ---- Run Simulation ----
set.seed(42)
start.time <- Sys.time()
result <- simulate_bioreactor_control()
history <- result$history
trial_lengths <- result$trial_lengths
global_steps_final <- result$global_steps
end.time <- Sys.time()
cat("Time taken: ", end.time - start.time, "\n")




stirring_rate <- 0.3
stirring_rate <- 0.0
# ---- Plotting Metrics and Learning Outcomes ----
# Plot: Cell Temperature Metrics over Time (Non-RL Simulation)
timesteps <- 25
results <- list()
grid <- init_grid()
for(t in 1:timesteps) {
  df <- expand.grid(x = 1:grid_size, y = 1:grid_size)
  df$value <- as.vector(t(grid))
  df$timestep <- t
  df$cell <- paste0(df$x, "-", df$y)
  results[[t]] <- df
  grid <- reactor_step(grid, stirring_rate, stirring_rate)  # using constant stirring_rate
}
all_data <- bind_rows(results)
all_data <- all_data %>%
  mutate(temp_state = case_when(
    value > 16 ~ "hot",
    value < 14 ~ "cold",
    TRUE ~ "good"
  ))
p_metric <- ggplot(all_data, aes(x = timestep, y = value, group = cell, color = value)) +
  geom_line() +
  scale_color_gradientn(
    colours = c("white", "blue", "green", "yellow", "red"),
    values = scales::rescale(c(min_temp, 10, goal_temp, 20, max_temp)),
    limits = c(min_temp, max_temp)
  ) +
  labs(title = "Bioreactor Cell Temperatures Over Time",
       x = "Timestep", y = "Temperature") +
  theme_minimal() +
  theme(legend.position = "none")

p_metric

ggsave(filename = "../figures/bioreactor_metric.png", plot = p_metric, width = 8, height = 6)

p_grid <- ggplot(all_data, aes(x = x, y = y, fill = value)) +
  geom_tile(colour = "grey80") +
  geom_text(aes(label = round(value, 1)), size = 3, vjust = -0.5) +
  geom_text(aes(label = temp_state), size = 3, vjust = 1.5) +
  scale_fill_gradientn(
    colours = c("white", "blue", "green", "yellow", "red"),
    values = scales::rescale(c(min_temp, 10, goal_temp, 20, max_temp)),
    limits = c(min_temp, max_temp)
  ) +
  coord_fixed() +
  labs(title = "Bioreactor Temperature Grid",
       subtitle = "Timestep: {frame_time}",
       x = "", y = "",
       caption = paste0("Parameters: \nGrid size = ", grid_size,
                        "\nGoal Temp = ", goal_temp,
                        "\nMax Temp = ", max_temp,
                        "\nMin Temp = ", min_temp,
                        "\nTimesteps = ", timesteps,
                        "\nStirring rate = ", stirring_rate)
  ) +
  theme_minimal() +
  theme(legend.position = "right") +
  transition_time(timestep) +
  shadow_mark(past = TRUE, alpha = 0.5, size = 1)

p_grid

anim1 <- animate(p_grid, nframes = timesteps, fps = 5, width = 500, height = 500)
anim_save("../figures/fig_ex_3_1_bioreactor_temp.gif", animation = anim1)

if(length(trial_lengths) == 0) {
  df_trials <- data.frame(Trial = 1, Steps = NA)
} else {
  df_trials <- data.frame(
    Trial = seq_along(trial_lengths),
    Steps = trial_lengths
  )
}
p1_rl <- ggplot(df_trials, aes(x = Trial, y = Steps)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  labs(subtitle = "Trial Lengths (Steps per Trial)",
       x = "Trial Number", y = "Steps") +
  theme_minimal()
p1_rl

p2_rl <- ggplot(history, aes(x = GlobalStep, y = rhat)) +
  geom_line(color = "darkred") +
  labs(subtitle = "Evolution of Reinforcement Signal (rhat)",
       x = "Global Step", y = "rhat") +
  theme_minimal()
# 
# p3_rl <- ggplot(history, aes(x = GlobalStep)) +
#   geom_point(aes(y = w_val, color = "Actor Weight (w)"), size = 0.5) +
#   geom_point(aes(y = v_val, color = "Critic Weight (v)"), size = 0.5) +
#   labs(subtitle = "Evolution of Weights in Current State",
#        x = "Global Step", y = "Weight Value") +
#   scale_color_manual(name = "Weight Type",
#                      values = c("Actor Weight (w)" = "darkblue",
#                                 "Critic Weight (v)" = "salmon")) +
#   theme_minimal()

p3_rl <- ggplot(history, aes(x = GlobalStep)) +
  geom_point(aes(y = avg_temp, color = "Average Temperature"), size = 0.5) +
  labs(subtitle = "Evolution of Average Temperature in Current State",
       x = "Global Step", y = "Temperature") +
  scale_color_manual(name = "Metric",
                     values = c("Average Temperature" = "darkblue")) +
  theme_minimal()


p4_rl <- ggplot(history, aes(x = GlobalStep, y = p_action)) +
  geom_point(color = "purple", size = 0.5) +
  labs(subtitle = "Evolution of Action Probability (Chosen Stirring)",
       x = "Global Step", y = "Probability") +
  theme_minimal()

rl_metrics <- (p1_rl / p2_rl) | (p3_rl / p4_rl)
rl_metrics <- rl_metrics + 
  plot_annotation(title = "Figure Example 3.1: Bioreactor Control Learning Dynamics")
combined_all <- p_metric / rl_metrics
print(combined_all)
filename <- file.path(paste0("../figures/fig_ex_3_1.png"))
ggsave(filename = filename, combined_all, width = 8, height = 10)

# final learnt plot ----
# 
# p_anim <- ggplot(history, aes(x = GlobalStep, y = avg_temp)) +
#   geom_line(color = "darkgreen") +
#   geom_point(aes(color = factor(StateIndex)), size = 1) +
#   labs(title = "Evolution of Reactor Average Temperature",
#        subtitle = paste("Goal Temp =", goal_temp,
#                         "| Fine Actions: Target =", paste(round(possible_target, 2), collapse = ", "),
#                         " & Stir =", paste(round(possible_stir, 2), collapse = ", "),
#                         "\nTotal Trials:", length(trial_lengths),
#                         "| Global Step: {round(frame_time,0)}"),
#        x = "Global Step", y = "Average Temperature")
#   transition_time(GlobalStep) +
#   shadow_mark(past = TRUE, alpha = 0.5, size = 1)
#   
# anim2 <- animate(p_anim, nframes = 100, fps = 10, width = 400, height = 300)
# 
# gif_filename <- file.path(paste0("../figures/fig_ex_3_1_anim.gif"))
# anim_save(gif_filename, anim2)
# 

total_trials <- ifelse(is.null(trial_lengths) || length(trial_lengths) == 0, 0, length(trial_lengths))


# First, add a column to history for dynamic subtitle text.
history <- history %>%
  mutate(subtitle_text = paste("StateIndex:", StateIndex,
                               "\nrhat:", round(rhat, 2),
                               "\nAction:", chosen_action,
                               "\nTarget:", target,
                               "\nStir:", stirring,
                               "\nReward:", round(reward, 2),
                               "\nAvg:", round(avg_temp, 2),
                               "\nStd:", round(std_temp, 2),
                               "\nw:", round(w_val, 2),
                               "\nv:", round(v_val, 2),
                               "\np:", round(p_action, 2)))

p_anim <- ggplot(history, aes(x = GlobalStep, y = avg_temp)) +
  geom_line(color = "darkgreen") +
  geom_point(aes(color = avg_temp), size = 1) +
  scale_color_gradientn(
    colours = c("white", "blue", "green", "yellow", "red"),
    values = scales::rescale(c(min_temp, 10, goal_temp, 20, max_temp)),
    limits = c(min_temp, max_temp)
  ) +
  labs(title = "Evolution of Reactor Average Temperature\n",
       subtitle = paste("Goal Temp =", goal_temp,
                        "| Fine Actions: Target =", paste(round(possible_target, 2), collapse = ", "),
                        " & Stir =", paste(round(possible_stir, 2), collapse = ", "),
                        "\nTotal Trials:", length(trial_lengths),
                        "| Global Step: {round(frame_time,0)}"),
       x = "Global Step", y = "Average Temperature") +
  # Add dynamic subtitle text as a geom_text layer at a fixed position.
  geom_text(aes(x = Inf, y = -Inf, label = subtitle_text),
            hjust = 1.1, vjust = -0.1, size = 3, color = "black", inherit.aes = FALSE) +
  transition_time(GlobalStep) +
  # Exclude the geom_text layer (which is layer 3) from being shadowed
  shadow_mark(exclude_layer = 3, past = TRUE, alpha = 0.5, size = 1)



anim2 <- animate(p_anim, nframes = 100, fps = 10, width = 700, height = 500, res = 120)
gif_filename <- file.path(paste0("../figures/fig_ex_3_1_anim.gif"))
anim_save(gif_filename, anim2)

