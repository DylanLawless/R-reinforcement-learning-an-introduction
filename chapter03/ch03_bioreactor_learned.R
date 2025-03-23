library(ggplot2); theme_set(theme_bw())
library(dplyr)
library(tidyr)
library(gganimate)
library(patchwork)
set.seed(666)

### 1. Environment (Reactor) Setup ###
grid_size <- 3
init_temp <- 20       # initial temperature
goal_temp <- 30       # (unused now)
limit_min <- 25 # 25
limit_max <- 35 # 35
min_temp <- 10
max_temp <- 50
base_heat <- 1  # constant heating term to drive rapid heating
rand_heat_min <- 0
rand_heat_max <- 0
cooling_constant = 1

# RL and simulation parameters
EPISODE_MAX_FAIL <- 50   # maximum episodes to try
MAX_STEPS <- 1000         # maximum steps per episode
STABILISE_STEPS <- 50    # steps to allow reactor to stabilise before checking avg_temp

# RL algorithm parameters
ALPHA <- 1000            # actor learning rate
BETA <- 0.5              # critic learning rate
GAMMA <- 0.95            # discount factor
LAMBDAw <- 0.9           # actor eligibility trace decay
LAMBDAv <- 0.8           # critic eligibility trace decay

# Actions: stirring levels only
num_stir <- 8
possible_stir <- seq(0, 2, length.out = num_stir)
num_actions <- num_stir

# UCB exploration bonus constant
UCB_C <- 2 # 2 at init_temp 30

### State Discretisation ###
N_avg <- 5   # number of bins for average temperature
N_std <- 4   # number of bins for standard deviation
N_STATES <- N_avg * N_std

avg_lower_bound <- min_temp
avg_upper_bound <- max_temp
std_lower_bound <- 0
std_upper_bound <- 10

discretise <- function(avg, std) {
  if(is.na(avg) || is.na(std)) return(NA)
  if(avg < min_temp || avg > max_temp) return(NA)
  bin_avg <- floor((avg - avg_lower_bound) / ((avg_upper_bound - avg_lower_bound)/ N_avg)) + 1
  if(bin_avg > N_avg) bin_avg <- N_avg
  bin_std <- floor((std - std_lower_bound) / ((std_upper_bound - std_lower_bound)/ N_std)) + 1
  if(bin_std > N_std) bin_std <- N_std
  (bin_avg - 1) * N_std + (bin_std - 1)
}

get_state_index <- function(state) {
  avg <- state[1]
  std <- state[2]
  if(is.na(avg) || is.na(std)) return(list(index = -1, bin_avg = NA, bin_std = NA, avg = avg, std = std))
  if(avg < min_temp || avg > max_temp) return(list(index = -1, bin_avg = NA, bin_std = NA, avg = avg, std = std))
  bin_avg <- floor((avg - avg_lower_bound) / ((avg_upper_bound - avg_lower_bound)/ N_avg)) + 1
  if(bin_avg > N_avg) bin_avg <- N_avg
  bin_std <- floor((std - std_lower_bound) / ((std_upper_bound - std_lower_bound)/ N_std)) + 1
  if(bin_std > N_std) bin_std <- N_std
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

# reactor_step <- function(grid, stirring) {
#   noise <- matrix(rnorm(length(grid), mean = 0, sd = 0.5), nrow = nrow(grid))
#   new_grid <- grid
#   # base_heat <- 2  # constant heating term to drive rapid heating
#   for (y in 1:grid_size) {
#     for (x in 1:grid_size) {
#       # slight random variations around zero
#       random_heat <- runif(1, min = 0, max = 0.5)
#       nb_avg <- get_neighbours(grid, x, y)
#       diffusion <- stirring * (nb_avg - grid[y, x])
#       cooling <- stirring * 0.05 * grid[y, x]
#       # use the constant base_heat plus small random fluctuations
#       delta <- base_heat + (1 - stirring) * random_heat + diffusion - cooling
#       new_grid[y, x] <- grid[y, x] + delta + noise[y, x]
#       new_grid[y, x] <- max(min_temp, min(new_grid[y, x], max_temp))
#     }
#   }
#   new_grid
# }

# reactor_step <- function(grid, stirring) {
#   # new_grid: we start with the current grid values.
#   new_grid <- grid
#   
#   # Cooling amount scales with stirring (acting like "pressure").
#   cooling_amount <- 1 * stirring
#   
#   # Loop over each cell
#   for (y in 1:grid_size) {
#     for (x in 1:grid_size) {
#       # Generate a small random heating term based on the globally set bounds.
#       random_heat <- runif(1, min = rand_heat_min, max = rand_heat_max)
#       
#       # Compute the new temperature:
#       # Add the constant base_heat plus a small random heating scaled down by (1 - stirring)
#       new_temp <- grid[y, x] + base_heat + (1 - stirring) * random_heat
#       
#       # With probability equal to stirring, apply a cooling event (simulate pressure)
#       if (runif(1) < stirring) {
#         new_temp <- new_temp - cooling_amount
#       }
#       
#       # Clamp the new temperature to be within [min_temp, max_temp]
#       new_temp <- max(min_temp, min(new_temp, max_temp))
#       
#       # Update the cell value
#       new_grid[y, x] <- new_temp
#     }
#   }
#   new_grid
# }


reactor_step <- function(grid, stirring) {
  new_grid <- grid
  cooling_effect <- stirring * cooling_constant  # e.g., cooling_constant = 1
  for (y in 1:grid_size) {
    for (x in 1:grid_size) {
      random_heat <- runif(1, min = rand_heat_min, max = rand_heat_max)
      new_temp <- grid[y, x] + base_heat + (1 - stirring) * random_heat - cooling_effect
      new_grid[y, x] <- max(min_temp, min(new_temp, max_temp))
    }
  }
  new_grid
}


get_state <- function(grid) {
  avg <- mean(grid)
  std <- sd(as.vector(grid))
  c(avg, std)
}

# New reward: reward = 1 if avg_temp in [limit_min, limit_max], otherwise penalize.
# static +1 / -1
compute_reward <- function(avg_temp, stirring) {
  if(avg_temp >= limit_min && avg_temp <= limit_max) {
    r <- 1
  } else if(avg_temp < limit_min) {
    r <- -(limit_min - avg_temp)
  } else {
    r <- -(avg_temp - limit_max)
  }
  r
}

# # ramps with distance from goal
# compute_reward <- function(avg_temp, stirring, k = 0.1) {
#   if(avg_temp >= limit_min && avg_temp <= limit_max) {
#     r <- 1
#   } else if(avg_temp < limit_min) {
#     err <- limit_min - avg_temp
#     r <- 1 - k * (err^2)
#   } else { # avg_temp > limit_max
#     err <- avg_temp - limit_max
#     r <- 1 - k * (err^2)
#   }
#   r
# }


# New get_action: return stirring value only
get_action <- function(action_index) {
  possible_stir[action_index + 1]
}

### 4. RL Simulation Setup with UCB metrics ###
simulate_bioreactor_control <- function() {
  set.seed(123)
  w <- matrix(runif(N_STATES * num_actions, -1, 1), nrow = N_STATES, ncol = num_actions)
  e <- matrix(0, nrow = N_STATES, ncol = num_actions)
  v <- runif(N_STATES, -1, 1)
  xbar <- rep(0, N_STATES)
  
  count_mat <- matrix(0, nrow = N_STATES, ncol = num_actions)
  
  history <- data.frame(
    GlobalStep = integer(),
    Episode = integer(),
    StepInEpisode = integer(),
    StateIndex = integer(),
    rhat = numeric(),
    chosen_action = integer(),
    stirring = numeric(),
    reward = numeric(),
    avg_temp = numeric(),
    std_temp = numeric(),
    w_val = numeric(),
    v_val = numeric(),
    ucb_val = numeric(),  # record UCB value for chosen action
    error = numeric()     # record error from desired range
  )
  
  overall_steps <- 0
  success <- FALSE
  
  for (episode in 1:EPISODE_MAX_FAIL) {
    step_in_episode <- 0
    grid <- init_grid()
    state_vec <- get_state(grid)
    state_info <- get_state_index(state_vec)
    if (is.null(state_info) || state_info$index < 0) next
    state_index <- state_info$index
    
    while(step_in_episode < MAX_STEPS) {
      overall_steps <- overall_steps + 1
      step_in_episode <- step_in_episode + 1
      
      ucb_values <- rep(NA, num_actions)
      for(a in 0:(num_actions - 1)) {
        count_val <- count_mat[state_index + 1, a + 1]
        if(count_val == 0) {
          ucb_values[a + 1] <- Inf
        } else {
          ucb_values[a + 1] <- w[state_index + 1, a + 1] + 
            UCB_C * sqrt(log(overall_steps + 1) / count_val)
        }
      }
      max_idx <- which.max(ucb_values)
      if(length(max_idx) == 0 || is.na(max_idx)) break
      action_index <- max_idx - 1
      count_mat[state_index + 1, action_index + 1] <- count_mat[state_index + 1, action_index + 1] + 1
      
      # For UCB, record the UCB value for the chosen action
      chosen_count <- count_mat[state_index + 1, action_index + 1]
      if(chosen_count == 0) {
        ucb_bonus <- NA
      } else {
        ucb_bonus <- UCB_C * sqrt(log(overall_steps + 1) / chosen_count)
      }
      ucb_val <- w[state_index + 1, action_index + 1] + ucb_bonus
      
      stirring <- get_action(action_index)
      old_value <- v[state_index + 1]
      
      # Compute error: difference from desired range
      if(state_vec[1] < limit_min) {
        err <- limit_min - state_vec[1]
      } else if(state_vec[1] > limit_max) {
        err <- state_vec[1] - limit_max
      } else {
        err <- 0
      }
      
      new_row <- data.frame(
        GlobalStep = overall_steps,
        Episode = episode,
        StepInEpisode = step_in_episode,
        StateIndex = state_index,
        rhat = NA,
        chosen_action = action_index,
        stirring = stirring,
        reward = NA,
        avg_temp = state_vec[1],
        std_temp = state_vec[2],
        bin_avg = state_info$bin_avg,
        bin_std = state_info$bin_std,
        w_val = w[state_index + 1, action_index + 1],
        v_val = v[state_index + 1],
        ucb_val = ucb_val,
        error = err
      )
      
      history <- rbind(history, new_row)
      
      grid <- reactor_step(grid, stirring)
      new_state_vec <- get_state(grid)
      new_state_info <- get_state_index(new_state_vec)
      if(is.null(new_state_info) || new_state_info$index < 0) break
      new_state_index <- new_state_info$index
      
      reward <- compute_reward(new_state_vec[1], stirring)
      
      failed <- (any(grid < min_temp) || any(grid > max_temp) ||
                   (step_in_episode > STABILISE_STEPS && 
                      (new_state_vec[1] < limit_min || new_state_vec[1] > limit_max)))
      if(failed) break
      
      p_new <- v[new_state_index + 1]
      rhat <- reward + GAMMA * p_new - old_value
      history$rhat[nrow(history)] <- rhat
      history$reward[nrow(history)] <- reward
      
      for(i in 1:N_STATES) {
        noise <- runif(num_actions, -0.01, 0.01)
        w[i, ] <- w[i, ] + ALPHA * rhat * e[i, ] + noise
        v[i] <- v[i] + BETA * rhat * xbar[i]
        e[i, ] <- e[i, ] * LAMBDAw
        xbar[i] <- xbar[i] * LAMBDAv
      }
      
      grad <- rep(0, num_actions)
      grad[action_index + 1] <- 1
      e[state_index + 1, ] <- e[state_index + 1, ] + (1 - LAMBDAw) * grad
      xbar[state_index + 1] <- xbar[state_index + 1] + (1 - LAMBDAv)
      
      state_vec <- new_state_vec
      state_info <- new_state_info
      state_index <- new_state_info$index
      
      if(step_in_episode %% 50 == 0) {
        # cat(sprintf("Reward %.2f | Stirring %.2f | Step %d | Ep: %d | EpLen: %d | Temp: %.2f | Error: %.2f | UCB: %.2f\n",
        #             reward, stirring, overall_steps, episode, step_in_episode, 
        #             new_state_vec[1], err, ucb_val), "\n")
        
        cat(sprintf("Reward %.2f | Stirring %.2f | Step %d | Ep: %d | EpLen: %d | Temp: %.2f | Error: %.2f | UCB: %.2f\nDiscretisation: Avg=%.2f (bin %d), Std=%.2f (bin %d)\n",
                    reward, stirring, overall_steps, episode, step_in_episode, 
                    new_state_vec[1], err, ucb_val,
                    new_state_info$avg, new_state_info$bin_avg, new_state_info$std, new_state_info$bin_std), "\n")
        
        
      }
      
      if(step_in_episode >= MAX_STEPS) {
        final_state_info <- get_state_index(state_vec)
        cat(sprintf("SUCCESS: Terminated after %d steps in Episode %d | Final Temp: %.2f | Error: %.2f\n",
                    step_in_episode, episode, state_vec[1],
                    ifelse(state_vec[1] < limit_min, limit_min - state_vec[1],
                           ifelse(state_vec[1] > limit_max, state_vec[1] - limit_max, 0))), "\n")
        success <- TRUE
        return(list(history = history, episode = episode, steps = step_in_episode))
      }
    }
  }
  
  if(!success) {
    final_state_info <- get_state_index(state_vec)
    cat(sprintf("FAILURE: No episode reached %d steps over %d episodes | Final Temp: %.2f\n",
                MAX_STEPS, EPISODE_MAX_FAIL, state_vec[1]), "\n")
  }
  
  list(history = history, episode = episode, steps = step_in_episode)
}

result <- simulate_bioreactor_control()
history <- result$history
episode_lengths <- result$episode_lengths
global_steps_final <- result$global_steps

# end of sim function ----

### 3. Testing the Reactor Model (No RL) ###
# Simulate reactor dynamics with fixed stirring rates: 0, 0.3, 0.6.
fixed_stir_rates <- possible_stir # same as agent's options

timesteps <- STABILISE_STEPS
results_list <- list()

# for (stir in fixed_stir_rates) {
#   grid <- init_grid()
#   sim_results <- list()
#   for (t in 1:timesteps) {
#     df <- expand.grid(x = 1:grid_size, y = 1:grid_size)
#     df$value <- as.vector(t(grid))
#     df$timestep <- t
#     df$cell <- paste0(df$x, "-", df$y)
#     sim_results[[t]] <- df
#     grid <- reactor_step(grid, target_temp = goal_temp, stirring = stir)
#   }

for (stir in fixed_stir_rates) {
  grid <- init_grid()
  sim_results <- list()
  for (t in 1:timesteps) {
    df <- expand.grid(x = 1:grid_size, y = 1:grid_size)
    df$value <- as.vector(t(grid))
    df$timestep <- t
    df$cell <- paste0(df$x, "-", df$y)
    sim_results[[t]] <- df
    grid <- reactor_step(grid, stirring = stir)
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

# start.time <- Sys.time()
# result <- simulate_bioreactor_control()
# history <- result$history
# episode_lengths <- result$episode_lengths
# global_steps_final <- result$global_steps
# end.time <- Sys.time()
# cat("Time taken: ", end.time - start.time, "\n")












### 4. Plotting ###
# ---- RL Outcome Plots ----
# if(length(episode_lengths) == 0) {
#   df_episodes <- data.frame(Episode = 1, Steps = NA)
# } else {
#   df_episodes <- data.frame(
#     Episode = seq_along(episode_lengths),
#     Steps = episode_lengths
#   )
# }
# p1_rl <- ggplot(df_episodes, aes(x = Episode, y = Steps)) +
#   geom_bar(stat = "identity", fill = "steelblue") +
#   labs(subtitle = "Episode Lengths (Steps per Episode)",
#        x = "Episode Number", y = "Steps") +
#   theme_minimal()
# print(p1_rl)
# 
# p2_rl <- ggplot(history, aes(x = GlobalStep, y = rhat)) +
#   geom_line(color = "darkred") +
#   labs(subtitle = "Evolution of Reinforcement Signal (rhat)",
#        x = "Global Step", y = "rhat") +
#   theme_minimal()
# print(p2_rl)
# 
# p3_rl <- ggplot(history, aes(x = GlobalStep)) +
#   geom_point(aes(y = avg_temp, color = "Average Temperature"), size = 0.5) +
#   labs(subtitle = "Evolution of Average Temperature in Current State",
#        x = "Global Step", y = "Temperature") +
#   scale_color_manual(name = "Metric",
#                      values = c("Average Temperature" = "darkblue")) +
#   theme_minimal()
# print(p3_rl)
# 
# p4_rl <- ggplot(history, aes(x = GlobalStep, y = p_action)) +
#   geom_point(color = "purple", size = 0.5) +
#   labs(subtitle = "Evolution of Action Probability (Chosen Stirring)",
#        x = "Global Step", y = "Probability") +
#   theme_minimal()
# print(p4_rl)
# 
# rl_metrics <- (p1_rl / p2_rl) | (p3_rl / p4_rl)
# rl_metrics <- rl_metrics + 
#   plot_annotation(title = "Figure Example 3.1: Bioreactor Control Learning Dynamics")
# combined_all <- p_metric / rl_metrics
# print(combined_all)
# filename <- file.path("../figures/fig_ex_3_1.png")
# ggsave(filename = filename, combined_all, width = 8, height = 10)
# 
# 
# 



### 4. Plotting using New Metrics ###

# Episode lengths plot remains unchanged
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

# Plot evolution of the reinforcement signal (rhat)
p2_rl <- ggplot(history, aes(x = GlobalStep, y = rhat)) +
  geom_line(color = "darkred") +
  labs(subtitle = "Evolution of Reinforcement Signal (rhat)",
       x = "Global Step", y = "rhat") +
  theme_minimal()
print(p2_rl)

# Plot evolution of average temperature along with its discretisation (bins)
p3_rl <- ggplot(history, aes(x = GlobalStep)) +
  geom_point(aes(y = avg_temp, color = "Average Temperature"), size = 0.5) +
  # geom_text(aes(y = Inf, label = paste0("Avg=", round(avg_temp, 2), " (bin ", bin_avg, ")")),
            # check_overlap = TRUE, hjust = -0.1, size = 2) +
  labs(subtitle = "Evolution of Average Temperature & Discretisation",
       x = "Global Step", y = "Temperature") +
  scale_color_manual(name = "Metric",
                     values = c("Average Temperature" = "darkblue")) +
  theme_minimal()
print(p3_rl)

# Plot evolution of the UCB value for the chosen action
p4_rl <- ggplot(history, aes(x = GlobalStep, y = ucb_val)) +
  geom_point(color = "purple", size = 0.5) +
  labs(subtitle = "Evolution of UCB Value (Chosen Stirring)",
       x = "Global Step", y = "UCB Value") +
  theme_minimal()
print(p4_rl)

# Plot evolution of error from the desired range ([limit_min, limit_max])
p5_rl <- ggplot(history, aes(x = GlobalStep, y = error)) +
  geom_line(color = "orange") +
  labs(subtitle = "Evolution of Temperature Error from Desired Range",
       x = "Global Step", y = "Error") +
  theme_minimal()
print(p5_rl)

# Combine plots using patchwork (adjust layout as desired)
rl_metrics <- p_metric / ((p1_rl / p2_rl) | (p3_rl / p4_rl / p5_rl))
rl_metrics <-   rl_metrics + 
  plot_annotation(title = "Figure Example 3.1: Bioreactor Control Learning Dynamics (New Metrics)")
print(rl_metrics)

filename <- file.path("../figures/fig_ex_3_1.png")
ggsave(filename = filename, rl_metrics, width = 8, height = 10)


















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


# static ----
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

ggsave(filename = "../figures/fig_ex_1_anim_reward.png", plot = p_reward_static, width = 8, height = 5)


p_temp_static <- ggplot(history, aes(x = GlobalStep, y = avg_temp)) +
  geom_line(color = "darkgreen") +
  geom_hline(yintercept = limit_min, linetype = "dotted", color = "darkred") +
  geom_hline(yintercept = limit_max, linetype = "dotted", color = "darkred") +
  geom_point(aes(color = avg_temp), size = 1) +
  scale_color_gradientn(
    colours = c("white", "blue", "green", "yellow", "red"),
    values = scales::rescale(c(min_temp, limit_min, goal_temp, limit_max, max_temp)),
    limits = c(min_temp, max_temp)
  ) +
  labs(title = "Evolution of Reactor Average Temperature",
       x = "Global Step", y = "Average Temperature")

p_temp_static

ggsave(filename = "../figures/fig_ex_1_temp_static.png", plot = p_temp_static, width = 8, height = 5)






# 
# 
# 
# # main animation ----
# 
# # First, compute a dynamic subtitle column in history using the new metrics.
# history <- history %>%
#   mutate(dynamic_subtitle = paste("Goal Temp =", goal_temp,
#                                   "| Fine Action: Stir =", paste(round(possible_stir, 2), collapse = ", "),
#                                   "\nEpisode:", Episode,
#                                   "\nStepInEpisode:", StepInEpisode,
#                                   "\nStateIndex:", StateIndex,
#                                   "\nrhat:", round(rhat, 2),
#                                   "\nAction:", chosen_action,
#                                   "\nReward:", round(reward, 2),
#                                   "\nAvg Temp:", round(avg_temp, 2),
#                                   "\nStd Temp:", round(std_temp, 2),
#                                   "\nw:", round(w_val, 2),
#                                   "\nv:", round(v_val, 2),
#                                   "\np:", round(p_action, 2),
#                                   "\nGlobal Step:", GlobalStep,
#                                   "\nStirring:",history$stirring))
# 
# # Then, include this dynamic subtitle as an on-plot text layer (rather than in labs)
# p_anim <- ggplot(history, aes(x = GlobalStep, y = avg_temp)) +
#   geom_line(color = "darkgreen") +
#   geom_point(aes(color = avg_temp), size = 1) +
#   # Add dynamic text showing the current state's details in the top-right corner:
#   geom_text(aes(label = dynamic_subtitle), 
#             x = -Inf, y = Inf, hjust = 0, vjust = 1.1, size = 3, color = "black") +
#   
#   scale_color_gradientn(
#     colours = c("white", "blue", "green", "yellow", "red"),
#     values = scales::rescale(c(min_temp, limit_min, goal_temp, limit_max, max_temp)),
#     limits = c(min_temp, max_temp)
#   ) +
#   labs(title = "Evolution of Reactor Average Temperature",
#        x = "Global Step", y = "Average Temperature") +
#   transition_time(GlobalStep) +
#   shadow_mark(exclude_layer = 3, past = TRUE, alpha = 0.5, size = 1)
# 
# # anim2 <- animate(p_anim, nframes = 100, fps = 10, width = 700, height = 500, res = 120)
# anim2 <- animate(p_anim, nframes = 200, fps = 20, width = 800, height = 500, res = 120)
# gif_filename <- file.path("../figures/fig_ex_3_1_anim.gif")
# anim_save(gif_filename, anim2)
# 
# # plot reward ----
# 
# p_anim_reward <- ggplot(history, aes(x = GlobalStep, y = reward)) +
#   geom_line(color = "darkgreen") +
#   geom_point(aes(color = avg_temp), size = 1) +
#   # Add dynamic text showing the current state's details in the top-left corner:
#   geom_text(aes(label = dynamic_subtitle), 
#             x = -Inf, y = Inf, hjust = 0, vjust = 1.1, size = 3, color = "black") +
#   scale_color_gradientn(
#     colours = c("white", "blue", "green", "yellow", "red"),
#     values = scales::rescale(c(min_temp, limit_min, goal_temp, limit_max, max_temp)),
#     limits = c(min_temp, max_temp)
#   ) +
#   labs(title = "Evolution of Reactor Reward",
#        x = "Global Step", y = "Reward") +
#   transition_time(GlobalStep) +
#   shadow_mark(exclude_layer = 3, past = TRUE, alpha = 0.5, size = 1)
# 
# anim_reward <- animate(p_anim_reward, nframes = 200, fps = 20, width = 800, height = 500, res = 120)
# gif_filename_reward <- file.path("../figures/fig_ex_1_anim_reward.gif")
# anim_save(gif_filename_reward, anim_reward)
# 
