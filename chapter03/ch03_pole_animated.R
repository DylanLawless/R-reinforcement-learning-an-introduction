library(ggplot2); theme_set(theme_bw())
library(patchwork)
library(gganimate)
library(dplyr)

# ---- RL Parameters for Temperature Control ----
N_BOXES <- 20          # Number of discrete temperature bins
num_actions <- 5       # Number of stirring options
ALPHA <- 100           # Learning rate for actor weights (reduced to allow gradual changes)
BETA <- 0.1            # Learning rate for critic weights
GAMMA <- 0.95          # Discount factor for critic
LAMBDAw <- 0.9         # Decay rate for actor eligibility trace
LAMBDAv <- 0.8         # Decay rate for critic eligibility trace
MAX_FAILURES <- 100    # Maximum number of failures before termination
MAX_STEPS <- 5000      # Maximum steps in a trial
allowed_steps_in_bad_temp <- 50  # Steps allowed in a persistently bad state

# ---- Reactor Environment Parameters ----
grid_size <- 3        # Reactor grid is grid_size x grid_size
goal_temp <- 15       # Desired average temperature
max_temp <- 60        # Maximum possible temperature (cap)
min_temp <- 0         # Minimum possible temperature

# Define fine-grained stirring values (actions)
possible_stirring <- seq(0.2, 0.8, length.out = num_actions)

# Softmax function for action probabilities with temperature parameter (set to 1)
softmax <- function(x) {
  exp_x <- exp(x - max(x))
  exp_x / sum(exp_x)
}

# ---- Reactor Simulation Functions ----

# Initialize reactor grid: all cells start at goal temperature.
init_grid <- function() {
  matrix(goal_temp, nrow = grid_size, ncol = grid_size)
}

# Compute average temperature of neighbours for cell (x,y)
get_neighbours <- function(grid, x, y) {
  indices <- expand.grid(dx = -1:1, dy = -1:1) %>%
    filter(!(dx == 0 & dy == 0)) %>%
    mutate(nx = x + dx, ny = y + dy) %>%
    filter(nx >= 1, nx <= grid_size, ny >= 1, ny <= grid_size)
  mean(sapply(1:nrow(indices), function(i) grid[indices$ny[i], indices$nx[i]]))
}

# Simulate one reactor step:
# A random cell receives a heat boost; then stirring mixes temperatures.
reactor_step <- function(grid, stirring) {
  rx <- sample(1:grid_size, 1)
  ry <- sample(1:grid_size, 1)
  grid[ry, rx] <- min(grid[ry, rx] + 3, max_temp)
  
  new_grid <- grid
  for (y in 1:grid_size) {
    for (x in 1:grid_size) {
      nb_avg <- get_neighbours(grid, x, y)
      new_grid[y, x] <- (1 - stirring) * grid[y, x] + stirring * nb_avg
      new_grid[y, x] <- max(min_temp, min(new_grid[y, x], max_temp))
    }
  }
  new_grid
}

# Discretise the average temperature into one of N_BOXES bins.
# Temperatures outside [5,30] are considered failure (returns -1).
get_box <- function(avg_temp) {
  if (avg_temp < 5 || avg_temp > 30) return(-1)
  bin_width <- (30 - 5) / N_BOXES
  bin <- floor((avg_temp - 5) / bin_width)
  if (bin >= N_BOXES) bin <- N_BOXES - 1
  return(bin)
}

# ---- RL Simulation for Temperature Control ----
simulate_temperature_control <- function() {
  # Initialise actor (w) and critic (v) weights and eligibility traces.
  # Initialize actor weights with small random numbers to promote exploration.
  w <- matrix(runif(N_BOXES * num_actions, -0.1, 0.1), nrow = N_BOXES, ncol = num_actions)
  e <- matrix(0, nrow = N_BOXES, ncol = num_actions)
  v <- rep(0, N_BOXES)
  xbar <- rep(0, N_BOXES)
  
  grid <- init_grid()
  state <- mean(grid)   # Initial average temperature
  box <- get_box(state)
  if (box < 0) stop("Initial state out of acceptable range.")
  
  history <- data.frame(
    GlobalStep = integer(),
    StepInTrial = integer(),
    Box = integer(),
    rhat = numeric(),
    chosen_action = integer(),
    stirring = numeric(),
    w_val = numeric(),
    v_val = numeric(),
    p_action = numeric(),
    state = numeric()
  )
  
  trial_lengths <- c()
  global_steps <- 0
  last_failure <- 0
  failures <- 0
  bad_start <- NA  # Step when a persistently bad state starts
  
  while (failures < MAX_FAILURES && (global_steps - last_failure) < MAX_STEPS) {
    global_steps <- global_steps + 1
    step_in_trial <- global_steps - last_failure
    
    # Compute action probabilities using softmax.
    p_actions <- softmax(w[box + 1, ])
    action <- sample(1:num_actions, size = 1, prob = p_actions)
    stirring <- possible_stirring[action]
    
    # Compute gradient and update eligibility traces for current box.
    grad <- rep(0, num_actions)
    grad[action] <- 1
    grad <- grad - p_actions
    e[box + 1, ] <- e[box + 1, ] + (1 - LAMBDAw) * grad
    xbar[box + 1] <- xbar[box + 1] + (1 - LAMBDAv)
    old_p <- v[box + 1]
    
    # Log current state before applying action.
    history <- rbind(history, data.frame(
      GlobalStep = global_steps,
      StepInTrial = step_in_trial,
      Box = box,
      rhat = NA,
      chosen_action = action,
      stirring = stirring,
      w_val = w[box + 1, action],
      v_val = v[box + 1],
      p_action = p_actions[action],
      state = state
    ))
    
    # Apply action: update reactor state.
    grid <- reactor_step(grid, stirring)
    state_new <- mean(grid)
    new_box <- get_box(state_new)
    
    # Determine reward and whether state is persistently bad.
    if (new_box < 0) {
      if (is.na(bad_start)) { bad_start <- step_in_trial }
      if ((step_in_trial - bad_start) >= allowed_steps_in_bad_temp) {
        failures <- failures + 1
        trial_length <- global_steps - last_failure
        trial_lengths <- c(trial_lengths, trial_length)
        r <- -1.0   # Penalty for failure.
        p_new <- 0.0
        failed <- TRUE
        last_failure <- global_steps
        grid <- init_grid()
        state_new <- mean(grid)
        new_box <- get_box(state_new)
        bad_start <- NA
        
        # Detailed console logging.
        cat(sprintf("Step: %d | Trial Step: %d | Box: %d -> %d | Temp: %.2f -> %.2f | Action: %d | Stir: %.2f | p_action: %.2f | Reward: %.2f | rhat: %.2f | w: [%.2f] | v: %.2f\n",
                    global_steps, step_in_trial, box, new_box, state, state_new, action, stirring, p_actions[action], r, rhat, 
                    w[box + 1, action], v[box + 1]),
            "\n")
        
      } else {
        r <- -abs(state_new - goal_temp)
        p_new <- v[box + 1]
        failed <- FALSE
        new_box <- box
      }
    } else {
      bad_start <- NA
      # Reward is positive if within 1 degree of the goal, else proportional penalty.
      if (abs(state_new - goal_temp) < 1) {
        r <- 1.0
      } else {
        r <- -abs(state_new - goal_temp)
      }
      p_new <- v[new_box + 1]
      failed <- FALSE
    }
    
    rhat <- r + GAMMA * p_new - old_p
    history$rhat[nrow(history)] <- rhat  # Log temporal-difference error.
    
    # Update weights and eligibility traces for all state bins.
    for (i in 1:N_BOXES) {
      w[i, ] <- w[i, ] + ALPHA * rhat * e[i, ]
      v[i] <- v[i] + BETA * rhat * xbar[i]
      if (failed) {
        e[i, ] <- 0.0
        xbar[i] <- 0.0
      } else {
        e[i, ] <- e[i, ] * LAMBDAw
        xbar[i] <- xbar[i] * LAMBDAv
      }
    }
    

    
    if (new_box >= 0) box <- new_box
    state <- state_new
  }
  
  list(history = history, trial_lengths = trial_lengths, global_steps = global_steps)
}

# ---- Run Full RL Simulation for Temperature Control ----
set.seed(42)
start.time <- Sys.time()
result <- simulate_temperature_control()
history <- result$history
trial_lengths <- result$trial_lengths
global_steps_final <- result$global_steps
end.time <- Sys.time()
time.taken <- end.time - start.time
cat("Time taken: ", time.taken, "\n")




# library(ggplot2); theme_set(theme_bw())
# library(patchwork)
# library(gganimate)
# library(dplyr)
# 
# #  Parameters ----
# N_BOXES <- 162       # Number of state-space boxes
# ALPHA <- 1000        # Learning rate for action weights (w)
# BETA <- 0.5          # Learning rate for critic weights (v)
# GAMMA <- 0.95        # Discount factor for critic
# LAMBDAw <- 0.9       # Decay rate for w eligibility trace
# LAMBDAv <- 0.8       # Decay rate for v eligibility trace
# MAX_FAILURES <- 99   # Maximum number of failures before termination
# MAX_STEPS <- 10000  # Maximum steps in a single trial (i.e. StepInTrial)
# 
# # Physical constants for cart-pole
# GRAVITY <- 9.8
# MASSCART <- 1.0
# MASSPOLE <- 0.1
# TOTAL_MASS <- MASSCART + MASSPOLE
# LENGTH <- 0.5                  # Half the pole's length
# POLEMASS_LENGTH <- MASSPOLE * LENGTH
# FORCE_MAG <- 10.0
# TAU <- 0.02                    # Time interval between state updates
# FOURTHIRDS <- 4/3
# 
# #  Helper functions ----
# prob_push_right <- function(s) {
#   s <- max(-50, min(s, 50))
#   1.0 / (1.0 + exp(-s))
# }
# 
# cart_pole <- function(action, state) {
#   # state: vector [x, x_dot, theta, theta_dot]
#   x <- state[1]
#   x_dot <- state[2]
#   theta <- state[3]
#   theta_dot <- state[4]
#   
#   force <- if (action > 0) FORCE_MAG else -FORCE_MAG
#   costheta <- cos(theta)
#   sintheta <- sin(theta)
#   
#   temp <- (force + POLEMASS_LENGTH * theta_dot^2 * sintheta) / TOTAL_MASS
#   thetaacc <- (GRAVITY * sintheta - costheta * temp) /
#     (LENGTH * (FOURTHIRDS - MASSPOLE * costheta^2 / TOTAL_MASS))
#   xacc <- temp - POLEMASS_LENGTH * thetaacc * costheta / TOTAL_MASS
#   
#   # Euler update
#   x <- x + TAU * x_dot
#   x_dot <- x_dot + TAU * xacc
#   theta <- theta + TAU * theta_dot
#   theta_dot <- theta_dot + TAU * thetaacc
#   
#   c(x, x_dot, theta, theta_dot)
# }
# 
# one_degree <- 0.0174532
# six_degrees <- 0.1047192
# twelve_degrees <- 0.2094384
# fifty_degrees <- 0.87266
# 
# get_box <- function(x, x_dot, theta, theta_dot) {
#   # Returns a box index (0-indexed) or -1 if failure state.
#   if (x < -2.4 || x > 2.4 || theta < -twelve_degrees || theta > twelve_degrees)
#     return(-1)
#   
#   if (x < -0.8) {
#     box <- 0
#   } else if (x < 0.8) {
#     box <- 1
#   } else {
#     box <- 2
#   }
#   
#   if (x_dot < -0.5) {
#     # no addition
#   } else if (x_dot < 0.5) {
#     box <- box + 3
#   } else {
#     box <- box + 6
#   }
#   
#   if (theta < -six_degrees) {
#     # no addition
#   } else if (theta < -one_degree) {
#     box <- box + 9
#   } else if (theta < 0) {
#     box <- box + 18
#   } else if (theta < one_degree) {
#     box <- box + 27
#   } else if (theta < six_degrees) {
#     box <- box + 36
#   } else {
#     box <- box + 45
#   }
#   
#   if (theta_dot < -fifty_degrees) {
#     # no addition
#   } else if (theta_dot < fifty_degrees) {
#     box <- box + 54
#   } else {
#     box <- box + 108
#   }
#   
#   return(box)
# }
# 
# #  Detailed simulation with a global step counter and state logging ----
# simulate_cartpole_detailed <- function() {
#   # Initialize weight and eligibility vectors
#   w <- rep(0, N_BOXES)     # Action weights
#   v <- rep(0, N_BOXES)     # Critic weights
#   e <- rep(0, N_BOXES)     # Eligibility trace for w
#   xbar <- rep(0, N_BOXES)  # Eligibility trace for v
#   
#   # Initial state: (x, x_dot, theta, theta_dot)
#   state <- c(0, 0, 0, 0)
#   box <- get_box(state[1], state[2], state[3], state[4])
#   if (box < 0) stop("Initial state in failure region.")
#   
#   # Data storage for per-step logging (record state as well)
#   history <- data.frame(
#     GlobalStep = integer(),
#     StepInTrial = integer(),
#     Box = integer(),
#     rhat = numeric(),
#     w_box = numeric(),
#     v_box = numeric(),
#     p_push = numeric(),
#     x = numeric(),
#     theta = numeric()
#   )
#   
#   trial_lengths <- c()  # Store trial lengths (in global steps)
#   global_steps <- 0     # Global step counter (never resets)
#   last_failure <- 0     # Global step count at last failure
#   
#   failures <- 0
#   # Continue simulation until either a trial reaches MAX_STEPS or MAX_FAILURES is hit.
#   while (failures < MAX_FAILURES && (global_steps - last_failure) < MAX_STEPS) {
#     
#     global_steps <- global_steps + 1
#     step_in_trial <- global_steps - last_failure
#     
#     # Debug:
#     #   print("\nvalues:")
#     #   print(failures)
#     #   print(MAX_FAILURES)
#     #   print(global_steps)
#     #   print(last_failure)
#     #   print(MAX_STEPS)
#     #   
#     # Print a message every 100 steps
#     # if (global_steps %% 100 == 0) {
#     #   cat("Global step:", global_steps, "\n")
#     # }
#     # 
#     # if (step_in_trial %% 100 == 0) {
#     #   cat("Global step:", step_in_trial, "\n")
#     # }
#     
#     # Choose action: probability from current weight
#     prob <- prob_push_right(w[box + 1])
#     y <- if (runif(1) < prob) 1 else 0
#     
#     # Update eligibility traces for current box
#     e[box + 1] <- e[box + 1] + (1 - LAMBDAw) * (y - 0.5)
#     xbar[box + 1] <- xbar[box + 1] + (1 - LAMBDAv)
#     
#     oldp <- v[box + 1]
#     
#     # Log current state BEFORE action update
#     history <- rbind(history, data.frame(
#       GlobalStep = global_steps,
#       StepInTrial = step_in_trial,
#       Box = box,
#       rhat = NA,      # to be filled after computing rhat
#       w_box = w[box + 1],
#       v_box = v[box + 1],
#       p_push = prob,
#       x = state[1],
#       theta = state[3]
#     ))
#     
#     # Apply action: update state
#     state <- cart_pole(y, state)
#     new_box <- get_box(state[1], state[2], state[3], state[4])
#     
#     if (new_box < 0) {
#       # Failure occurred: record trial length as difference
#       failures <- failures + 1
#       trial_length <- global_steps - last_failure
#       trial_lengths <- c(trial_lengths, trial_length)
#       cat(sprintf("Trial %d was %d steps.\n", failures, trial_length))
#       
#       # Set reinforcement and prediction for failure
#       r <- -1.0
#       p <- 0.0
#       failed <- TRUE
#       
#       # Update last_failure without resetting global_steps
#       last_failure <- global_steps
#       
#       # Reset state to initial and continue
#       state <- c(0, 0, 0, 0)
#       new_box <- get_box(state[1], state[2], state[3], state[4])
#     } else {
#       r <- 0.0
#       p <- v[new_box + 1]
#       failed <- FALSE
#     }
#     
#     rhat <- r + GAMMA * p - oldp
#     history$rhat[nrow(history)] <- rhat  # update logged rhat
#     
#     # Update weights and traces for all boxes
#     for (i in 1:N_BOXES) {
#       w[i] <- w[i] + ALPHA * rhat * e[i]
#       v[i] <- v[i] + BETA * rhat * xbar[i]
#       if (failed) {
#         e[i] <- 0.0
#         xbar[i] <- 0.0
#       } else {
#         e[i] <- e[i] * LAMBDAw
#         xbar[i] <- xbar[i] * LAMBDAv
#       }
#     }
#     
#     box <- new_box
#   }
#   
#   list(step_in_trial = step_in_trial, trial_lengths = trial_lengths, history = history, global_steps = global_steps)
# }
# 
# #  Run full RL simulation ----
# set.seed(123)
# start.time <- Sys.time()
# result <- simulate_cartpole_detailed()
# trial_lengths <- result$trial_lengths
# trial_lengths_count <- length(trial_lengths)
# history <- result$history
# global_steps_final <- result$global_steps
# end.time <- Sys.time()
# time.taken <- end.time - start.time
# cat("Time taken: ", time.taken, "\n")

#  Final status message based on maximum trial length ----
# last_trial <- if (length(trial_lengths) > 0) max(trial_lengths) else NA
final_trial_steps <- result$step_in_trial

if (final_trial_steps >= MAX_STEPS) {
  final_status <- sprintf("Pole balanced successfully for %d steps in trial run %d and training with a total of %d steps.",
                          final_trial_steps, length(trial_lengths), global_steps_final)
} else {
  final_status <- sprintf("Pole not balanced. Stopping after %d run failures.", length(trial_lengths))
}
cat(final_status)

#  Plot 1: Overall Trial Lengths ----
df_trials <- data.frame(
  Trial = seq_along(trial_lengths),
  Steps = trial_lengths
)
p1 <- ggplot(df_trials, aes(x = Trial, y = Steps)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  labs(subtitle = "Overall Trial Lengths\n(Steps in Each Trial)",
       x = "Trial Number", y = "Steps") +
  theme_minimal()

#  Plot 2: Evolution of Reinforcement Signal (rhat) ----
p2 <- ggplot(history, aes(x = GlobalStep, y = rhat)) +
  geom_line(color = "darkred") +
  labs(subtitle = "Evolution of Reinforcement Signal (rhat)",
       x = "Global Step", y = "rhat") +
  theme_minimal()

#  Plot 3: Evolution of Weights in the Current State Box ----
p3 <- ggplot(history, aes(x = GlobalStep)) +
  geom_point(aes(y = w_box, color = "Action Weight (w)"), size = .1) +
  geom_point(aes(y = v_box, color = "Critic Weight (v)"), size = .1) +
  labs(subtitle = "Evolution of Weights in Current State Box",
       x = "Global Step", y = "Weight Value") +
  scale_color_manual(name = "Weight Type",
                     values = c("Action Weight (w)" = "darkblue",
                                "Critic Weight (v)" = "salmon")) +
  theme_minimal()

#  Plot 4: Evolution of Action Probability (Push Right) ----
p4 <- ggplot(history, aes(x = GlobalStep, y = p_push)) +
  geom_point(color = "purple", size = .1) +
  labs(subtitle = "Evolution of Action Probability\n(Push Right)",
       x = "Global Step", y = "Probability") +
  theme_minimal()

#  Combine plots using patchwork and annotate final trial length ----
final_plot <- (p1 / p2) | (p3 / p4)
# final_plot <- final_plot + 
#   plot_annotation(title = sprintf("Figure example 3_4: Cart-Pole Balancing Learning Dynamics\nFinal Trial Length: %s steps", 
#                                   ifelse(is.na(final_trial_steps), "N/A")))

final_plot <- final_plot + 
  plot_annotation(title = "Figure example 3_4: Cart-Pole Balancing Learning Dynamics",
                  subtitle = final_status)

print(final_plot)

# Save final plot as PNG
fig_num <- "ex_3_4"
filename <- file.path(paste0("../figures/fig_", fig_num, ".png"))
ggsave(filename = filename, final_plot, width = 8, height = 6)

#  Animation using real RL simulation data ----
anim_data <- history %>% 
  mutate(cart_left = x - 0.5,
         cart_right = x + 0.5,
         cart_top = 0.2,   # fixed cart height
         cart_bottom = 0)

# Add a trial_count column to the animation data.
anim_data <- anim_data %>%
  mutate(trial_count = cumsum(c(0, diff(StepInTrial) < 0)))

p_anim <- ggplot(anim_data, aes(frame = GlobalStep)) +
  geom_rect(aes(xmin = cart_left, xmax = cart_right, ymin = cart_bottom, ymax = cart_top),
            fill = "blue", color = "black") +
  geom_segment(aes(x = x, y = cart_top,
                   xend = x + 1.5 * sin(theta),
                   yend = cart_top + 1.5 * cos(theta)),
               size = 2, color = "red") +
  coord_fixed(xlim = c(min(anim_data$x, na.rm = TRUE) - 2, max(anim_data$x, na.rm = TRUE) + 2),
              ylim = c(0, 4)) +
  labs(title = stringr::str_wrap(final_status, 60)) +
  geom_text(aes(x = 0, y= 3, 
                label = paste0(
                  "Total runs:", trial_lengths_count,
                  "\nTrial run: ", trial_count,
                  "\nTrial step: ", StepInTrial)),
            vjust = 0, color = "blue",  hjust = 0) +
  transition_time(GlobalStep)

n_unique <- length(unique(anim_data$GlobalStep))
# nframes <- round(n_unique / 100)
nframes <- 999 # e.g. google slide max is 1000 frames
anim <- animate(p_anim, nframes = nframes, renderer = gifski_renderer()) #  fps = 20,
gif_filename <- file.path(paste0("../figures/fig_", fig_num, ".gif"))
anim_save(gif_filename, anim)

# Save a static final frame ----
final_frame_data <- tail(anim_data, 1)

final_plot <- ggplot(final_frame_data, aes(x = x, y = cart_top)) +
  geom_rect(aes(xmin = cart_left, xmax = cart_right, ymin = cart_bottom, ymax = cart_top),
            fill = "blue", color = "black") +
  geom_segment(aes(xend = x + 1.5 * sin(theta), yend = cart_top + 1.5 * cos(theta)),
               size = 2, color = "red") +
  coord_fixed(xlim = c(min(final_frame_data$x, na.rm = TRUE) - 2, max(final_frame_data$x, na.rm = TRUE) + 2),
              ylim = c(0, 4)) +
  labs(title = stringr::str_wrap(final_status, 60),
       subtitle = paste("Total runs:", trial_lengths_count, "\nLast trial step:", final_frame_data$StepInTrial)) +
  theme_minimal()

final_plot <- final_plot + 
  theme(#panel.background = element_rect(fill = "white"),
        plot.background = element_rect(fill = "white")) 

# Save the final frame as a PNG file
fig_num <- "ex_3_4_gif_static"
filename <- file.path(paste0("../figures/fig_", fig_num, ".png"))
ggsave(filename, final_plot, width = 6, height = 6)
