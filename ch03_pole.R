# Figure 3_2: Cart-Pole Simulation and Learning
# This R code reimplements the cart-pole simulation and learning algorithm.
# It simulates the system until either MAX_STEPS is reached or MAX_FAILURES occur.
# The simulation uses a reinforcement learning algorithm to update action (w) and critic (v) weights.
#
#
# This code implements a simulation of a cart‐pole system combined with a reinforcement learning algorithm to learn how to balance the pole. The simulation follows these steps:
#   Initialisation:
#   The system state is defined by four variables (cart position and velocity, pole angle and angular velocity), all initialised to zero. The continuous state space is discretised into 162 boxes using predefined thresholds, so each state falls into a specific box.
# Action Selection:
#   For the current state (box), an action is selected probabilistically. The probability of pushing the cart to the right is computed using a logistic function applied to the current action weight associated with that box.
# Dynamics Update:
#   The chosen action is applied to the cart-pole system. The system’s state is updated using Euler’s method, based on the physical dynamics of the cart and pole.
# Failure Check and Reinforcement:
#   If the new state falls outside allowable limits, a failure is registered (reinforcement of -1), the trial ends, and the system is reset to the initial state. Otherwise, the reinforcement is zero and the critic’s prediction is used.
# Learning Updates:
#   A heuristic reinforcement signal is computed and used to update both the action weights (w) and critic weights (v) with eligibility traces, which decay over time. These updates aim to adjust the weights so that the system gradually learns to avoid failure.
# Iteration and Output:
#   The simulation runs for many steps or until a specified number of failures occur. The number of steps achieved before each failure (trial length) is recorded and then plotted to illustrate the learning performance.

library(ggplot2); theme_set(theme_bw())
library(patchwork)

# --- Parameters ---
N_BOXES <- 162       # Number of state-space boxes
ALPHA <- 1000        # Learning rate for action weights (w)
BETA <- 0.5          # Learning rate for critic weights (v)
GAMMA <- 0.95        # Discount factor for critic
LAMBDAw <- 0.9       # Decay rate for w eligibility trace
LAMBDAv <- 0.8       # Decay rate for v eligibility trace
MAX_FAILURES <- 100  # Maximum number of failures before termination
MAX_STEPS <- 100000    # Maximum steps per trial (for demonstration)

# Physical constants for cart-pole
GRAVITY <- 9.8
MASSCART <- 1.0
MASSPOLE <- 0.1
TOTAL_MASS <- MASSCART + MASSPOLE
LENGTH <- 0.5                  # Half the pole's length
POLEMASS_LENGTH <- MASSPOLE * LENGTH
FORCE_MAG <- 10.0
TAU <- 0.02                    # Time interval between state updates
FOURTHIRDS <- 4/3

# --- Helper functions ---
prob_push_right <- function(s) {
  s <- max(-50, min(s, 50))
  1.0 / (1.0 + exp(-s))
}

cart_pole <- function(action, state) {
  # state: vector [x, x_dot, theta, theta_dot]
  x <- state[1]
  x_dot <- state[2]
  theta <- state[3]
  theta_dot <- state[4]
  
  force <- if (action > 0) FORCE_MAG else -FORCE_MAG
  costheta <- cos(theta)
  sintheta <- sin(theta)
  
  temp <- (force + POLEMASS_LENGTH * theta_dot^2 * sintheta) / TOTAL_MASS
  thetaacc <- (GRAVITY * sintheta - costheta * temp) /
    (LENGTH * (FOURTHIRDS - MASSPOLE * costheta^2 / TOTAL_MASS))
  xacc <- temp - POLEMASS_LENGTH * thetaacc * costheta / TOTAL_MASS
  
  # Euler update
  x <- x + TAU * x_dot
  x_dot <- x_dot + TAU * xacc
  theta <- theta + TAU * theta_dot
  theta_dot <- theta_dot + TAU * thetaacc
  
  c(x, x_dot, theta, theta_dot)
}

one_degree <- 0.0174532
six_degrees <- 0.1047192
twelve_degrees <- 0.2094384
fifty_degrees <- 0.87266

get_box <- function(x, x_dot, theta, theta_dot) {
  # Returns a box index (0-indexed) or -1 if failure state.
  if (x < -2.4 || x > 2.4 || theta < -twelve_degrees || theta > twelve_degrees)
    return(-1)
  
  if (x < -0.8) {
    box <- 0
  } else if (x < 0.8) {
    box <- 1
  } else {
    box <- 2
  }
  
  if (x_dot < -0.5) {
    # no addition
  } else if (x_dot < 0.5) {
    box <- box + 3
  } else {
    box <- box + 6
  }
  
  if (theta < -six_degrees) {
    # no addition
  } else if (theta < -one_degree) {
    box <- box + 9
  } else if (theta < 0) {
    box <- box + 18
  } else if (theta < one_degree) {
    box <- box + 27
  } else if (theta < six_degrees) {
    box <- box + 36
  } else {
    box <- box + 45
  }
  
  if (theta_dot < -fifty_degrees) {
    # no addition
  } else if (theta_dot < fifty_degrees) {
    box <- box + 54
  } else {
    box <- box + 108
  }
  
  return(box)
}

# --- Detailed simulation with a global step counter ---
simulate_cartpole_detailed <- function() {
  # Initialize weight and eligibility vectors
  w <- rep(0, N_BOXES)     # Action weights
  v <- rep(0, N_BOXES)     # Critic weights
  e <- rep(0, N_BOXES)     # Eligibility trace for w
  xbar <- rep(0, N_BOXES)  # Eligibility trace for v
  
  # Initial state: (x, x_dot, theta, theta_dot)
  state <- c(0, 0, 0, 0)
  box <- get_box(state[1], state[2], state[3], state[4])
  if (box < 0) stop("Initial state in failure region.")
  
  # Data storage for per-step logging (for one trial)
  history <- data.frame(
    GlobalStep = integer(),
    StepInTrial = integer(),
    Box = integer(),
    rhat = numeric(),
    w_box = numeric(),
    v_box = numeric(),
    p_push = numeric()
  )
  
  trial_lengths <- c()  # Store trial lengths (in global steps)
  global_steps <- 0     # Global step counter (never resets)
  last_failure <- 0     # Global step count at last failure
  
  failures <- 0
  # Main simulation loop until MAX_FAILURES reached
  while (failures < MAX_FAILURES && global_steps < MAX_STEPS) {
    global_steps <- global_steps + 1
    step_in_trial <- global_steps - last_failure
    
    # Choose action: probability from current weight
    prob <- prob_push_right(w[box + 1])
    y <- if (runif(1) < prob) 1 else 0
    
    # Update eligibility traces for current box
    e[box + 1] <- e[box + 1] + (1 - LAMBDAw) * (y - 0.5)
    xbar[box + 1] <- xbar[box + 1] + (1 - LAMBDAv)
    
    oldp <- v[box + 1]
    
    # Log current state before update
    history <- rbind(history, data.frame(
      GlobalStep = global_steps,
      StepInTrial = step_in_trial,
      Box = box,
      rhat = NA,      # will be filled after computing rhat
      w_box = w[box + 1],
      v_box = v[box + 1],
      p_push = prob
    ))
    
    # Apply action: update state
    state <- cart_pole(y, state)
    new_box <- get_box(state[1], state[2], state[3], state[4])
    
    if (new_box < 0) {
      # Failure occurred: record trial length as difference
      failures <- failures + 1
      trial_length <- global_steps - last_failure
      trial_lengths <- c(trial_lengths, trial_length)
      cat(sprintf("Trial %d was %d steps.\n", failures, trial_length))
      
      # Set reinforcement and prediction for failure
      r <- -1.0
      p <- 0.0
      failed <- TRUE
      
      # Do NOT reset global_steps; instead update last_failure - I forgot this in earlier version
      last_failure <- global_steps
      
      # Reset state to initial and continue
      state <- c(0, 0, 0, 0)
      new_box <- get_box(state[1], state[2], state[3], state[4])
    } else {
      r <- 0.0
      p <- v[new_box + 1]
      failed <- FALSE
    }
    
    rhat <- r + GAMMA * p - oldp
    history$rhat[nrow(history)] <- rhat  # update logged rhat
    
    # Update weights and traces for all boxes
    for (i in 1:N_BOXES) {
      w[i] <- w[i] + ALPHA * rhat * e[i]
      v[i] <- v[i] + BETA * rhat * xbar[i]
      if (failed) {
        e[i] <- 0.0
        xbar[i] <- 0.0
      } else {
        e[i] <- e[i] * LAMBDAw
        xbar[i] <- xbar[i] * LAMBDAv
      }
    }
    
    box <- new_box
  }
  
  if (failures >= MAX_FAILURES) {
    cat(sprintf("Pole not balanced. Stopping after %d failures.\n", failures))
  } else {
    cat(sprintf("Global steps reached: %d\n", global_steps))
  }
  
  list(trial_lengths = trial_lengths, history = history)
}

# --- Run simulation ---
set.seed(123)
start.time <- Sys.time()
result <- simulate_cartpole_detailed()
trial_lengths <- result$trial_lengths
history <- result$history
end.time <- Sys.time()
time.taken <- end.time - start.time
cat("Time taken: ", time.taken, "\n")

# --- Plot 1: Overall Trial Lengths ---
df_trials <- data.frame(
  Trial = seq_along(trial_lengths),
  Steps = trial_lengths
)
p1 <- ggplot(df_trials, aes(x = Trial, y = Steps)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  labs(subtitle = "Overall Trial Lengths\n(Global Steps Between Failures)",
       x = "Trial Number", y = "Steps") +
  theme_minimal()

# --- Plot 2: Evolution of Reinforcement Signal (rhat) ---
p2 <- ggplot(history, aes(x = GlobalStep, y = rhat)) +
  geom_line(color = "darkred") +
  labs(subtitle = "Evolution of Reinforcement Signal (rhat)",
       x = "Global Step", y = "rhat") +
  theme_minimal()

# --- Plot 3: Evolution of Weights in the Current State Box ---
p3 <- ggplot(history, aes(x = GlobalStep)) +
  geom_point(aes(y = w_box, color = "Action Weight (w)"), size = .1) +
  geom_point(aes(y = v_box, color = "Critic Weight (v)"), size = .1) +
  labs(subtitle = "Evolution of Weights in Current State Box",
       x = "Global Step", y = "Weight Value") +
  scale_color_manual(name = "Weight Type",
                     values = c("Action Weight (w)" = "darkblue",
                                "Critic Weight (v)" = "salmon")) +
  theme_minimal()

# --- Plot 4: Evolution of Action Probability (Push Right) ---
p4 <- ggplot(history, aes(x = GlobalStep, y = p_push)) +
  geom_point(color = "purple", size = .1) +
  labs(subtitle = "Evolution of Action Probability\n(Push Right)",
       x = "Global Step", y = "Probability") +
  theme_minimal()

# --- Combine plots using patchwork ---
final_plot <- (p1 / p2) | (p3 / p4)
final_plot <- final_plot + plot_annotation(title = "Example 3_4: Cart-Pole Balancing Learning Dynamics")
print(final_plot)

fig_num <- "ex_3_4"
filename <- file.path(paste0("figures/fig_", fig_num, ".png"))
ggsave(filename = filename, final_plot)


# Result with MAX_STEPS <- 100000 ----
# 
# Trial 1 was 21 steps.
# Trial 2 was 21 steps.
# Trial 3 was 9 steps.
# Trial 4 was 78 steps.
# Trial 5 was 10 steps.
# Trial 6 was 22 steps.
# Trial 7 was 21 steps.
# Trial 8 was 65 steps.
# Trial 9 was 48 steps.
# Trial 10 was 10 steps.
# Trial 11 was 99 steps.
# Trial 12 was 42 steps.
# Trial 13 was 21 steps.
# Trial 14 was 49 steps.
# Trial 15 was 90 steps.
# Trial 16 was 75 steps.
# Trial 17 was 26 steps.
# Trial 18 was 43 steps.
# Trial 19 was 138 steps.
# Trial 20 was 124 steps.
# Trial 21 was 58 steps.
# Trial 22 was 127 steps.
# Trial 23 was 136 steps.
# Trial 24 was 146 steps.
# Trial 25 was 137 steps.
# Trial 26 was 152 steps.
# Trial 27 was 136 steps.
# Trial 28 was 148 steps.
# Trial 29 was 106 steps.
# Trial 30 was 253 steps.
# Trial 31 was 178 steps.
# Trial 32 was 172 steps.
# Trial 33 was 171 steps.
# Trial 34 was 207 steps.
# Trial 35 was 170 steps.
# Trial 36 was 357 steps.
# Trial 37 was 238 steps.
# Trial 38 was 198 steps.
# Trial 39 was 175 steps.
# Trial 40 was 173 steps.
# Trial 41 was 172 steps.
# Trial 42 was 156 steps.
# Trial 43 was 260 steps.
# Trial 44 was 210 steps.
# Trial 45 was 253 steps.
# Trial 46 was 269 steps.
# Trial 47 was 351 steps.
# Trial 48 was 186 steps.
# Trial 49 was 500 steps.
# Trial 50 was 1786 steps.
# Trial 51 was 652 steps.
# Trial 52 was 1765 steps.
# Trial 53 was 1765 steps.
# Trial 54 was 1765 steps.
# Trial 55 was 439 steps.
# Trial 56 was 303 steps.
# Trial 57 was 1765 steps.
# Trial 58 was 303 steps.
# Trial 59 was 1765 steps.
# Global steps reached: 100000

# Time taken:  19.3311 (minutes)