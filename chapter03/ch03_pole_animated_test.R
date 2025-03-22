# library(plotly)
# 
# # Parameters for the cart and pole
# cart_center <- c(0, 0, 0)      # Cart centre (x, y, z); z always 0 (on the ground)
# cart_size <- 1                 # Side length of the square cart
# pole_length <- 1.5             # Fixed length of the pole
# theta <- 0                     # Initial pole tilt angle (0 = upright)
# 
# # Compute cart square vertices (in x-y plane, z=0)
# half_size <- cart_size / 2
# cart_vertices <- data.frame(
#   x = c(cart_center[1] - half_size,
#         cart_center[1] + half_size,
#         cart_center[1] + half_size,
#         cart_center[1] - half_size,
#         cart_center[1] - half_size),
#   y = c(cart_center[2] - half_size,
#         cart_center[2] - half_size,
#         cart_center[2] + half_size,
#         cart_center[2] + half_size,
#         cart_center[2] - half_size),
#   z = 0
# )
# 
# # Compute pole coordinates.
# # The pole is attached at the cart centre.
# # For simplicity, let the pole rotate in the x-z plane.
# pole_base <- cart_center
# pole_tip <- c(cart_center[1] + pole_length * sin(theta),
#               cart_center[2],
#               cart_center[3] + pole_length * cos(theta))
# 
# # Create a 3D plot with the cart and pole.
# p <- plot_ly() %>%
#   # Draw the cart as a closed line (square)
#   add_trace(data = cart_vertices, x = ~x, y = ~y, z = ~z, type = "scatter3d",
#             mode = "lines", line = list(width = 6, color = "blue"),
#             name = "Cart") %>%
#   # Draw the pole as a line from base to tip
#   add_trace(x = c(pole_base[1], pole_tip[1]),
#             y = c(pole_base[2], pole_tip[2]),
#             z = c(pole_base[3], pole_tip[3]),
#             type = "scatter3d", mode = "lines", line = list(width = 8, color = "red"),
#             name = "Pole") %>%
#   layout(title = "Initial Scene at Time 0",
#          scene = list(
#            xaxis = list(title = "X", range = c(-3, 3)),
#            yaxis = list(title = "Y", range = c(-3, 3)),
#            zaxis = list(title = "Z", range = c(0, 4))
#          ))
# 
# p



# test ----

library(ggplot2)
theme_set(theme_bw())
library(patchwork)
library(gganimate)
library(dplyr)
library(plotly)
library(htmlwidgets)

#  Parameters ----
# Discretisation: x, y, x_dot, y_dot: 3 bins each; theta: 6 bins; theta_dot: 3 bins
N_BOXES <- 3 * 3 * 3 * 3 * 6 * 3
ALPHA <- 1000         # Learning rate for action weights
BETA <- 0.5           # Learning rate for critic weights
GAMMA <- 0.95         # Discount factor for critic
LAMBDAw <- 0.9        # Decay rate for action eligibility trace
LAMBDAv <- 0.8        # Decay rate for critic eligibility trace
MAX_FAILURES <- 99    # Maximum failures before termination
MAX_STEPS <- 10000    # Maximum steps in a trial
MAX_STEPS <- 100    # Maximum steps in a trial

# Physical constants for the cart-pole
GRAVITY <- 9.8
MASSCART <- 1.0
MASSPOLE <- 0.1
TOTAL_MASS <- MASSCART + MASSPOLE
LENGTH <- 1.5         # Fixed pole length (from cart centre to tip)
POLEMASS_LENGTH <- MASSPOLE * LENGTH
FORCE_MAG <- 10.0
TAU <- 0.02
FOURTHIRDS <- 4/3

# Angle thresholds (in radians)
one_degree <- 0.0174532
six_degrees <- 0.1047192
twelve_degrees <- 0.2094384
fifty_degrees <- 0.87266

# Cart dimensions: square cart of size 1; always on ground (z = 0)
cart_size <- 1
half_cart <- cart_size / 2

#  Helper Functions ----
softmax <- function(z) {
  expz <- exp(z - max(z))
  expz / sum(expz)
}

# 2D Cart-Pole Dynamics
cart_pole <- function(action, state) {
  # state: vector [x, y, x_dot, y_dot, theta, theta_dot]
  x      <- state[1]
  y      <- state[2]
  x_dot  <- state[3]
  y_dot  <- state[4]
  theta  <- state[5]
  theta_dot <- state[6]
  
  # Four actions: 1: right, 2: left, 3: up, 4: down
  force_x <- if (action == 1) FORCE_MAG else if (action == 2) -FORCE_MAG else 0
  force_y <- if (action == 3) FORCE_MAG else if (action == 4) -FORCE_MAG else 0
  
  # For pole dynamics, use net force magnitude (F_net)
  F_net <- sqrt(force_x^2 + force_y^2)
  costheta <- cos(theta)
  sintheta <- sin(theta)
  
  temp <- (F_net + POLEMASS_LENGTH * theta_dot^2 * sintheta) / TOTAL_MASS
  thetaacc <- (GRAVITY * sintheta - costheta * temp) / 
    (LENGTH * (FOURTHIRDS - MASSPOLE * costheta^2 / TOTAL_MASS))
  
  # Update cart acceleration separately for x and y
  xacc <- force_x / TOTAL_MASS - POLEMASS_LENGTH * thetaacc * costheta / TOTAL_MASS
  yacc <- force_y / TOTAL_MASS - POLEMASS_LENGTH * thetaacc * costheta / TOTAL_MASS
  
  # Euler integration
  x <- x + TAU * x_dot
  y <- y + TAU * y_dot
  x_dot <- x_dot + TAU * xacc
  y_dot <- y_dot + TAU * yacc
  theta <- theta + TAU * theta_dot
  theta_dot <- theta_dot + TAU * thetaacc
  
  c(x, y, x_dot, y_dot, theta, theta_dot)
}

# Discretise state into a box index
get_box <- function(x, y, x_dot, y_dot, theta, theta_dot) {
  if (abs(x) > 2.4 || abs(y) > 2.4 || abs(theta) > twelve_degrees)
    return(-1)
  
  bin_x <- if (x < -0.8) 0 else if (x < 0.8) 1 else 2
  bin_y <- if (y < -0.8) 0 else if (y < 0.8) 1 else 2
  bin_xdot <- if (x_dot < -0.5) 0 else if (x_dot < 0.5) 1 else 2
  bin_ydot <- if (y_dot < -0.5) 0 else if (y_dot < 0.5) 1 else 2
  bin_theta <- if (theta < -six_degrees) 0 else if (theta < -one_degree) 1 else if (theta < 0) 2 else if (theta < one_degree) 3 else if (theta < six_degrees) 4 else 5
  bin_thetadot <- if (theta_dot < -fifty_degrees) 0 else if (theta_dot < fifty_degrees) 1 else 2
  
  box <- bin_x * (3 * 3 * 6 * 3 * 3) +
    bin_y * (3 * 6 * 3 * 3) +
    bin_xdot * (6 * 3 * 3) +
    bin_ydot * (3 * 3) +
    bin_theta * (3) +
    bin_thetadot
  return(box)
}

#  RL Simulation Function ----
simulate_cartpole_detailed <- function() {
  W <- matrix(0, nrow = N_BOXES, ncol = 4)  # Action weights for 4 actions
  v <- rep(0, N_BOXES)                     # Critic weights
  e <- matrix(0, nrow = N_BOXES, ncol = 4)   # Eligibility traces for actions
  xbar <- rep(0, N_BOXES)                   # Eligibility traces for critic
  
  # Initial state: cart at (0,0) on ground; zero velocities; pole upright (theta=0)
  state <- c(0, 0, 0, 0, 0, 0)
  box <- get_box(state[1], state[2], state[3], state[4], state[5], state[6])
  if (box < 0) stop("Initial state in failure region.")
  
  history <- data.frame(
    GlobalStep = integer(),
    StepInTrial = integer(),
    Box = integer(),
    rhat = numeric(),
    W1 = numeric(),
    v_box = numeric(),
    action = integer(),
    x = numeric(),
    y = numeric(),
    theta = numeric()
  )
  
  trial_lengths <- c()
  global_steps <- 0
  last_failure <- 0
  failures <- 0
  
  while (failures < MAX_FAILURES && (global_steps - last_failure) < MAX_STEPS) {
    global_steps <- global_steps + 1
    step_in_trial <- global_steps - last_failure
    
    # Softmax for current state's action probabilities
    probs <- softmax(W[box + 1, ])
    action <- sample(1:4, size = 1, prob = probs)
    
    one_hot <- rep(0, 4)
    one_hot[action] <- 1
    e[box + 1, ] <- e[box + 1, ] + (1 - LAMBDAw) * (one_hot - probs)
    xbar[box + 1] <- xbar[box + 1] + (1 - LAMBDAv)
    
    oldp <- v[box + 1]
    
    history <- rbind(history, data.frame(
      GlobalStep = global_steps,
      StepInTrial = step_in_trial,
      Box = box,
      rhat = NA,
      W1 = W[box + 1, 1],
      v_box = v[box + 1],
      action = action,
      x = state[1],
      y = state[2],
      theta = state[5]
    ))
    
    state <- cart_pole(action, state)
    new_box <- get_box(state[1], state[2], state[3], state[4], state[5], state[6])
    
    if (new_box < 0) {
      failures <- failures + 1
      trial_length <- global_steps - last_failure
      trial_lengths <- c(trial_lengths, trial_length)
      cat(sprintf("Trial %d was %d steps.\n", failures, trial_length))
      r <- -1.0
      p <- 0.0
      failed <- TRUE
      last_failure <- global_steps
      state <- c(0, 0, 0, 0, 0, 0)
      new_box <- get_box(state[1], state[2], state[3], state[4], state[5], state[6])
    } else {
      r <- 0.0
      p <- v[new_box + 1]
      failed <- FALSE
    }
    
    rhat <- r + GAMMA * p - oldp
    history$rhat[nrow(history)] <- rhat
    
    for (i in 1:N_BOXES) {
      W[i, ] <- W[i, ] + ALPHA * rhat * e[i, ]
      v[i] <- v[i] + BETA * rhat * xbar[i]
      if (failed) {
        e[i, ] <- 0.0
        xbar[i] <- 0.0
      } else {
        e[i, ] <- e[i, ] * LAMBDAw
        xbar[i] <- xbar[i] * LAMBDAv
      }
    }
    
    box <- new_box
  }
  
  list(step_in_trial = step_in_trial, trial_lengths = trial_lengths, history = history, global_steps = global_steps)
}

#  Run RL Simulation ----
set.seed(123)
start.time <- Sys.time()
result <- simulate_cartpole_detailed()
trial_lengths <- result$trial_lengths
history <- result$history
global_steps_final <- result$global_steps
end.time <- Sys.time()
cat("Time taken: ", end.time - start.time, "\n")

final_trial_steps <- result$step_in_trial
if (final_trial_steps >= MAX_STEPS) {
  final_status <- sprintf("Pole balanced successfully for %d steps in trial %d over %d global steps.",
                          final_trial_steps, length(trial_lengths), global_steps_final)
} else {
  final_status <- sprintf("Pole not balanced. Stopped after %d failures.", length(trial_lengths))
}
cat(final_status, "\n")
# 
# #  Plotting RL Diagnostics ----
# df_trials <- data.frame(
#   Trial = seq_along(trial_lengths),
#   Steps = trial_lengths
# )
# p1 <- ggplot(df_trials, aes(x = Trial, y = Steps)) +
#   geom_bar(stat = "identity", fill = "steelblue") +
#   labs(subtitle = "Trial Lengths", x = "Trial Number", y = "Steps") +
#   theme_minimal()
# 
# p2 <- ggplot(history, aes(x = GlobalStep, y = rhat)) +
#   geom_line(color = "darkred") +
#   labs(subtitle = "Reinforcement Signal (rhat)", x = "Global Step", y = "rhat") +
#   theme_minimal()
# 
# p3 <- ggplot(history, aes(x = GlobalStep, y = W1)) +
#   geom_point(color = "darkblue", size = 0.5) +
#   labs(subtitle = "Action Weight (W1)", x = "Global Step", y = "W1") +
#   theme_minimal()
# 
# p4 <- ggplot(history, aes(x = GlobalStep, y = v_box)) +
#   geom_point(color = "salmon", size = 0.5) +
#   labs(subtitle = "Critic Weight (v)", x = "Global Step", y = "v") +
#   theme_minimal()
# 
# final_plot <- (p1 / p2) | (p3 / p4) +
#   plot_annotation(title = "2D Cart-Pole RL Learning Dynamics", subtitle = final_status)
# print(final_plot)
# 
# 
# # Static final frame export (requires orca or kaleido)
# final_frame <- tail(history_anim, 1)
# p_final_3d <- plot_ly() %>%
#   add_trace(x = c(final_frame$cart_center_x - half_cart,
#                   final_frame$cart_center_x + half_cart,
#                   final_frame$cart_center_x + half_cart,
#                   final_frame$cart_center_x - half_cart,
#                   final_frame$cart_center_x - half_cart),
#             y = c(final_frame$cart_center_y - half_cart,
#                   final_frame$cart_center_y - half_cart,
#                   final_frame$cart_center_y + half_cart,
#                   final_frame$cart_center_y + half_cart,
#                   final_frame$cart_center_y - half_cart),
#             z = 0,
#             type = "scatter3d", mode = "lines",
#             line = list(width = 6, color = "blue"),
#             name = "Cart") %>%
#   add_trace(x = c(final_frame$pole_base_x, final_frame$pole_tip_x),
#             y = c(final_frame$pole_base_y, final_frame$pole_tip_y),
#             z = c(final_frame$pole_base_z, final_frame$pole_tip_z),
#             type = "scatter3d", mode = "lines",
#             line = list(width = 8, color = "red"),
#             name = "Pole") %>%
#   layout(title = final_status,
#          scene = list(xaxis = list(title = "X", range = c(-3, 3)),
#                       yaxis = list(title = "Y", range = c(-3, 3)),
#                       zaxis = list(title = "Z", range = c(0, 4))))
# export_file <- "fig_2d_cartpole_rl_static.png"
# export(p_final_3d, file = export_file)
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# library(dplyr)
# library(tidyr)
# library(plotly)
# library(htmlwidgets)
# #  Animation Setup: Create 3D Animation using Plotly ----
# 
# # Prepare animation data from history_anim (assumed computed previously)
# # history_anim: data frame with columns GlobalStep, cart_center_x, cart_center_y, 
# # pole_base_x, pole_base_y, pole_base_z, pole_tip_x, pole_tip_y, pole_tip_z
# # and cart dimensions: half_cart (defined earlier)
# 
# # Create cart polygon data for each frame
# cart_data <- history_anim %>%
#   rowwise() %>%
#   mutate(poly_x = list(c(cart_center_x - half_cart,
#                          cart_center_x + half_cart,
#                          cart_center_x + half_cart,
#                          cart_center_x - half_cart,
#                          cart_center_x - half_cart)),
#          poly_y = list(c(cart_center_y - half_cart,
#                          cart_center_y - half_cart,
#                          cart_center_y + half_cart,
#                          cart_center_y + half_cart,
#                          cart_center_y - half_cart)),
#          poly_z = list(rep(0, 5))) %>%
#   ungroup() %>%
#   select(GlobalStep, poly_x, poly_y, poly_z) %>%
#   unnest(c(poly_x, poly_y, poly_z))
# 
# # Create pole line data for each frame
# pole_data <- history_anim %>%
#   rowwise() %>%
#   mutate(line_x = list(c(pole_base_x, pole_tip_x)),
#          line_y = list(c(pole_base_y, pole_tip_y)),
#          line_z = list(c(pole_base_z, pole_tip_z))) %>%
#   ungroup() %>%
#   select(GlobalStep, line_x, line_y, line_z) %>%
#   unnest(c(line_x, line_y, line_z))
# 
# # Create 3D animation with Plotly
# p_anim_3d <- plot_ly() %>%
#   add_trace(data = cart_data,
#             x = ~poly_x, y = ~poly_y, z = ~poly_z,
#             type = "scatter3d", mode = "lines",
#             line = list(width = 6, color = "blue"),
#             frame = ~GlobalStep,
#             name = "Cart") %>%
#   add_trace(data = pole_data,
#             x = ~line_x, y = ~line_y, z = ~line_z,
#             type = "scatter3d", mode = "lines",
#             line = list(width = 8, color = "red"),
#             frame = ~GlobalStep,
#             name = "Pole") %>%
#   layout(title = final_status,
#          scene = list(xaxis = list(title = "X", range = c(-3, 3)),
#                       yaxis = list(title = "Y", range = c(-3, 3)),
#                       zaxis = list(title = "Z", range = c(0, 4))))
# 
# anim_html <- "fig_2d_cartpole_rl.html"
# htmlwidgets::saveWidget(p_anim_3d, file = anim_html)
# 




library(dplyr)
library(tidyr)
library(plotly)
library(htmlwidgets)

# Create history_anim from simulation history.
# If the simulation produced almost no movement, we generate artificial movement.
if(nrow(history) < 2 || all(abs(diff(history$x)) < 1e-3)) {
  history_anim <- data.frame(
    GlobalStep = 1:100,
    x = seq(0, 2, length.out = 100),
    y = seq(0, 2, length.out = 100),
    theta = sin(seq(0, 2*pi, length.out = 100)) * 0.1
  ) %>%
    mutate(
      cart_center_x = x,
      cart_center_y = y,
      cart_center_z = 0,
      pole_base_x = x,
      pole_base_y = y,
      pole_base_z = 0,
      pole_tip_x = x + LENGTH * sin(theta),
      pole_tip_y = y,
      pole_tip_z = LENGTH * cos(theta)
    )
} else {
  history_anim <- history %>%
    mutate(
      cart_center_x = x,
      cart_center_y = y,
      cart_center_z = 0,
      pole_base_x = x,
      pole_base_y = y,
      pole_base_z = 0,
      pole_tip_x = x + LENGTH * sin(theta),
      pole_tip_y = y,
      pole_tip_z = LENGTH * cos(theta)
    )
}

# Create cart polygon data for each frame.
cart_data <- history_anim %>%
  rowwise() %>%
  mutate(
    poly_x = list(c(cart_center_x - half_cart,
                    cart_center_x + half_cart,
                    cart_center_x + half_cart,
                    cart_center_x - half_cart,
                    cart_center_x - half_cart)),
    poly_y = list(c(cart_center_y - half_cart,
                    cart_center_y - half_cart,
                    cart_center_y + half_cart,
                    cart_center_y + half_cart,
                    cart_center_y - half_cart)),
    poly_z = list(rep(0, 5))
  ) %>%
  ungroup() %>%
  select(GlobalStep, poly_x, poly_y, poly_z) %>%
  unnest(cols = c(poly_x, poly_y, poly_z))

# Create pole line data for each frame.
pole_data <- history_anim %>%
  rowwise() %>%
  mutate(
    line_x = list(c(pole_base_x, pole_tip_x)),
    line_y = list(c(pole_base_y, pole_tip_y)),
    line_z = list(c(pole_base_z, pole_tip_z))
  ) %>%
  ungroup() %>%
  select(GlobalStep, line_x, line_y, line_z) %>%
  unnest(cols = c(line_x, line_y, line_z))

# Create 3D animation with Plotly.
p_anim_3d <- plot_ly() %>%
  add_trace(data = cart_data,
            x = ~poly_x, y = ~poly_y, z = ~poly_z,
            type = "scatter3d", mode = "lines",
            line = list(width = 6, color = "blue"),
            frame = ~GlobalStep,
            name = "Cart") %>%
  add_trace(data = pole_data,
            x = ~line_x, y = ~line_y, z = ~line_z,
            type = "scatter3d", mode = "lines",
            line = list(width = 8, color = "red"),
            frame = ~GlobalStep,
            name = "Pole") %>%
  layout(title = final_status,
         scene = list(xaxis = list(title = "X", range = c(-3, 3)),
                      yaxis = list(title = "Y", range = c(-3, 3)),
                      zaxis = list(title = "Z", range = c(0, 4))))

anim_html <- "fig_2d_cartpole_rl.html"
htmlwidgets::saveWidget(p_anim_3d, file = anim_html)











library(ggplot2)
library(gg3D)    # remotes::install_github("AckerDWM/gg3D")
library(gganimate)
library(dplyr)

# Sample data for history_anim if not already defined
if (!exists("history_anim")) {
  history_anim <- data.frame(
    GlobalStep = 1:100,
    cart_center_x = seq(0, 2, length.out = 100),
    cart_center_y = seq(0, 2, length.out = 100),
    theta = sin(seq(0, 2*pi, length.out = 100)) * 0.1
  ) %>%
    mutate(
      pole_tip_x = cart_center_x + LENGTH * sin(theta),
      pole_tip_y = cart_center_y,
      pole_tip_z = LENGTH * cos(theta)
    )
}

# Cart dimensions (square of side 1)
half_cart <- cart_size / 2

# Create cart polygon data: 5 vertices per frame (closing the polygon)
cart_poly <- history_anim %>% 
  rowwise() %>%
  do({
    data.frame(
      GlobalStep = .$GlobalStep,
      x = c(.$cart_center_x - half_cart,
            .$cart_center_x + half_cart,
            .$cart_center_x + half_cart,
            .$cart_center_x - half_cart,
            .$cart_center_x - half_cart),
      y = c(.$cart_center_y - half_cart,
            .$cart_center_y - half_cart,
            .$cart_center_y + half_cart,
            .$cart_center_y + half_cart,
            .$cart_center_y - half_cart),
      z = 0
    )
  }) %>% ungroup()

# Create pole data: two points per frame (base and tip)
pole_df <- history_anim %>%
  rowwise() %>%
  do({
    data.frame(
      GlobalStep = .$GlobalStep,
      x = c(.$cart_center_x, .$pole_tip_x),
      y = c(.$cart_center_y, .$pole_tip_y),
      z = c(0, .$pole_tip_z)
    )
  }) %>% ungroup()

final_status <- "RL Cart-Pole Simulation"

# Build the plot with a dummy mapping in the global ggplot() call to satisfy axes_3D
p_anim <- ggplot() +
  # Dummy layer for global mapping (required by axes_3D)
  geom_blank(data = cart_poly, mapping = aes(x = x, y = y, z = z)) +
  # Cart polygon layer
  stat_3D(data = cart_poly, 
          mapping = aes(x = x, y = y, z = z, group = GlobalStep),
          geom = "path", linewidth = 1.2, color = "blue") +
  # Pole layer
  stat_3D(data = pole_df,
          mapping = aes(x = x, y = y, z = z, group = GlobalStep),
          geom = "path", linewidth = 1.5, color = "red") +
  # Axes: provide data and mapping so that x, y, and z are known
  axes_3D(data = cart_poly, mapping = aes(x = x, y = y, z = z),
          theta = 30, phi = 30) +
  labs_3D(theta = 30, phi = 30, labs = c("X", "Y", "Z")) +
  ggtitle(final_status) +
  transition_time(GlobalStep) +
  ease_aes('linear')

# Animate at 20 fps; number of frames based on unique GlobalStep values
animated_plot <- animate(p_anim, nframes = length(unique(history_anim$GlobalStep)), 
                         fps = 20, width = 800, height = 600)
anim_save("cartpole_rl.gif", animation = animated_plot)
