library(ggplot2); theme_set(theme_bw())
library(dplyr)
library(patchwork)

rows <- 5
cols <- 5
gamma <- 0.9
threshold <- 1e-6
actions <- list(
  up = c(0, 1),
  right = c(1, 0),
  down = c(0, -1),
  left = c(-1, 0)
)

# Special states (0-indexed)
A <- c(1, 4)
Aprime <- c(1, 0)
B <- c(3, 4)
Bprime <- c(3, 2)

special_transition <- function(state) {
  if (all(state == A)) {
    return(list(TRUE, 10, Aprime))
  }
  if (all(state == B)) {
    return(list(TRUE, 5, Bprime))
  }
  list(FALSE, NA, NA)
}

full_backup <- function(state, action, V_mat) {
  sp <- special_transition(state)
  if (sp[[1]]) {
    ns <- sp[[3]]
    r <- sp[[2]]
    return(r + gamma * V_mat[ns[2] + 1, ns[1] + 1])
  }
  
  ns <- state + action
  if (ns[1] < 0 || ns[1] >= cols || ns[2] < 0 || ns[2] >= rows) {
    ns <- state
    r <- -1
  } else {
    r <- 0
  }
  V_next <- V_mat[ns[2] + 1, ns[1] + 1]
  r + gamma * V_next
}

compute_V <- function() {
  V <- matrix(0, nrow = rows, ncol = cols)
  repeat {
    delta <- 0
    V_new <- V
    for (y in 0:(rows - 1)) {
      for (x in 0:(cols - 1)) {
        state <- c(x, y)
        backups <- sapply(actions, function(act) full_backup(state, act, V))
        old_val <- V[y + 1, x + 1]
        new_val <- mean(backups)
        V_new[y + 1, x + 1] <- new_val
        delta <- delta + abs(new_val - old_val)
      }
    }
    V <- V_new
    if (delta < threshold) break
  }
  V
}

compute_V_star <- function() {
  V <- matrix(0, nrow = rows, ncol = cols)
  policy <- vector("list", rows * cols)
  repeat {
    delta <- 0
    V_new <- V
    for (y in 0:(rows - 1)) {
      for (x in 0:(cols - 1)) {
        state <- c(x, y)
        q_vals <- sapply(actions, function(act) full_backup(state, act, V))
        old_val <- V[y + 1, x + 1]
        new_val <- max(q_vals)
        V_new[y + 1, x + 1] <- new_val
        delta <- delta + abs(new_val - old_val)
        best_actions <- names(q_vals)[abs(q_vals - new_val) < 1e-6]
        policy[[state_to_index(state)]] <- best_actions
      }
    }
    V <- V_new
    if (delta < threshold) break
  }
  list(V = V, policy = policy)
}

state_to_index <- function(state) {
  state[2] * cols + state[1] + 1
}

prepare_plot_df <- function(V) {
  df <- expand.grid(x = 0:(cols - 1), y = 0:(rows - 1))
  df$value <- as.vector(t(V))
  df
}

plot_value_function <- function(V) {
  df <- prepare_plot_df(V)
  ggplot(df, aes(x = x, y = y, fill = value)) +
    geom_tile() +
    geom_text(aes(label = round(value, 1)), size = 4) +
    scale_fill_gradient(low = "white", high = "steelblue") +
    labs(title = "Optimal state-value function (V*)", x = "x", y = "y") +
    scale_x_continuous(breaks = 0:(cols - 1)) +
    scale_y_continuous(breaks = 0:(rows - 1)) +
    coord_fixed()
}

plot_optimal_policy <- function(V, policy) {
  df <- prepare_plot_df(V)
  
  arrow_df <- expand.grid(x = 0:(cols - 1), y = 0:(rows - 1))
  arrow_df$action <- ""
  
  action_symbols <- c(up = "\u2191", right = "\u2192", down = "\u2193", left = "\u2190")
  
  for (y in 0:(rows - 1)) {
    for (x in 0:(cols - 1)) {
      idx <- state_to_index(c(x, y))
      acts <- policy[[idx]]
      if (length(acts) > 0) {
        arrow_df$action[arrow_df$x == x & arrow_df$y == y] <- paste(action_symbols[acts], collapse = ",")
      }
    }
  }
  
  ggplot(df, aes(x = x, y = y, fill = value)) +
    geom_tile() +
    geom_text(data = arrow_df, aes(x = x, y = y, label = action),
              size = 3, vjust = 0.5, inherit.aes = FALSE) +
    scale_fill_gradient(low = "white", high = "steelblue") +
    labs(title = "Optimal state-value function\nand policy (π*)", x = "x", y = "y") +
    scale_x_continuous(breaks = 0:(cols - 1)) +
    scale_y_continuous(breaks = 0:(rows - 1)) +
    coord_fixed()
}

plot_exceptional_dynamics <- function() {
  df <- expand.grid(x = 0:(cols - 1), y = 0:(rows - 1))
  p <- ggplot(df, aes(x = x, y = y)) +
    geom_tile(color = "grey90", fill = "white", size = 0.5) +
    scale_x_continuous(breaks = 0:(cols - 1)) +
    scale_y_continuous(breaks = 0:(rows - 1)) +
    coord_fixed() +
    labs(title = "Exceptional reward dynamics", x = "x", y = "y")
  
  transitions <- data.frame(
    x = c(A[1], B[1]),
    y = c(A[2], B[2]),
    xend = c(Aprime[1], Bprime[1]),
    yend = c(Aprime[2], Bprime[2]),
    reward = c(10, 5)
  )
  
  p <- p + geom_curve(data = transitions,
                      aes(x = x, y = y, xend = xend, yend = yend),
                      curvature = -0.3,
                      arrow = arrow(length = unit(0.2, "cm")),
                      colour = "red", size = 1)
  
  transitions$mid_x <- (transitions$x + transitions$xend) / 2
  transitions$mid_y <- (transitions$y + transitions$yend) / 2
  
  p + geom_text(data = transitions,
                aes(x = mid_x, y = mid_y, label = paste0("+", reward)),
                colour = "red", size = 5)
}

# Compute value functions and policy
V_uniform <- compute_V()
optimal <- compute_V_star()
V_optimal <- optimal$V
policy <- optimal$policy

# fig 3_2 ----
# Plot from plot_exceptional_dynamics()
p_ex <- plot_exceptional_dynamics()
print(p_ex)

# Plot from plot_value_function() with uniform policy values
p_val <- plot_value_function(V_uniform)
print(p_val)

combined <- p_ex + p_val + plot_annotation(
  title = "Figure 3.2: Exceptional reward dynamics and \nstate-value function fo the equiprobable (uniform) random policy")

print(combined)

fig_num <- "3_2"
filename <- file.path(paste0("../figures/fig_", fig_num, ".png"))
ggsave(filename = filename, plot = combined, height = 4, width = 7)

# fig 3_5 ----
# p_ex: Exceptional reward dynamics plot
p_ex <- plot_exceptional_dynamics()

# p_Vstar: Optimal state-value function plot (V*) with printed values
p_Vstar <- plot_value_function(V_optimal)

# p_pi: Optimal policy plot (π*) with arrows inside cells
p_pi <- plot_optimal_policy(V_optimal, policy)

combined_fig2 <- (p_ex + p_Vstar + p_pi) + plot_annotation(
  title = "Figure 3.5: Optimal solutions for the gridworld example")

print(combined_fig2)

fig_num <- "3_5"
filename <- file.path(paste0("../figures/fig_", fig_num, ".png"))
ggsave(filename = filename, plot = combined_fig2, height = 4, width = 10)

# The gridworld example represents a simple Markov Decision Process (MDP) with a 5×5 grid where each cell is a state. The environment is defined by standard grid dynamics, with a discount factor (γ = 0.9) and a termination criterion based on a small change threshold (1e–6). The agent can move in one of four directions (up, right, down, left), but there are two special states (A and B) that trigger exceptional transitions:
#   
# **Special Transitions:**  
# 
# * When in state A (cell at (1,4)), any action leads the agent to state A′ (cell at (1,0)) with a reward of +10.  
# * When in state B (cell at (3,4)), any action leads the agent to state B′ (cell at (3,2)) with a reward of +5.
# 
# For other states, moving off the grid results in staying in the same state with a penalty of –1, while valid moves provide zero immediate reward.
# 
# Two main value functions are computed:
# - **Uniform Policy Value Function (V):**  
#   This is computed by averaging the backups (expected returns) over all actions, simulating a uniform random policy.
# - **Optimal Value Function (V\*):**  
#   Here, the Bellman optimality equation is used (taking the maximum over actions) to iteratively compute the optimal state-value function. Alongside V\*, the optimal policy (π\*) is derived, indicating for each state the action(s) that maximise the expected return.
# 
# The resulting figures illustrate these concepts:
#   
#   - **Figure 3.2 (Combined Uniform Policy and Exceptional Dynamics):**  
#   - *Top Panel – Exceptional Reward Dynamics:*  
#   This plot shows the grid with curved red arrows that indicate the special transitions from state A to A′ and from state B to B′. The arrows are annotated with the respective rewards (+10 and +5), highlighting the exceptional dynamics of the MDP.
# - *Bottom Panel – Uniform State-Value Function:*  
#   Here, the state-value function computed under a uniform random policy is visualised. Each cell displays its value (printed as text) and is colour-coded, providing a clear picture of how the environment’s dynamics (including the special transitions) propagate values throughout the grid.
# 
# - **Figure 3.5 (Combined Optimal Value and Policy):**  
#   This composite figure consists of three subplots:
#   1. *Exceptional Reward Dynamics:* (Same as in Figure 3.2)  
# Reiterates the exceptional transitions to contextualise the optimal calculations.
# 2. *Optimal State-Value Function (V\*):*  
#   The computed optimal values are displayed in each cell with both numerical annotations and a colour gradient, illustrating the effect of choosing the best possible actions.
# 3. *Optimal Policy (π\*):*  
#   This plot overlays directional arrows inside each grid cell, representing the optimal action(s) derived from V\*. The arrows correspond to the directions (up, right, down, left) that yield the highest expected return. In some cells, multiple arrows appear if several actions are equally optimal.
# 
# Together, these figures provide a comprehensive visualisation of the theory behind the gridworld MDP. They demonstrate how exceptional rewards influence value propagation, how a uniform policy evaluates state values, and how optimal decision-making is represented both in value terms and as explicit action choices within the grid.
