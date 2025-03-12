library(R6)
library(ggplot2); theme_set(theme_bw())
library(patchwork)

# BaseBandit class ----
BaseBandit <- R6Class("BaseBandit",
  public = list(
    k_arm = NULL,
    eps = NULL,
    initial_q = NULL,
    true_q_mean = NULL,
    possible_actions = NULL,
    q_true = NULL,
    q_estimate = NULL,
    action_count = NULL,
    optimal_action_freq = NULL,
    
    initialize = function(k_arm = 10, eps = 0, initial_q = 0, true_q_mean = 0) {
      self$k_arm <- k_arm
      self$eps <- eps
      self$initial_q <- initial_q
      self$true_q_mean <- true_q_mean
      self$possible_actions <- 1:k_arm
      self$reset()
    },
    
    reset = function() {
      self$q_true <- rnorm(self$k_arm, mean = self$true_q_mean, sd = 1)
      self$q_estimate <- rep(self$initial_q, self$k_arm)
      self$action_count <- rep(0, self$k_arm)
      self$optimal_action_freq <- 0
    },
    
    act = function() {
      if (runif(1) < self$eps) {
        sample(self$possible_actions, 1)
      } else {
        which.max(self$q_estimate)
      }
    },
    
    reward = function(action) {
      rnorm(1, mean = self$q_true[action], sd = 1)
    },
    
    update_q = function(action, reward) {
      self$q_estimate[action] <- self$q_estimate[action] +
        1 / self$action_count[action] * (reward - self$q_estimate[action])
    },
    
    step = function() {
      action <- self$act()
      r <- self$reward(action)
      self$action_count[action] <- self$action_count[action] + 1
      self$update_q(action, r)
      if (action == which.max(self$q_true)) {
        self$optimal_action_freq <- self$optimal_action_freq +
          1 / sum(self$action_count) * (1 - self$optimal_action_freq)
      }
      list(action = action, reward = r)
    }
  )
)

# ExponentialAverageBandit class ----
ExponentialAverageBandit <- R6Class("ExponentialAverageBandit",
  inherit = BaseBandit,
  public = list(
    step_size = NULL,
    initialize = function(k_arm = 10, eps = 0, initial_q = 0, true_q_mean = 0, step_size = 0.1) {
      super$initialize(k_arm = k_arm, eps = eps, initial_q = initial_q, true_q_mean = true_q_mean)
      self$step_size <- step_size
    },
    update_q = function(action, reward) {
      self$q_estimate[action] <- self$q_estimate[action] + self$step_size * (reward - self$q_estimate[action])
    }
  )
)

# UCBBandit class ----
UCBBandit <- R6Class("UCBBandit",
  inherit = BaseBandit,
  public = list(
    c = NULL,
    initialize = function(k_arm = 10, eps = 0, initial_q = 0, true_q_mean = 0, c = 2) {
      super$initialize(k_arm = k_arm, eps = eps, initial_q = initial_q, true_q_mean = true_q_mean)
      self$c <- c
    },
    act = function() {
      if (runif(1) < self$eps) {
        sample(self$possible_actions, 1)
      } else {
        t <- sum(self$action_count) + 1
        # adding a small value to avoid division by zero
        q <- self$q_estimate + self$c * sqrt(log(t) / (self$action_count + 1e-6))
        which.max(q)
      }
    }
  )
)

# GradientBandit class ----
GradientBandit <- R6Class("GradientBandit",
  inherit = BaseBandit,
  public = list(
    baseline = NULL,
    step_size = NULL,
    average_reward = NULL,
    softmax = NULL,
    initialize = function(k_arm = 10, eps = 0, initial_q = 0, true_q_mean = 0,
                          baseline = TRUE, step_size = 0.1) {
      super$initialize(k_arm = k_arm, eps = eps, initial_q = initial_q, true_q_mean = true_q_mean)
      self$baseline <- baseline
      self$step_size <- step_size
      self$average_reward <- 0
    },
    act = function() {
      exp_est <- exp(self$q_estimate)
      self$softmax <- exp_est / sum(exp_est)
      sample(self$possible_actions, 1, prob = self$softmax)
    },
    update_q = function(action, reward) {
      self$average_reward <- self$average_reward +
        1 / sum(self$action_count) * (reward - self$average_reward)
      baseline_val <- if (self$baseline) self$average_reward else 0
      mask <- rep(0, self$k_arm)
      mask[action] <- 1
      self$q_estimate <- self$q_estimate + self$step_size * (reward - baseline_val) * (mask - self$softmax)
    }
  )
)

# Run bandits simulation ----
run_bandits <- function(bandits, n_runs, n_steps) {
  n_bandits <- length(bandits)
  rewards_array <- array(0, dim = c(n_bandits, n_runs, n_steps))
  optimal_action_freqs <- array(0, dim = c(n_bandits, n_runs, n_steps))
  
  for (b in seq_len(n_bandits)) {
    for (run in seq_len(n_runs)) {
      bandits[[b]]$reset()
      for (step in seq_len(n_steps)) {
        res <- bandits[[b]]$step()
        rewards_array[b, run, step] <- res$reward
        if (res$action == which.max(bandits[[b]]$q_true)) {
          optimal_action_freqs[b, run, step] <- 1
        }
      }
    }
  }
  
  avg_rewards <- apply(rewards_array, c(1, 3), mean)
  avg_optimal_action_freqs <- apply(optimal_action_freqs, c(1, 3), mean)
  
  list(avg_rewards = avg_rewards, avg_optimal_action_freqs = avg_optimal_action_freqs)
}

# Figure 2.1: Violin plot of reward distributions ----
fig_2_1 <- function() {
  set.seed(666)
  data_mat <- matrix(rnorm(200 * 10), ncol = 10)
  shifts <- rnorm(10)
  for (i in 1:10) {
    data_mat[, i] <- data_mat[, i] + shifts[i]
  }
  df <- data.frame(
    value = as.vector(data_mat),
    action = factor(rep(1:10, each = 200))
  )
  
  p <- ggplot(df, aes(x = action, y = value)) +
    geom_violin(trim = FALSE) +
    geom_jitter(alpha= 0.1) +
    stat_summary(fun = mean, geom = "point", color = "red") +
    geom_hline(yintercept = 0, linetype="dotted") +
    labs(x = "Action", y = "Reward distribution",
         title = "Figure 2.1: Violin Plot of Reward Distributions")
  
  print(p)
  fig_num <- "2_1"
  filename <- file.path(paste0("figures/fig_", fig_num, ".png"))
  ggsave(filename = filename)
}

fig_2_1()

# Figure 2.2: Average performance of ε-greedy methods ----
# runs=2000, steps=1000
fig_2_2 <- function(runs = 2000, steps = 1000, epsilons = c(0, 0.01, 0.1)) {
  bandits <- lapply(epsilons, function(eps) BaseBandit$new(eps = eps))
  res <- run_bandits(bandits, runs, steps)
  avg_rewards <- res$avg_rewards  # rows: bandits, columns: steps
  avg_optimal <- res$avg_optimal_action_freqs
  
  df_rewards <- data.frame(
    step = rep(1:steps, times = length(epsilons)),
    avg_reward = as.vector(t(avg_rewards)),
    epsilon = factor(rep(epsilons, each = steps))
  )
  
  p1 <- ggplot(df_rewards, aes(x = step, y = avg_reward, colour = epsilon)) +
    geom_line() +
    labs(title = "Figure 2.2 (Top): Average Reward", x = "Steps", y = "Average Reward")
  
  df_optimal <- data.frame(
    step = rep(1:steps, times = length(epsilons)),
    optimal = as.vector(t(avg_optimal)),
    epsilon = factor(rep(epsilons, each = steps))
  )
  
  p2 <- ggplot(df_optimal, aes(x = step, y = optimal, colour = epsilon)) +
    geom_line() +
    labs(title = "Figure 2.2 (Bottom): % Optimal Action", x = "Steps", y = "% Optimal Action")
  
  print(p1 / p2)
  # print(p2)
  
  fig_num <- "2_2"
  filename <- file.path(paste0("figures/fig_", fig_num, ".png"))
  ggsave(filename = filename, p2)

}

# Run the figures (they will be printed, not saved)
start.time <- Sys.time()
fig_2_2()
end.time <- Sys.time()
time.taken <- end.time - start.time
time.taken


# EXPERIMENT START PARALLEL ----
library(parallel)

# Updated run_bandits with parallel execution over runs ----
run_bandits <- function(bandits, n_runs, n_steps) {
  n_bandits <- length(bandits)
  rewards_array <- array(0, dim = c(n_bandits, n_runs, n_steps))
  optimal_action_freqs <- array(0, dim = c(n_bandits, n_runs, n_steps))
  
  # Determine the number of cores to use
  n_cores <- detectCores() - 1
  
  for (b in seq_len(n_bandits)) {
    # Parallel loop over runs for bandit b
    run_results <- mclapply(seq_len(n_runs), function(run) {
      bandits[[b]]$reset()
      rewards_run <- numeric(n_steps)
      optimal_run <- numeric(n_steps)
      for (step in seq_len(n_steps)) {
        res <- bandits[[b]]$step()
        rewards_run[step] <- res$reward
        if (res$action == which.max(bandits[[b]]$q_true)) {
          optimal_run[step] <- 1
        }
      }
      list(rewards = rewards_run, optimal = optimal_run)
    }, mc.cores = n_cores)
    
    # Collect the results from the parallel loop
    for (run in seq_len(n_runs)) {
      rewards_array[b, run, ] <- run_results[[run]]$rewards
      optimal_action_freqs[b, run, ] <- run_results[[run]]$optimal
    }
  }
  
  avg_rewards <- apply(rewards_array, c(1, 3), mean)
  avg_optimal_action_freqs <- apply(optimal_action_freqs, c(1, 3), mean)
  
  list(avg_rewards = avg_rewards, avg_optimal_action_freqs = avg_optimal_action_freqs)
}

# Figure 2.2: Average performance of ε-greedy methods using ggplot ----
fig_2_2 <- function(runs = 2000,
                    steps = 1000,
                    epsilons = c(0, 0.01, 0.1)) {
  bandits <- lapply(epsilons, function(eps) BaseBandit$new(eps = eps))
  res <- run_bandits(bandits, runs, steps)
  avg_rewards <- res$avg_rewards  # rows: bandits, columns: steps
  avg_optimal <- res$avg_optimal_action_freqs
  
  df_rewards <- data.frame(
    step = rep(1:steps, times = length(epsilons)),
    avg_reward = as.vector(t(avg_rewards)),
    epsilon = factor(rep(epsilons, each = steps))
  )
  
  p1 <- ggplot(df_rewards, aes(x = step, y = avg_reward, colour = epsilon)) +
    geom_line() +
    labs(title = "Figure 2.2 (Top): Average Reward", x = "Steps", y = "Average Reward")
  
  df_optimal <- data.frame(
    step = rep(1:steps, times = length(epsilons)),
    optimal = as.vector(t(avg_optimal)),
    epsilon = factor(rep(epsilons, each = steps))
  )
  
  p2 <- ggplot(df_optimal, aes(x = step, y = optimal, colour = epsilon)) +
    geom_line() +
    labs(title = "Figure 2.2 (Bottom): % Optimal Action", x = "Steps", y = "% Optimal Action")
  
  print(p1 / p2)
  print(p2)
  
  fig_num <- "2_2"
  filename <- file.path(paste0("figures/fig_", fig_num, ".png"))
  ggsave(filename = filename, p2)
}

# Run the figure
start.time <- Sys.time()
fig_2_2()
end.time <- Sys.time()
time.taken <- end.time - start.time
time.taken

# EXPERIMENT END PARALLEL ----

# Figure 2.3: The effect of optimistic initial action-value estimates ----
# on the 10-armed testbed using exponential average bandits.
# Both methods use a constant step-size parameter, step_size = 0.1.
fig_2_3 <- function(runs = 2000, steps = 1000, epsilons = c(0, 0.1), initial_qs = c(5, 0)) {
  bandits <- list()
  for (i in seq_along(epsilons)) {
    bandits[[i]] <- ExponentialAverageBandit$new(eps = epsilons[i],
                                                 initial_q = initial_qs[i],
                                                 step_size = 0.1)
  }
  res <- run_bandits(bandits, runs, steps)
  avg_optimal <- res$avg_optimal_action_freqs  # matrix: rows=bandits, columns=steps
  
  # Create a dataframe for plotting
  df <- data.frame(
    step = rep(1:steps, times = length(epsilons)),
    optimal = as.vector(t(avg_optimal)),
    label = factor(rep(paste0("Q1 = ", initial_qs, ", eps = ", epsilons),
                       each = steps))
  )
  
  p <- ggplot(df, aes(x = step, y = optimal, colour = label)) +
    geom_line() +
    labs(title = "Figure 2.3: Optimistic Initial Action-Value Estimates",
         x = "Steps", y = "% Optimal Action")
  
  print(p)
  fig_num <- "2_3"
  filename <- file.path(paste0("figures/fig_", fig_num, ".png"))
  ggsave(filename = filename)
}

start.time <- Sys.time()
fig_2_3()
end.time <- Sys.time()
time.taken <- end.time - start.time
time.taken
# Time difference of 1.791621 mins

# Figure 2.4: Average performance of UCB action selection vs. ε-greedy. ----
fig_2_4 <- function(runs = 2000, steps = 1000) {
  bandits <- list(
    UCBBandit$new(eps = 0, c = 2),
    BaseBandit$new(eps = 0.1)
  )
  res <- run_bandits(bandits, runs, steps)
  avg_optimal <- res$avg_optimal_action_freqs
  
  # Create labels for the two methods
  labels <- c(paste("UCB c =", bandits[[1]]$c),
              paste("ε-greedy ε =", bandits[[2]]$eps))
  
  df <- data.frame(
    step = rep(1:steps, times = 2),
    optimal = as.vector(t(avg_optimal)),
    label = factor(rep(labels, each = steps))
  )
  
  p <- ggplot(df, aes(x = step, y = optimal, colour = label)) +
    geom_line() +
    labs(title = "Figure 2.4: UCB vs. ε-greedy Performance",
         x = "Steps", y = "% Optimal Action")
  
  print(p)
  fig_num <- "2_4"
  filename <- file.path(paste0("figures/fig_", fig_num, ".png"))
  ggsave(filename = filename)
}

start.time <- Sys.time()
fig_2_4()
end.time <- Sys.time()
time.taken <- end.time - start.time
time.taken

# Figure 2.5: Average performance of the gradient bandit algorithm with and without a reward baseline. -----
fig_2_5 <- function(runs = 2000, steps = 1000) {
  bandits <- list(
    GradientBandit$new(step_size = 0.1, true_q_mean = 4, baseline = TRUE),
    GradientBandit$new(step_size = 0.4, true_q_mean = 4, baseline = TRUE),
    GradientBandit$new(step_size = 0.1, true_q_mean = 4, baseline = FALSE),
    GradientBandit$new(step_size = 0.4, true_q_mean = 4, baseline = FALSE)
  )
  res <- run_bandits(bandits, runs, steps)
  avg_optimal <- res$avg_optimal_action_freqs
  
  # Create labels based on bandit parameters
  labels <- sapply(bandits, function(b) {
    paste("step_size =", b$step_size, ", baseline =", b$baseline)
  })
  
  df <- data.frame(
    step = rep(1:steps, times = length(bandits)),
    optimal = as.vector(t(avg_optimal)),
    label = factor(rep(labels, each = steps))
  )
  
  p <- ggplot(df, aes(x = step, y = optimal, colour = label)) +
    geom_line() +
    labs(title = "Figure 2.5: Gradient Bandit Performance",
         x = "Steps", y = "% Optimal Action")
  
  print(p)
  
  fig_num <- "2_5"
  filename <- file.path(paste0("figures/fig_", fig_num, ".png"))
  ggsave(filename = filename)
}

start.time <- Sys.time()
fig_2_5()
end.time <- Sys.time()
time.taken <- end.time - start.time
time.taken
# Time difference of 5.332234 mins

