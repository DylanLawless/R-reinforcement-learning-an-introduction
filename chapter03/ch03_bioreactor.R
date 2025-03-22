library(ggplot2)
library(gganimate)
library(dplyr)
library(tidyr)

# Parameters
grid_size <- 3
goal_temp <- 15
max_temp <- 60
min_temp <- 0
timesteps <- 25
stirring_rate <- 0.3  # between 0 (no mix) and 1 (full blend with neighbours)

# Initialise grid
init_grid <- matrix(goal_temp, nrow = grid_size, ncol = grid_size)

# Neighbour averaging function
get_neighbours <- function(grid, x, y) {
  indices <- expand.grid(dx = -1:1, dy = -1:1) %>%
    filter(!(dx == 0 & dy == 0)) %>%
    mutate(nx = x + dx, ny = y + dy) %>%
    filter(nx >= 1, nx <= grid_size, ny >= 1, ny <= grid_size)
  
  mean(sapply(1:nrow(indices), function(i) grid[indices$ny[i], indices$nx[i]]))
}

# Simulation loop
grid_list <- list()
current_grid <- init_grid

for (t in 1:timesteps) {
  # Random heating: one random cell increases by 3 (capped at max_temp)
  hot_x <- sample(1:grid_size, 1)
  hot_y <- sample(1:grid_size, 1)
  current_grid[hot_y, hot_x] <- min(current_grid[hot_y, hot_x] + 3, max_temp)
  
  # Stirring: each cell mixes with its neighbours
  next_grid <- current_grid
  for (y in 1:grid_size) {
    for (x in 1:grid_size) {
      neighbour_avg <- get_neighbours(current_grid, x, y)
      next_grid[y, x] <- (1 - stirring_rate) * current_grid[y, x] + stirring_rate * neighbour_avg
      next_grid[y, x] <- max(min_temp, min(next_grid[y, x], max_temp))
    }
  }
  
  current_grid <- next_grid
  grid_df <- expand.grid(x = 1:grid_size, y = 1:grid_size)
  grid_df$value <- as.vector(t(current_grid))
  grid_df$timestep <- t
  grid_list[[t]] <- grid_df
}

# Combine data from all timesteps
all_data <- bind_rows(grid_list)

# Create a label based on temperature thresholds:
# "hot" if >16, "good" if between 14 and 16, "cold" if <14.
all_data <- all_data %>%
  mutate(temp_state = case_when(
    value > 16 ~ "hot",
    value < 14 ~ "cold",
    TRUE ~ "good"
  ))

# Plot and animate with parameter annotations and temperature state labels
p <- ggplot(all_data, aes(x = x, y = y, fill = value)) +
  geom_tile(colour = "grey80") +
  # Print the temperature value
  geom_text(aes(label = round(value, 1)), size = 3, vjust = -0.5) +
  # Print the qualitative state below the number
  geom_text(aes(label = temp_state), size = 3, vjust = 1.5) +
  scale_fill_gradientn(
    colours = c("white", "blue", "green", "yellow", "red"),
    values = scales::rescale(c(min_temp, 10, goal_temp, 20, max_temp)),
    limits = c(min_temp, max_temp)
  ) +
  coord_fixed() +
  labs(
    title = "Bioreactor Temperature Grid",
    subtitle = "Timestep: {frame_time}",
    x = "",
    y = "",
    caption = paste0("Parameters: \nGrid size = ", grid_size,
                     "\n | Goal Temp = ", goal_temp,
                     "\n | Max Temp = ", max_temp,
                     "\n | Min Temp = ", min_temp,
                     "\n | Timesteps = ", timesteps,
                     "\n | Stirring rate = ", stirring_rate)
  ) +
  theme_minimal() +
  theme(legend.position = "right") +
  transition_time(timestep)

p

anim <- animate(p, nframes = timesteps, fps = 5, width = 500, height = 500)
anim_save("../figures/fig_ex_3_1_bioreactor_temp.gif", animation = anim)
