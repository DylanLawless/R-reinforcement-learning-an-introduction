library(ggplot2)

# Create a synthetic dataset for a static frame
synthetic_data <- data.frame(
  x = 0,
  theta = 0.2,
  cart_left = 0 - 0.5,
  cart_right = 0 + 0.5,
  cart_bottom = 0,
  cart_top = 0.2,
  final_status = "Synthetic trial: pole balanced for 50 steps"
)

# Static ggplot similar to the final frame of the animation
ggplot(synthetic_data, aes(x = x, y = cart_top)) +
  geom_rect(aes(xmin = cart_left, xmax = cart_right,
                ymin = cart_bottom, ymax = cart_top),
            fill = "blue", colour = "black") +
  geom_segment(aes(xend = x + 1.5 * sin(theta),
                   yend = cart_top + 1.5 * cos(theta)),
               size = 2, colour = "red") +
  coord_fixed(xlim = c(synthetic_data$x - 2, synthetic_data$x + 2),
              ylim = c(0, 4)) +
  labs(title = synthetic_data$final_status,
       subtitle = "Static view of a synthetic cart-pole frame") +
  theme_minimal() +
  theme(plot.background = element_rect(fill = "white"))

