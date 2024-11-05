#install.packages("R6")
library(R6)


# Diffusion ---------------------------------------------------------------

linear_beta_schedule <- function(timesteps) {
  scale <- 1000 / timesteps
  beta_start <- scale * 0.0001
  beta_end <- scale * 0.02
  seq(beta_start, beta_end, length.out = timesteps)
}

identity <- function(t, ...) {
  return(t)
}

LinearDiffusion <- R6Class("LinearDiffusion",
  public = list(
   image_size = NULL,
   timesteps = NULL,
   betas = NULL,
   alphas = NULL,
   alphas_cumprod = NULL,
   alphas_cumprod_prev = NULL,
   num_timesteps = NULL,
   sampling_timesteps = NULL,
   is_ddim_sampling = NULL,
   sqrt_alphas_cumprod = NULL,
   sqrt_one_minus_alphas_cumprod = NULL,
   sqrt_recip_alphas_cumprod = NULL,
   sqrt_recipm1_alphas_cumprod = NULL,
   posterior_variance = NULL,
   posterior_log_variance_clipped = NULL,
   posterior_mean_coef1 = NULL,
   posterior_mean_coef2 = NULL,
   models = NULL,
   
   initialize = function(image_size, timesteps) {
     self$image_size <- image_size
     self$timesteps <- timesteps
     self$betas <- linear_beta_schedule(timesteps)
     
     self$alphas <- 1 - self$betas
     self$alphas_cumprod <- cumprod(self$alphas)
     self$alphas_cumprod_prev <- c(1, self$alphas_cumprod[-length(self$alphas_cumprod)])
     
     self$num_timesteps <- timesteps
     self$sampling_timesteps <- timesteps
     self$is_ddim_sampling <- self$sampling_timesteps < self$timesteps
     
     self$sqrt_alphas_cumprod <- sqrt(self$alphas_cumprod)
     self$sqrt_one_minus_alphas_cumprod <- sqrt(1 - self$alphas_cumprod)
     self$sqrt_recip_alphas_cumprod <- sqrt(1 / self$alphas_cumprod)
     self$sqrt_recipm1_alphas_cumprod <- sqrt(1 / self$alphas_cumprod - 1)
     
     self$posterior_variance <- self$betas * (1 - self$alphas_cumprod_prev) / (1 - self$alphas_cumprod)
     self$posterior_log_variance_clipped <- log(pmax(self$posterior_variance, 1e-20))
     self$posterior_mean_coef1 <- self$betas * sqrt(self$alphas_cumprod_prev) / (1 - self$alphas_cumprod)
     self$posterior_mean_coef2 <- (1 - self$alphas_cumprod_prev) * sqrt(self$alphas) / (1 - self$alphas_cumprod)
     
     # Placeholder models for simplicity
     self$models <- list()#replicate(timesteps, lm(), simplify = FALSE)
   },
   
   predict_start_from_noise = function(x_t, t, noise) {
     self$sqrt_recip_alphas_cumprod[t + 1] * x_t - self$sqrt_recipm1_alphas_cumprod[t + 1] * noise
   },
   
   predict_noise_from_start = function(x_t, t, x0) {
     self$sqrt_recip_alphas_cumprod[t + 1] * x_t - x0 / self$sqrt_recipm1_alphas_cumprod[t + 1]
   },
   
   q_posterior = function(x_start, x_t, t) {
     posterior_mean <- self$posterior_mean_coef1[t + 1] * x_start + self$posterior_mean_coef2[t + 1] * x_t
     posterior_variance <- self$posterior_variance[t + 1]
     posterior_log_variance_clipped <- self$posterior_log_variance_clipped[t + 1]
     list(posterior_mean, posterior_variance, posterior_log_variance_clipped)
   },
   
   model_predictions = function(x, t, clip_x_start = FALSE) {
     model_output <- cbind(1, x) %*% self$models[[t + 1]]
     #model_output <- predict(self$models[[t + 1]], x)
     maybe_clip <- if (clip_x_start) function(x) pmax(-1, pmin(1, x)) else identity
     
     pred_noise <- model_output
     x_start <- self$predict_start_from_noise(x, t, pred_noise)
     x_start <- maybe_clip(x_start)
     
     list(pred_noise, x_start)
   },
   
   p_mean_variance = function(x, t, clip_denoised = TRUE) {
     preds <- self$model_predictions(x, t)
     x_start <- preds[[2]]
     
     if (clip_denoised) {
       x_start <- pmax(-1, pmin(1, x_start))
     }
     
     posterior <- self$q_posterior(x_start, x, t)
     list(posterior[[1]], posterior[[2]], posterior[[3]], x_start)
   },
   
   p_sample = function(x, t, clip_denoised = TRUE) {
     mean_variance <- self$p_mean_variance(x, t, clip_denoised)
     model_mean <- mean_variance[[1]]
     model_log_variance <- mean_variance[[3]]
     noise <- if (t > 0) rnorm(length(x)) else 0
     pred_img <- model_mean + exp(0.5 * model_log_variance) * noise
     list(pred_img, mean_variance[[4]])
   },
   
   p_sample_loop = function(shape, return_all_timesteps = FALSE, clip_denoised = TRUE) {
     img <- matrix(rnorm(prod(shape)), nrow = shape[1])
     imgs <- list(img)
     
     for (t in rev(seq_len(self$num_timesteps)-1)) {
       sample <- self$p_sample(img, t, clip_denoised)
       img <- sample[[1]]
       imgs <- append(imgs, list(img))
     }
     
     if (!return_all_timesteps) {
       return(img)
     } else {
       do.call(cbind, imgs)
     }
   },
   
   sample = function(shape, return_all_timesteps = FALSE, clip_denoised = TRUE) {
     self$p_sample_loop(shape, return_all_timesteps, clip_denoised)
   },
   
   q_sample = function(x_start, t) {
     noise <- array(rnorm(length(x_start)), dim = dim(x_start))
     x_sample <- self$sqrt_alphas_cumprod[t + 1] * x_start + self$sqrt_one_minus_alphas_cumprod[t + 1] * noise
     list(x=x_sample, noise=noise)
   },
   
   train = function(x_start, t) {
     sampled <- self$q_sample(x_start, t)
     # Fit the model for the current timestep t
     self$models[[t + 1]] <- lm.fit(x = cbind(1, sampled$x), y = sampled$noise)$coefficients #lm(noise ~ x, data=sampled)
   }
  )
)



# Example -----------------------------------------------------------------
#install.packages("dslabs")  # Uncomment to install if necessary
#install.packages("progress")
#install.packages("irlba")
library(Matrix)
library(irlba)
library(progress)
library(dslabs)
library(caret)
library(ggplot2)
library(gridExtra)


# Settings
image_size <- 28
timesteps <- 100
latent_size <- 9

# Dataset
mnist <- tryCatch({
  get.mnist = function(){
    mnist<-read_mnist(
      path = ".",
      download = TRUE,
      destdir = ".",
      url = "https://www2.harvardx.harvard.edu/courses/IDS_08_v2_03/",
      keep.files = TRUE
    )
    mnist$train$images <- mnist$train$images / 255.
    mnist$test$images <- mnist$test$images / 255.
    return(mnist)
  }
  get.mnist()
  }, 
  error = function(e) {
    get.mnist = function(){
      mnist<-read_mnist(
        path = NULL,
        download = TRUE,
        destdir = ".",
        url = "https://www2.harvardx.harvard.edu/courses/IDS_08_v2_03/",
        keep.files = TRUE
      )
      mnist$train$images <- mnist$train$images / 255.
      mnist$test$images <- mnist$test$images / 255.
      return(mnist)
    }
    get.mnist()
  })

images_flat <- mnist$train$images
if (is.matrix(images_flat)) {
  colnames(images_flat) <- paste0("V", seq_len(ncol(images_flat)))
} else if (is.data.frame(images_flat)) {
  names(images_flat) <- paste0("V", seq_len(ncol(images_flat)))
}

# Standardize the images
images_scaled <- (images_flat * 2) - 1

# Apply PCA 
image_encoder <- irlba::prcomp_irlba(images_scaled, n = latent_size, center = FALSE, scale. = FALSE)
image_encoded <- image_encoder$x

# Define Model
ldm <- LinearDiffusion$new(image_size, timesteps)

# Train Model
pb <- progress_bar$new(
  format = "training model [:bar] :percent in :elapsed",
  total = timesteps,
  clear = FALSE,
  width = 60
)
for (i in 0:(timesteps-1)){
  pb$tick()
  ldm$train(image_encoded, i)
}
  

# Generate sample
{
  rows <- 3
  cols <- 5
  sample_size <- rows * cols
  
  sample <- ldm$sample(c(sample_size, latent_size), clip_denoised = FALSE)
  
  # Inverse transformation
  sample <- as.matrix(sample) %*% t(image_encoder$rotation)
  #if (length(image_encoder$center) > 0) {
  #  sample <- sweep(sample, 2, image_encoder$center, "+")
  #}
  #if (length(image_encoder$scale) > 0) {
  #  sample <- sweep(sample, 2, image_encoder$scale, "*")
  #}
  sample <- (sample + 1) / 2
  
  # Plot
  sample <- array(sample, dim = c(sample_size, 28, 28))
  plot_list <- list()
  for (i in 1:sample_size) {
    # Convert image to a data frame for ggplot
    image_df <- as.data.frame(as.table(sample[i,,]))
    names(image_df) <- c("x", "y", "intensity")
    
    # Create a ggplot object for each image
    p <- ggplot(image_df, aes(x = x, y = y, fill = intensity)) +
      geom_raster() +
      scale_fill_gradient(low = "black", high = "white") +
      theme_void() +
      theme(legend.position = "none") +
      coord_fixed(ratio = 1)
    
    plot_list[[i]] <- p
  }
  grid.arrange(grobs = plot_list, nrow = rows, ncol = cols)
}




