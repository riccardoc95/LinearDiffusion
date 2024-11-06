# Linear Diffusion Model in R

This repository is a fork of [willkurt/linear-diffusion](https://github.com/willkurt/linear-diffusion), originally in Python, with modifications and translation to R. It contains an implementation of a linear diffusion model using R6 classes in R. The model can be trained on images (e.g., MNIST dataset) and can generate samples based on the learned diffusion process. 

## Overview

The script defines a `LinearDiffusion` class that encapsulates the functionality for training and sampling from a linear diffusion model. It utilizes the R6 package for object-oriented programming in R and includes functions for linear beta scheduling, noise prediction, and sampling.

<p align="center">
  <img src="https://github.com/riccardoc95/LinearDiffusion/blob/main/sample.png" />
</p>

## Requirements

To run the script, you need to have R installed along with the following packages:

- R6
- Matrix
- irlba
- progress
- dslabs
- caret
- ggplot2
- gridExtra

You can install the required packages using the following command:

```r
install.packages(c("R6", "Matrix", "irlba", "progress", "dslabs", "caret", "ggplot2", "gridExtra"))
```

## Getting Started

### Installation

Clone the repository to your local machine:

```bash
git clone https://github.com/riccardoc95/LinearDiffusion
cd LinearDiffusion
```

### Usage

1. **Load the Script**: Open your R environment and source the script.

   ```r
   source("LinearDiffusionModel.R")
   ```

2. **Initialize the Model**: Specify the image size and the number of timesteps.

   ```r
   image_size <- 28
   timesteps <- 100
   ldm <- LinearDiffusion$new(image_size, timesteps)
   ```

3. **Train the Model**: Call the `train` method to train the model on the encoded image data.

   ```r
   for (i in 0:(timesteps - 1)) {
       ldm$train(image_encoded, i)
   }
   ```

4. **Generate Samples**: Use the `sample` method to generate new images based on the learned model.

   ```r
   sample <- ldm$sample(c(sample_size, latent_size), clip_denoised = FALSE)
   ```


### Example

The script includes an example section that demonstrates how to initialize the model, train it, and generate sample images. You can run this section to see the model in action.

## Contributing

If you would like to contribute to this project, feel free to fork the repository and submit a pull request. Please ensure that your code adheres to the project's coding standards.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The MNIST dataset is provided by Yann LeCun, Corinna Cortes, and Christopher J.C. Burges.
  
