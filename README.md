# MPG Modeling Pipeline

This repository contains a pipeline for running an MPG (Miles Per Gallon) modeling API service. Follow these steps to set up and run the API locally:

==========================================
## Getting Started

1. **Navigate to the Root Directory:**

   Go to the root directory of this project.

2. **Run the Pipeline Script:**

   Execute the following command in your terminal:

   ```shell
   /bin/bash mpg_modeling_pipeline/run_pipeline.sh

## Using the Predict API

To use the predict API, follow these steps:

1. **Endpoint URL**:
   After running the pipeline, you can access the API service locally by clicking on the following URL in your terminal:

   http://127.0.0.1:5000

   You should see the message "The Server is Up" indicating that the server is running successfully.

2. **Predict Endpoint URL**:
  Use the following URL to access the predict API:
   http://127.0.0.1:5000/predict
         
   1. Input Format:

      The input of this endpoint is an array of feature sets, represented as JSON objects. For example:
   [
    {
        "cylinders": 8,
        "displacement": 307,
        "horsepower": 130,
        "weight": 3504,
        "acceleration": 12,
        "model-year": 70
    },
    {
        "cylinders": 4,
        "displacement": 120,
        "horsepower": 95,
        "weight": 2500,
        "acceleration": 15,
        "model-year": 75
    }
]
   2. Output Format:

      The output of this endpoint will be a JSON object containing predictions. It has the following structure:
      {
          "predictions": [
              15.43,
              25.95
          ]
      }


## Reflections and Potential Enhancements

In light of the time constraints, the current project implementation focuses on fundamental functionalities for training and prediction. Here are some reflections on the project and potential areas for improvement:

1. **Data Quality**:
   1. *Exploratory Data Analysis and Data Cleaning*: During the model fitting process, it became evident that the original dataset contained missing values. Rather than simply dropping these missing values (what I did), a more robust approach involves data imputation to preserve the integrity of the original dataset. Furthermore, conducting thorough Exploratory Data Analysis (EDA) before model fitting is essential to ensure data quality.
   2. *Input Validation*: The model itself lacks constraints and may yield nonsensical results (e.g., negative miles per gallon). Such outcomes can adversely affect the user experience of the API. Implementing input validation checks can ensure that input parameters are valid and within reasonable ranges. For example, verifying that the number of cylinders is greater than 0 and less than 20.

2. **Pipeline Enhancement**:
   1. *Flexible Function Parameters*: Currently, the `train_and_save_ridge_model` function accepts two file names as parameters, indicating where to save metrics and the trained model. In a production environment, enhancing this function to accept model hyperparameters (e.g., Alpha, cross-validation, selected model, test split size) would provide greater flexibility. Additionally, incorporating input parameter validation ensures that the format aligns with expectations.
   2. *Coding Improvements*: Several coding improvements are possible, such as the addition of documentation tests (doc tests) for better code documentation and addressing potential security concerns associated with bypassing SSL.

3. **Unit Testing**:
   1. To improve code organization, it's advisable to separate unit tests for the pipeline and class into two distinct test classes.
   2. While the current unit tests cover basic functionality, more comprehensive testing, especially for the predict function in the API, is essential. Creating dedicated testing data, rather than relying on the original dataset, is recommended to achieve better test coverage.
   3. Expanding unit testing to include corner cases would enhance the robustness of the testing suite. I would like to create my own unit test data to train the model so that I can imitate more difference scenario.

4. **Project Management**:
   1. Consider splitting the pipeline and API into separate repositories and managing them independently. This approach can lead to better organization and facilitate more focused development efforts.

Again, there is much improvement space left space. These reflections and potential enhancements can contribute to the project's robustness, maintainability, and user experience.
