# Report

Head fine-tuning more stable since by freezing the features it makes impossible to forget those. 
There fewer parameters to update which makes it faster. In addition it is less prone to overfitting.
In our scenario there 50 new sample while the model is trained to 200 samples, doing a full fine tuning on such a small
sample would lead to overfitting which means memorization instead of generalization and high variance and consequently 
perform bad on unseen data. If we had a larger new dataset it would make sense to do full fine-tuning, or a different data preprocessing
and feature engineering would be necessary. Head fine tuning on the other hand allow the integration of the new sparse 
data without disrupting the training with the historical data. Since we fine tune only the output layer, the model 
learns the new dependencies more efficiently by being computationally efficient and having generalization stability.

# Project architecture

The project is split in different steps:
1. Data generation
   1. Generate initial data
   2. Generate the sparse data
2. Model training
   1. Train an MLP on initial datset
   2. Save the newly trained model weights
3. Model fine-tuning
   1. Load the pretrained model
   2. Freeze the feature extractor
   3. Fine tune on sparse dataset
   4. Save the newly adapted model
4. Evaluation
   1. Run both base and fine-tuned models
   2. Compare by MSE to see the effect of the new data

Each step can be reused if you want to train a completely new model or refine a different one.
A dockerfile is added to make the project reproducible in different environments.