# Ideas

### Loading and saving vanilla keras models
- Potentially possible to load the vanilla model using python and inject it into the TFKG format saved model. This would allow further training and inference.
- Potentially possible to save a vanilla model as a saved_model.pb before injecting it into the TFKG format and training, then saving the trained variables alongside the vanilla saved_model.pb

### Transfer learning
Would be dirty, but potentially possible to load a vanilla model in python, generate a model from TFKG configs, and transfer the weights over then train.

### Processors
- Investigate saving processors as a config file which can be automatically loaded by `data.NewInference`. Though it would only support readers and converters already present in the framework

### macOS pluggable device gpu support
In `tensorflow/c/c_api_experimental.h` there is a method `TF_LoadPluggableDeviceLibrary` to load pluggable device libraries. This could be used on macOS systems to accelerate training on compatible apple products

### Losses
Use a python object with differing functions for the loss types, pass the loss type key into the python generator and select an appropriate loss

### Custom layers
Allow custom python layers by offering a method to load the requisite python files

### Web server
- A tensorboard style ui for examining saved models with train/saved/test stats
- Queueing jobs with below training orchestration
- Hyperparameter tuning
 
### Training orchestration
- Create docker container for training with GPU acceleration
- Run docker container on all available machines for training
- Create api for accepting configurations and distributing jobs to available workers

# Todos

- Implement common layers
  - Implement full config spec of each layer
- Implement common losses
- Documentation
- Testing
- More real world examples
- Make logger and errorhandler interfaces so users can provide their own
- Add logic to catch possible errors before python compilation of model
- Move python code into a .py file and add it as a resource to the binary/go generate it
- Modify class weighting logic in the python file to allow for more than two classes
- Callback to save train/saved/test stats to database
- Optimise single file dataset with multiple concurrent readers to improve performance
- Save the keras model json on model creation, and load config to add correct layers into a TFKG model
- Automatically tailor metrics to different model losses
- When using model.Load, verify that the loaded model is in fact a compatible tfkg model
- Add image dataset
- Add more preprocessors: Images, Audio
- Allow custom python models to be used with a provided model generation python script
- Intelligent hyperparameter optimisation
- Optionally filter out tensorflow c logs if possible
- Log levels