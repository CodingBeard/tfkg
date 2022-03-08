# Ideas

### Saving vanilla keras models
- Potentially possible to save a vanilla model as a saved_model.pb before injecting it into the TFKG format and training, then saving the trained variables alongside the vanilla saved_model.pb

### Transfer learning from vanilla tensorflow models
Would be dirty, but potentially possible to load a vanilla model in python, generate a model from TFKG configs, and transfer the weights over then train.

### Processors
- Investigate saving processors as a config file which can be automatically loaded by `data.NewInference`. Though it would only support readers and converters already present in the framework

### macOS pluggable device gpu support
In `tensorflow/c/c_api_experimental.h` there is a method `TF_LoadPluggableDeviceLibrary` to load pluggable device libraries. This could be used on macOS systems to accelerate training on compatible apple products

### Losses
Use a python object with differing functions for the loss types, pass the loss type key into the python generator and select an appropriate loss

### Web server
- Queueing jobs with below training orchestration
- Hyperparameter tuning
 
### Training orchestration
- Create docker container for training with GPU acceleration
- Run docker container on all available machines for training
- Create api for accepting configurations and distributing jobs to available workers

# Todos

- Implement common losses
- Documentation
- Testing
- More real world examples
- Make logger and errorhandler interfaces so users can provide their own
- Add logic to catch possible errors before python compilation of model
- Callback to save train/saved/test stats to database
- Save the keras model json on model creation, and load config to add correct layers into a TFKG model
- Automatically tailor metrics to different model losses
- Add more preprocessors: Video, Audio
- Intelligent hyperparameter optimisation
- Optionally filter out tensorflow c logs if possible
- Log levels
- Method on dataset to validate if it meets the expected configuration 