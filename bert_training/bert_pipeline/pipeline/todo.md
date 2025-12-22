# Todo

# Configs

## Global Config

- [ ] Check if the options which we offer actually work
    Meaning if I change smth like precision, etc.

- [ ] Increase options:
    Output path of the model
    There are some options hidden in the script as constants atm - need to put those into a config (global or module)

- [ ] Introduce an option to skip training and use baseline modell

## Modell-Loader Config

- [ ] Check if the options we offer actually work

## Datalodaer Config

- [ ] Check if the options which we offer actually work

## Trainer Config

- [ ] Check if the options which we offer actually work
  
- [ ] Validation is currently shown in the validation but I dont know if it acutally validates correctly. Also as far as I now in this step we try to find the best hyperparameters. I dont know if we currently do that.

# Modules

## Controller

- [ ] Unify syntax - currently there are many parameters which dont say for what they stand.
  - [ ] Also there are some option which are set in the controller which need to be abstracted and put into the global config
  - [ ] 

## Modell-Loader

- [ ] Can sercan run this easyly even though I have the token bound by my account (atm  have access to the meta-llama models had to request that)

- [ ] If there are fallbacks to ensure that the pipline does not just crash - need to intruduce a default value which is not confusing

## Dataloader

- [ ] Order the augmentation of data a bit better
    Maybe seperate python file

- [ ] Check if the current augmentations even work
    GPT jsut did them as en example and what i thought could be usefull ontop of my head

- [ ] If there are fallbacks to ensure that the pipline does    not just crash - need to intruduce a default value which is not confusing

## Base Trainer

- [ ] If there are fallbacks to ensure that the pipline does not just crash - need to intruduce a default value which is not confusing

## Llama Trainer

- [ ] If there are fallbacks to ensure that the pipline does not just crash - need to intruduce a default value which is not confusing
  
- [ ] Ensure that the trainer really trains the modell in a way we want (currently the low loss seems concerning)

## Logging

- [ ] Needs to logg training (loss, mse, etc.)
Logg the current configuration off the global_config and the other config

- [ ] Look into which data actually is worth logging.

- [ ] If there are fallbacks to ensure that the pipline does not just crash - need to intruduce a default value which is not confusing

## Plotter

- [ ] Design plotts which we need.

- [ ] If there are fallbacks to ensure that the pipline does not just crash - need to intruduce a default value which is not confusing

## Other

- [ ] Need to make login and token for HF easier. Currently need to do login via CLI and add the token there. Then you can download the modell.

- [ ] Need to make comments for important methods to understand what is happening.

- [ ] Humanize some code - there is some gpt magic
    Or understand the underlying python code (even though it uses complex libraries)

- [ ] There are still some #todo comments in the scripts - need to make those





