# About the Pipeline

The pipeline uses Lora with 4 bit precision and the adam optimizer. It takes the question, reference_answer and the provided_answer with the label normalized_grade and formulates a prompt-text for the modell to train on (only llama 3.1 works this way). The base_trainer abstracts the training into methods to always use.

# Configure the Pipeline

Configure the global config to suit your needs.

Atm downloading new models is not fully supported since you need to login via the HG CLI. And I (Kevin) only am the one which requested access to the meta modell families.

# Start the Pipeline

First activate the virtual python environment in the CLI:

`source ~/projektarbeit/asag_env/bin/activate`

Then activate this option in the CLI:

`export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 

(stopps hard memory segmentation and improves memory management)

In the projektarbeit directory use this command in the CLI:

`python -m pipeline.pipeline_controller`

The pipeline should start now and do the following steps:

1. Download the modell sepcified in the global_config. If the modell is already on disk it will not download anything.
2. It will load the data from the global_config and augment the data specified in the dataloader if it is activated in the global_config
3. It will start the trainer of the corresponding modell specified in the global_config
    3.1 It will train, validate adn test n epochs


## Beobachtungen

Loss ist sehr gering - wahrscheinlich weil wir auf einer Nromalized Skala arbeiten. 
Ich habe einfmal versucht denn loss um einen Faktor zu erhöhen, dass hat zu leichten Verbesserung des MSE geführt.
