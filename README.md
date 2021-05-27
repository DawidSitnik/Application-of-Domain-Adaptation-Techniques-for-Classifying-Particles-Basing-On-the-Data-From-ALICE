The corresponding project implements all the experiments described in the article: 
['Application of Domain Adaptation Techniques for Classifying Particles Basing On the Data From ALICE Experiments'](documents/Article_ALICE_PID_Dawid_Sitnik.pdf)

## Data
To request a data used in the experiments please contact: [Dawid Sitnik](mailto:sitnik.dawid@gmail.com?subject=[GitHub]%20ALICE%20Data%20Request) (sitnik.dawid@gmail.com)

## Reproducing The Experiments Described In The Article
### Creating Models
1. After getting access to the pickles with data, put production.pkl and simulation.pkl into data/pickles/training_data
2. Run my_scripts/raw_data_preprocess.py
3. Run training_models/train_source_model.py
4. Run **all** other training scripts from /training_models in any order. If you do not create all the models you will probably get errors in further steps.

*The training parameters can be changed in utils/config.py*

## Project Structure
- **data** - all the data created during experiments which consists of: 
  * pickles objects created during execution of scripts from /my_scripts
  * plots created during execution of scripts from /my_scripts
  * tables created during execution of scripts from /my_scripts
  * .pt objects with model weights created during creating models  
- **documents** - article, presentation and a master thesis
- **domain_adaptation** - classes and functions which are used in the models which implements domain adaptation
- **my_scripts** - scripts used for experiments
- **tools** - additional classes and functions for domain adaptation
- **training_models** - scripts used for training the models
- **utils** - utility functions for training the models and experiments

### Validating Models
1. Train **all** the models.
2. Run my_scripts/validate_models.py

### Comparing Classification With Adaptation Quality
1. Train **all** the models.
2. Run my_scripts/get_output_from_models_last_layer.py
3. Run my_scripts/validate_domain_adaptation.py
4. Run my_scripts/compare_classification_and_adaptation_quality.py

### Identifying Unadapted Particles
1. Train **all** the models.
2. Run my_scripts/get_output_from_models_last_layer.py
3. Run my_scripts/cluster_and_mark_unadapted_particles.py
4. Run my_scripts/plot_unadapted_particles.py

### Comparing Attributes Distribution In Production, Simulation and Perturbed Datasets
1. Run my_scripts/compare_datasets_distributions.py

### Compare Distribution of Wrongly Classified / Unadapted Particles With The Distribution of All The Particles.
1. Train **all** the models.
2. Run my_scripts/create_lists_of_wrongly_classified_particles.py
3. Run my_scripts/plot_some_particles_attributes.py
    

