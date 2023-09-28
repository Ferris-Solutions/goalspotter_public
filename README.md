# GoalSpotter: A Sustainability Objective Detection System

Sustainable development is nowadays a prominent factor for the public. As a result, companies publish their sustainability visions and strategies in various reports to show their commitment to saving the environment and promoting social progress. However, not all statements in these sustainability reports are fact based. When a company tries to mislead the public with its non-fact-based sustainability claims, greenwashing happens. To combat greenwashing, society needs effective automated approaches to identify the sustainability claims of companies in their heterogeneous reports. 

In this project, we build a new sustainability objective detection system that automatically identifies the environmental and social claims of companies in their heterogeneous reports. Our system extracts text blocks of diverse reports, preprocesses and labels them using domain expert annotations, and then fine-tunes transformer models on the labeled text blocks. This way, our system can detect sustainability objectives in any new unannotated heterogeneous report. As our experiments show, our system outperforms existing state-of-the-art sustainability objective detection approaches.


# Installation
After cloning the repository, go into the folder and run the following command:    
```pip3 install -e .```


# Usage
The `source` folder contains source codes of the essential classes.
- `document.py` defines the `Document` class to represent each document (e.g., a pdf or html file).
- `data_preprocessing.py` defines the `DataPreprocessing` class to preprocess the text of documents.
- `transformer_model.py` defines the `TransformerModel` class to build a transformer model.

The `notebooks` folder contains interactive notebooks for various purposes.
- `dataset_construction.ipynb` creates our labeled dataset using domain expert annotations.
- `detection_modeling.ipynb` trains a transformer model to detect text blocks that contain a sustainability objective.
- `comparison_with_baselines.ipynb` compares our system with the baseline approaches.
- `demo.ipynb` demonstrates our system on a new sustainability report.
- `objective_extraction.ipynb` applies our objective detection system to the new sustainability reports to extract new objectives.
