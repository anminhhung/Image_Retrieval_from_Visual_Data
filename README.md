<h1>HCM AI CHALLENGE 2022 - Event Retrieval from Visual Data</h1>



+ Folder Structure

```
  Main-folder/
  │
  ├── config/ 
  │   ├── config.py - configuration
  │
  ├── data/ - default directory for storing input data
  │
  ├── data_loader/ - anything about data loading goes here
  │   └── data_loaders.py
  |
  ├── model/ - this folder contains any net of your project.
  │   ├── model.py
  │   ├── metric.py
  │   └── loss.py
  │
  ├── saved/
  │   ├── models/ - trained models are saved here
  │   └── log/ - default logdir for tensorboard and logging 
  │   └── submission/ -  submission file are saved here
  │
  ├── scripts/ - main function 
  │   └── pipeline.py
  │   └── OCR.py
  │   └── segment.py
  |
  ├── test/ - test functions
  │   └── run.py
  │   └── ...
  |
  ├── tools/ - open source are saved here
  │   └── detectron2 dir
  │   └── ...
  │  
  └── utils/ - small utility functions
      ├── util.py
      └── ...
```
