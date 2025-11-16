# honours_cool_project
A research project on logical skill composition

#### How to run code:
- make sure you have a version of python >= 3.12 
- make sure that all the u have installed all the necessary packages from the requirements.txt file in this repo
- The are three main files to consider:
  - _transformer_pipeline.ipynb_, which is a notebook used to train the transformer based agent,         simple change the variable in the if '__name__' == '__main__' section to the desired               cofiguration and run the cell. make sure the train parameter on the main method is set to          True
  - _mlp_baseline.ipynb_, which is also a notebook, and its use to train the mlp based when the         train argument is True, when its False then post-training evaluation of both the transformer       and mlp basedagents are evaluated across there different seeds, so make sure you have models       alreadytraine before evaluation
  - _PlotLearnigCurve.ipynb, which is used to extract trainig related data for evaluation and           creating learnign curves      
