# Agatha Semantic Visualizer

Visualize hypotheses generated with AGATHA system to find new undiscovered knowledge.

![Agatha Semantic Visualizer working window][main_screenshot]

[main_screenshot]: https://github.com/IlyaTyagin/AgathaSemanticVisualizer/blob/main/ASV_image.png "AgathaHypothesisSpace main window"

Visualizer is a part of a bigger AGATHA project (current version: [AGATHA C/GP](https://github.com/IlyaTyagin/AGATHA-C-GP))

## How to install?

1. Clone this git repo: `git clone https://github.com/IlyaTyagin/AgathaSemanticVisualizer`
2. Install requirements: `pip install -r requirements.txt`
3. Download Semantic Groups description file: https://metamap.nlm.nih.gov/Docs/SemGroups_2018.txt and put it in `AgathaSemanticVisualizer` folder

## Project structure

___Agatha Semantic Visualizer___ includes 2 client(frontend) and 1 server (backend) applications.

Frontend applications:

* `AgathaHypothesisSpace.py` - shows interactive hypothesis space
* `CorpusExplorer.py` - shows what terms and sentences were used in a checkpoint

They both work with a checkpoint file directly and do not require any additional data.

Backend application:

* `Vis_topic_query.py`

It has a bunch of dependencies (most of them are AGATHA-dependant) and their description will be added later.
If you already have a checkpoint, backend part is not required.

## How to use?

### Frontend part (for end users)

First of all, you need to obtain a checkpoint file you want to visualize (you need to either contact me or generate it yourself from the AGATHA full model).

When the checkpoint is obtained, you need to go to `AgathaSemanticVisualizer/Checkpoints` directory and paste the name of the checkpoint inside the file `checkpoint_name.txt`. If your `Checkpoints` folder contains multiple checkpoints, the visualizer will read only the one mentioned in `checkpoint_name.txt`.

Then you need to run 2 two `bokeh` servers:

* `AgathaHypothesisSpace.py`:

  ```bokeh serve --show AgathaHypothesisSpace.py --port 5005```
  
* `CorpusExplorer.py`:

  ```bokeh serve --show CorpusExplorer.py --port 5006```

Bokeh servers are independant and can be used separately, there is no need to run them both if you need only one app.

### Backend part (how to generate hypothesis spaces from AGATHA models)

Description for this part will be added later.
