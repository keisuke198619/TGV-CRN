### Requirements
* python 3
* To install requirements:

```setup
pip install -r requirements.txt
```

### Preprocessing 
* The boid simulation can be performed at `./simulation` and the data is created in this folder.
* The CARLA simulation data can be download from [here](https://www.dropbox.com/sh/f356wip5ouwhbug/AAAF8nWRY2pf4Gsx7cuG3i4Ua?dl=0) and should be set in the folder `./datasets/TGV-CRN-carla`.
* The NBA data can be download from [here](https://www.dropbox.com/sh/k5whjbn5iiqxgag/AABAEvfivY3UFvllST3M5259a?dl=0) and should be set in the folder `./datasets/TGV-CRN-nba`.

### Main analysis
* see `run.sh` for commands using various datasets.
* Further details are documented within the code.

### Reference for a baseline model
- Deep Sequential Weighting (DSW): `https://github.com/ruoqi-liu/DSW`
