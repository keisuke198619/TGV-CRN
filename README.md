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

### Counterfactual prediction in a CARLA simulation (car: red)
<video style='max-width:700px' controls>
  <source src="https://github.com/keisuke198619/TGV-CRN/assets/18593155/b39493e8-6a16-4b98-9760-3737b4f500de" type="video/mp4">
</video>

https://github.com/keisuke198619/TGV-CRN/assets/18593155/b39493e8-6a16-4b98-9760-3737b4f500de

### Ground truth in a CARLA simulation 
<video style='max-width:700px' controls>
  <source src="https://github.com/keisuke198619/TGV-CRN/assets/18593155/2a7ad422-34c9-4692-847a-60c2df0b9d45" type="video/mp4">
</video>

https://github.com/keisuke198619/TGV-CRN/assets/18593155/2a7ad422-34c9-4692-847a-60c2df0b9d45

### Counterfactual prediction in a boid simulation 
<video style='max-width:700px' controls>
  <source src="https://github.com/keisuke198619/TGV-CRN/assets/18593155/17ff4959-4199-438b-b804-9a30bb860a45" type="video/mp4">
</video>

https://github.com/keisuke198619/TGV-CRN/assets/18593155/17ff4959-4199-438b-b804-9a30bb860a45

### Ground truth in a boid simulation 
<video style='max-width:700px' controls>
  <source src="https://github.com/keisuke198619/TGV-CRN/assets/18593155/3045eff0-5023-467f-be52-c9237edee354" type="video/mp4">
</video>

### Counterfactual prediction (magenta and cyan) and ground truth (red and blue) in NBA dataset
<video style='max-width:700px' controls>
  <source src="https://github.com/keisuke198619/TGV-CRN/assets/18593155/337bfc25-b390-426b-a552-509e78ab9171" type="video/mp4">
</video>
  
https://github.com/keisuke198619/TGV-CRN/assets/18593155/337bfc25-b390-426b-a552-509e78ab9171

### Reference for a baseline model
- Deep Sequential Weighting (DSW): `https://github.com/ruoqi-liu/DSW`
