# TODO:
- afegir carpeta o algo amb el codi per fer els plots mes xulos
- afegir imatges aqui amb els plots xulos tipo el teaser
- afegir imatges de la demo despres de les linies de codi


# YCB-Affordance Dataset

## Introduction

## Prerequisites

Prerequisites are only needed for visualization. You can download the dataset ...

- Python requirements. Run `pip install -r requirements.txt`.

- **MANO layer**: Follow instructions from the MANO layer project in [here](https://raw.githubusercontent.com/hassony2/manopth).

## Download data

- **YCB object set CAD models**: Models from the YCB object set can be downloaded from [here](https://drive.google.com/open?id=1FdAWKpZTJBYctLNOZmlXGP7FGhE4etf0)

- **YCB-Affordance grasps**: Available from [here](https://drive.google.com/open?id=1dhnjeZxdqLqIWSkUS4bfTtdDU7omDyNO)

- OPTIONAL. **YCB-Video Dataset**: This provides the +133k multi-object scene images that combine with +28M grasps. If you are only interested in the grasps for the YCB object set CAD models, you don't need these. The YCB-Video Dataset can be downloaded from [here](https://drive.google.com/file/d/1if4VoEXNx9W3XCn0Y7Fp15B4GpcYbyYi/view?usp=sharing)


Downloaded data should go in ./data/ following:

```
data/
    models
    grasps
    YCBA...
```

## Visualization of grasps

### Load objects and grasps on CAD models

```
python -u visualize_grasps.py
```

### Load grasps on the YCB-Video dataset

```
python -u visualize_YCB_Affordance.py
```

## Citing

If this dataset is useful in your research, please cite:

```
bibtex missing
```

## License

The YCB-Affordance dataset is released under the MIT License

## Acknowledgements

missing



