# Animal Image Classification using CNN

A Convolutional Neural Network (CNN) that classifies animal images into the categories represented by the image folders. The original notebook was designed around a 90-category animal image dataset.

## Dataset
Download the 90 Animal Image Dataset and arrange it so that each animal has its own folder:

```text
data/animals/
├── cat/
│   ├── image1.jpg
│   └── image2.jpg
├── dog/
│   ├── image1.jpg
│   └── image2.jpg
└── ...
```

The folder names become the class labels. Keep the dataset private or use the dataset provider's permitted sharing method; do not commit a large dataset to GitHub unless its license allows it.

## Run
```bash
pip install -r requirements.txt
python main.py
```

## Example output format
```text
Found [image_count] files belonging to [class_count] classes.
Using [training_count] files for training.
Found [image_count] files belonging to [class_count] classes.
Using [validation_count] files for validation.
Number of animal classes: [class_count]
Epoch 1/5
...
Validation loss: [loss]
Validation accuracy: [accuracy]
Saved model as animal_classifier.keras
Saved labels as class_names.json
```
Training time and accuracy depend on the dataset, hardware and number of epochs. The script discovers the number of classes from the folders rather than assuming it is always 90.

## Repository structure
```text
animal-image-classification-cnn/
├── data/
│   └── animals/
│       ├── animal_class_1/
│       └── animal_class_2/
├── main.py
├── requirements.txt
└── README.md
```
