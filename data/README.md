# PlantDoc data

The pipeline does not download data automatically. Obtain the Object Detection Dataset from
<https://github.com/pratikkayal/PlantDoc-Object-Detection-Dataset>, review its CC BY 4.0
attribution and terms, then place its original contents at `data/raw/plantdoc/`.

The expected layout is:

```text
data/raw/plantdoc/
  TRAIN/
  TEST/
  train_labels.csv
  test_labels.csv
```

The repository's size and exact contents can change, so check the source page before downloading.
Do not commit the downloaded images or derived image files. On Windows PowerShell:

```powershell
git clone --depth 1 https://github.com/pratikkayal/PlantDoc-Object-Detection-Dataset data/raw/plantdoc
tdet data inspect
tdet data prepare
```

`tdet data inspect` reports the actual filenames, source labels, image failures, malformed boxes,
and labels outside the nine configured tomato classes before `prepare` writes any processed data.
