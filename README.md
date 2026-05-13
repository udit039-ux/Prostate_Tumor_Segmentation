# Prostate Tumor Segmentation

This project implements a deep learning-based segmentation model for prostate tumors using the MONAI framework. It utilizes the Task05_Prostate dataset from the Medical Segmentation Decathlon for training and evaluation.

## Features

- Data preprocessing for medical images (DICOM to NIfTI conversion)
- 3D U-Net model training for tumor segmentation
- Evaluation metrics and visualization
- Jupyter notebook for testing and experimentation

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/udit039-ux/Prostate_Tumor_Segmentation.git
   cd Prostate_Tumor_Segmentation
   ```

2. Create a virtual environment:
   ```bash
   python -m venv env
   # On Windows:
   env\Scripts\activate
   # On macOS/Linux:
   source env/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Dependencies

- matplotlib
- numpy
- tqdm
- glob2
- dicom2nifti
- pytest-shutil
- nibabel
- torch
- monai

## Usage

### Preprocessing

Run the preprocessing script to prepare the data:
```bash
python preprocess.py
```

### Training

Train the segmentation model:
```bash
python train.py
```

### Testing

Use the Jupyter notebook for testing and visualization:
```bash
jupyter notebook testing.ipynb
```

## Dataset

This project uses the Task05_Prostate dataset from the Medical Segmentation Decathlon. The dataset includes:
- Training images and labels in `Task05_Prostate/imagesTr/` and `Task05_Prostate/labelsTr/`
- Test images in `Task05_Prostate/imagesTs/`

## Results

Trained models and evaluation metrics are saved in the `Results/` directory:
- `final_model.pth`: The trained PyTorch model
- `loss_train.npy`: Training loss history
- `metric_train.npy`: Training metrics

## Project Structure

- `preprocess.py`: Data preprocessing utilities
- `train.py`: Model training script
- `utilities.py`: Helper functions
- `testing.ipynb`: Jupyter notebook for testing
- `requirements.txt`: Python dependencies
- `Task05_Prostate/`: Dataset directory
- `Results/`: Output directory for models and metrics

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.