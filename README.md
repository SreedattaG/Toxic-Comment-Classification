# Toxic Comment Classification

This project detects toxic comments using deep learning. It processes text data, vectorizes comments, and trains a bidirectional LSTM model with pre-trained GloVe embeddings to classify comments as toxic or non-toxic.

## Features
- Uses GloVe word embeddings for better text representation.
- Implements a Bidirectional LSTM model for classification.
- Loads a pre-trained model if available; otherwise, trains a new one.
- Evaluates performance using accuracy and a confusion matrix.

## Requirements
- Python 3.x
- TensorFlow
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

Install dependencies using:
```sh
pip install tensorflow pandas numpy matplotlib seaborn scikit-learn
```

## Dataset
The program expects a `train.csv` file containing comment text and toxicity labels. The GloVe embeddings file (`glove.6B.100d.txt`) is also required for word vector initialization.

### **Handling Large Files**
Since GitHub has a file size limit of 25MB for uploads, we have split large files (`train.csv` and `glove.6B.100d.txt`) into smaller parts. Before running the program, you need to merge them back together.

#### **Merging `train.csv` Back Together**
If `train.csv` was split into multiple parts (`train_part_aa`, `train_part_ab`, etc.), you can merge them back using:
- **Mac/Linux:**
  ```sh
  cat train_part_* > train.csv
  ```
- **Windows (PowerShell):**
  ```powershell
  Get-Content train_part_* | Set-Content train.csv
  ```

#### **Merging `glove.6B.100d.txt` Back Together**
If the GloVe file was split into multiple parts (`glove_part_aa`, `glove_part_ab`, etc.), merge them back using:
- **Mac/Linux:**
  ```sh
  cat glove_part_* > glove.6B.100d.txt
  ```
- **Windows (PowerShell):**
  ```powershell
  Get-Content glove_part_* | Set-Content glove.6B.100d.txt
  ```

This ensures the program can properly access the full dataset and embedding file before running.

## Usage
Run the script with:
```sh
python main.py
```
The script will train the model if no pre-trained version is found. It then evaluates performance and visualizes results.

## Output
- Trained model saved as `toxicity.h5`
- Confusion matrix for classification performance

## License
MIT License  
Copyright (c) 2025 Sreedatta Gudapudi  

Permission is hereby granted, free of charge, to any person obtaining a copy  
of this software and associated documentation files (the "Software"), to use  
the Software **for personal or commercial purposes**, subject to the following conditions:  

- **Modification, redistribution, sublicensing, or creating derivative works** is strictly prohibited.  
- **Copies of the Software** may not be shared, published, or distributed without prior written permission.  

The above copyright notice and this permission notice shall be included in all  
copies or substantial portions of the Software.  

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR  
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,  
FITNESS FOR A PARTICULAR PURPOSE, AND NONINFRINGEMENT. IN NO EVENT SHALL THE  
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER  
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT, OR OTHERWISE, ARISING FROM,  
OUT OF, OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE  
SOFTWARE.
