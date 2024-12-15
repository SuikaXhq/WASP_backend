# WASP Backend

This repo is for the backend server of [WASP](https://github.com/HarryWangHC/WASP_PRO).

## Requirements

- Python >= 3.9
- Numpy < 2.0
- PyTorch == 2.0.1+cu118
- opencv_python == 4.7.0.72
- flask >= 3.0.3
> Other requirements is in `requirements.txt`

## Usage

- To start the server:
  ```bash
  python app.py
  ```

- To use API:
  - Send HTTP POST request to `127.0.0.1:5000` with form data `image_url: <image_url>`.
  - Response would be a dict consisting outputs from all types of models.