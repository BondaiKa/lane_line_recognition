### Installation :hammer:

1) Install requirements

```bash
    pip install -r requirements.txt
```

2) Rename `.env.example` file to `.env`
3) Add download video and add path to `CAMERA_PATH` var
4) Add neural net weights and path to `NEURAL_NETWORK_WEIGHTS_MODEL_PATH` var

### Usage :gun:

To run application use command:

```bash
  python3 main.py
```

### Camera specification :scroll:

Name: Front Camera Basler daA1280-54uc

- Resolution (H x V pixels): **1280 x 960px**
- Pixel Size horizontal/vertical: **3.75 μm x 3.75 μm**
- FPS: **45** in Car (54 max.)
-

### FAQ :question:

- If issue appears while installing opencv. Just remove wheel lib from requirements.txt
- Current version requires connection to ithernet due to loading pretrained *InceptionResNetV2* model weights
